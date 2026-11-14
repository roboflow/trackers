# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Reproducible manual lifecycle replay for the shared EdgeTAM backend."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from trackers.cli._annotate import annotate_masks, draw_status_lines
from trackers.cli.inspect._common import (
    INSPECT_OUTPUT_ROOT,
    get_mask_tracklet_ids_in_order,
    list_selected_frame_paths,
    load_rgb_image,
    save_rgb_image,
    timestamped_run_dir,
    validate_and_clip_xyxy_box,
)
from trackers.cli.inspect._mask_lifecycle import (
    group_add_events_by_next_frame,
    group_remove_events,
    parse_add_mask_event,
    parse_remove_mask_event,
    validate_lifecycle_events,
)
from trackers.core.masks.base import MaskOutput, TrackletSnapshot
from trackers.core.masks.edgetam import DEFAULT_EDGETAM_MODEL

if TYPE_CHECKING:
    from trackers.core.masks.edgetam import EdgeTAMMaskBackend


def _save_frame(
    frame: np.ndarray,
    output: MaskOutput | None,
    output_path: Path,
    backend: EdgeTAMMaskBackend,
    rollover: bool,
) -> list[int]:
    """Draw independent mask layers and the session position on one image."""
    ids = [] if output is None else get_mask_tracklet_ids_in_order(output.tracklet_mask_dict)
    visual = frame.copy()
    if output is not None and output.masks is not None and len(output.masks):
        visual = annotate_masks(visual, output.masks, ids)
    index = backend.session_frame_count - 1 if backend.session_frame_count else None
    status = [
        output_path.name,
        f"Manual replay | IDs: {ids}",
        f"Session {backend.session_index} | local frame {index}" + (" | rollover" if rollover else ""),
    ]
    save_rgb_image(draw_status_lines(visual, status), output_path)
    return ids


def _versions() -> dict[str, str | None]:
    """Record the installed inference dependencies without importing them."""
    result: dict[str, str | None] = {}
    for package in ("trackers", "transformers", "torch", "torchvision"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def edgetam_inspection(
    image_dir: Path,
    start_file: str,
    end_file: str,
    box: list[tuple[float, float, float, float]],
    add_at: list[str] | None = None,
    remove_at: list[str] | None = None,
    output_root: Path = INSPECT_OUTPUT_ROOT / "edgetam",
    device: str = "auto",
    model_id: str = DEFAULT_EDGETAM_MODEL,
    revision: str | None = None,
    max_session_frames: int | None = 200,
    mask_threshold: float = 0.0,
) -> int:
    """Inspect EdgeTAM creation, propagation, addition, removal and rollover.

    This is a manual mask replay, not a tracker or an accuracy evaluation.
    Saves annotated frames, a reusable ``config.json`` and ``run.json`` with
    frame hashes, model revision, dependency versions and lifecycle events.

    Only this inspection command lists the selected frame paths in advance,
    to validate and schedule manual events. The EdgeTAM backend receives frames
    one at a time and needs no future frames or final sequence length.

    Args:
        image_dir: Directory containing input images, selected in filename order.
        start_file: First frame filename, inclusive.
        end_file: Last frame filename, inclusive.
        box: Initial xyxy boxes, for example ``--box='[[10,20,110,220]]'``.
            IDs start at one and follow box order. An empty list starts inactive.
        add_at: New boxes as ``filename:x1,y1,x2,y2``. Each gets the next unused
            ID. The box is valid on that image and applied before propagation
            to the next selected image. This does not repair existing objects.
        remove_at: Removals as ``filename:manual_id``, applied before propagation
            to that image. Missing filename numbers do not advance the session.
        output_root: Root for timestamped output directories.
        device: PyTorch device; ``auto`` selects CUDA when available, else CPU.
        model_id: Transformers-compatible EdgeTAM checkpoint ID or local path.
        revision: Model and processor revision override. With the default model,
            ``None`` selects the pinned tested commit; with a custom model it
            leaves revision selection to Transformers. Use a commit to pin it.
        max_session_frames: Maximum stored images including the seed, at least
            two; ``None`` disables rollover. Defaults to 200 for every input.
        mask_threshold: Binary-mask logit threshold; not a confidence gate.

    Returns:
        Zero on success; one on invalid inputs, missing dependencies or inference
        failure. Partial runs keep their manifest with ``status="failed"``.
    """
    backend = None
    manifest: dict[str, Any] = {"status": "failed", "mode": "manual-mask-replay", "frames": []}
    output_dir = None
    exit_code = 0
    try:
        frame_paths = list_selected_frame_paths(image_dir, start_file, end_file)
        if len(frame_paths) < 2:
            raise ValueError("At least two frames are required for EdgeTAM propagation.")
        additions = [parse_add_mask_event(event) for event in add_at or []]
        removals = [parse_remove_mask_event(event) for event in remove_at or []]
        validate_lifecycle_events(frame_paths, additions, removals)
        additions_by_file = group_add_events_by_next_frame(frame_paths, additions)
        removals_by_file = group_remove_events(removals)
        initial_frame = load_rgb_image(frame_paths[0])
        tracklets = [
            TrackletSnapshot(index + 1, validate_and_clip_xyxy_box(value, initial_frame.shape[:2]))
            for index, value in enumerate(box)
        ]

        # Keep optional inference imports behind validation and CLI dispatch.
        from trackers.core.masks.edgetam import EdgeTAMMaskBackend

        backend = EdgeTAMMaskBackend(
            model_id=model_id,
            revision=revision,
            device=device,
            max_session_frames=max_session_frames,
            mask_threshold=mask_threshold,
        )
        output_dir = timestamped_run_dir(output_root)
        config = {
            "image_dir": str(image_dir.resolve()),
            "start_file": start_file,
            "end_file": end_file,
            "box": box,
            "add_at": add_at,
            "remove_at": remove_at,
            "output_root": str(output_root.resolve()),
            "device": str(backend.device),
            "model_id": model_id,
            "revision": backend.model_revision or revision,
            "max_session_frames": max_session_frames,
            "mask_threshold": mask_threshold,
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        manifest.update(
            config=config,
            requested_revision=revision,
            resolved_revision=backend.model_revision,
            versions=_versions(),
        )
        print(f"Manual mask replay on {backend.device}; saving to {output_dir}")
        initial_output = backend.generate(initial_frame, tracklets)
        backend.initialize(initial_frame, initial_output)
        output: MaskOutput | None = initial_output
        previous_frame = initial_frame
        next_id = len(tracklets) + 1
        for position, frame_path in enumerate(frame_paths):
            current_frame = initial_frame if position == 0 else load_rgb_image(frame_path)
            added_ids = []
            removed_ids = removals_by_file.get(frame_path.name, [])
            rollover = False
            if position:
                new_tracklets = []
                for event in additions_by_file.get(frame_path.name, []):
                    new_tracklets.append(
                        TrackletSnapshot(next_id, validate_and_clip_xyxy_box(event.xyxy, previous_frame.shape[:2]))
                    )
                    added_ids.append(next_id)
                    next_id += 1
                if new_tracklets:
                    backend.add_masks(previous_frame, backend.generate(previous_frame, new_tracklets))
                backend.remove_masks(removed_ids)
                previous_session_index = backend.session_index
                output = backend.propagate(current_frame)
                rollover = backend.session_index > previous_session_index
            ids = _save_frame(current_frame, output, output_dir / frame_path.name, backend, rollover)
            manifest["frames"].append(
                {
                    "file": frame_path.name,
                    "sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                    "session_index": backend.session_index,
                    "session_frame_index": backend.session_frame_count - 1 if backend.session_frame_count else None,
                    "session_frame_count": backend.session_frame_count,
                    "rollover": rollover,
                    "active_ids": ids,
                    "added_ids": added_ids if position else [tracklet.tracker_id for tracklet in tracklets],
                    "removed_ids": removed_ids,
                }
            )
            previous_frame = current_frame
        manifest["status"] = "completed"
    except (OSError, ImportError, ValueError, RuntimeError) as error:
        manifest["error"] = str(error)
        print(f"Error: {error}", file=sys.stderr)
        exit_code = 1
    finally:
        if backend is not None:
            backend.reset()
        if output_dir is not None:
            try:
                (output_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
            except OSError as error:
                print(f"Error: Could not write run manifest in {output_dir}: {error}", file=sys.stderr)
                exit_code = 1
    if exit_code:
        return exit_code
    print(f"Saved manual mask replay to {output_dir}")
    return 0
