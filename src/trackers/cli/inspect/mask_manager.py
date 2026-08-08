# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Visual sanity check for :class:`MaskManager` with SAM + Cutie.

Validates the real MaskManager orchestration:

- initialize masks from previous-frame tracklets,
- propagate masks to the current frame,
- add masks for new tracklets,
- remove masks for terminated tracklets.

``MaskManager.get_updated_masks()`` is called directly, not Cutie.

Two modes, selected with the required ``--mode``:

``manual``
    Tracklets come from boxes given on the command line, with lifecycle events
    scheduled by ``--add_at`` / ``--remove_at``. Good for constructing a precise
    scenario by hand.

``gt``
    Tracklets are replayed from a MOT-format ground-truth file. Good for
    watching delayed mask creation and pending tracklets on real data. This mode
    additionally draws lifecycle-aware box colors, because GT replay knows which
    tracklets are new, pending, or already masked.

Each mode reads its own options; the other mode's options are rejected rather
than silently ignored. Options shared by both (``--device``, the SAM and Cutie
model settings, ``--output_root``) apply either way.

Usage
-----

::

    trackers inspect mask-manager --mode manual \\
        --image_dir frames --start_file 000001.jpg --end_file 000010.jpg \\
        --box='[[10,20,110,220]]'

    trackers inspect mask-manager --mode gt \\
        --image_dir frames --gt_file gt.txt --start_frame 1 --end_frame 100

Options come from the :func:`mask_manager_command` signature, parsed with
jsonargparse through the shared ``trackers`` parser, so the shared conventions
hold: ``--image-dir`` and ``--image_dir`` are the same option, and ``--config
run.yaml`` supplies the same keys from a file. The repeatable options are
lists: boxes are ``--box='[[x1,y1,x2,y2]]'``, lifecycle events are
``--add_at='["frame.jpg:x1,y1,x2,y2"]'`` and ``--remove_at='["frame.jpg:3"]'``,
and tracklet selection is ``--tracklet_id='[3,7]'``. Every one of them also
appends with ``+``, as in ``--add_at+ frame.jpg:10,20,110,220``.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

# Imported at runtime, not under TYPE_CHECKING: jsonargparse resolves this
# command's annotations when it builds the parser.
from jsonargparse.typing import PositiveInt

from trackers.cli.inspect._common import (
    INSPECT_OUTPUT_ROOT,
    list_selected_frame_paths,
    load_rgb_image,
    parse_xyxy_box,
    save_rgb_image,
    timestamped_run_dir,
    validate_device,
)
from trackers.cli.inspect._mask_manager_gt import run_gt_mode
from trackers.core.masks.base import MaskOutput, TrackletSnapshot

DEFAULT_OUTPUT_ROOT = INSPECT_OUTPUT_ROOT / "mask_manager"

MaskManagerMode = Literal["manual", "gt"]

_MANUAL_ONLY_OPTIONS = ("start_file", "end_file", "box", "add_at", "remove_at")
_GT_ONLY_OPTIONS = ("gt_file", "start_frame", "end_frame", "tracklet_id")


@dataclass(frozen=True)
class AddTrackletEvent:
    """Manual new-tracklet event parsed from ``--add_at``."""

    frame_file: str
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class RemoveTrackletEvent:
    """Manual removed-tracklet event parsed from ``--remove_at``."""

    frame_file: str
    tracker_id: int


def parse_add_tracklet_event(event: str) -> AddTrackletEvent:
    """Parse ``filename:x1,y1,x2,y2`` add-tracklet event."""
    parts = event.split(":")
    if len(parts) != 2:
        raise ValueError("Add event must have format filename:x1,y1,x2,y2.")

    frame_file, box_str = parts
    try:
        xyxy = parse_xyxy_box(box_str)
    except ValueError as exc:
        raise ValueError("Add event must have format filename:x1,y1,x2,y2.") from exc

    return AddTrackletEvent(frame_file=frame_file, xyxy=xyxy)


def parse_remove_tracklet_event(event: str) -> RemoveTrackletEvent:
    """Parse ``filename:tracklet_id`` remove-tracklet event."""
    parts = event.split(":")
    if len(parts) != 2:
        raise ValueError("Remove event must have format filename:tracklet_id.")

    frame_file, tracker_id_str = parts
    try:
        tracker_id = int(tracker_id_str)
    except ValueError as exc:
        raise ValueError("Remove event must have format filename:tracklet_id.") from exc

    if tracker_id <= 0:
        raise ValueError("tracklet_id must be a positive integer.")

    return RemoveTrackletEvent(frame_file=frame_file, tracker_id=tracker_id)


def validate_and_clip_xyxy_box(
    box: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Validate and clip an ``xyxy`` box to image boundaries."""
    height, width = image_shape
    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid xyxy box with non-positive size: {box}")

    x1 = np.clip(x1, 0, width)
    x2 = np.clip(x2, 0, width)
    y1 = np.clip(y1, 0, height)
    y2 = np.clip(y2, 0, height)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Box is outside image after clipping: {box}")

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def group_add_events_by_source_frame(
    frame_paths: list[Path],
    events: list[AddTrackletEvent],
) -> dict[str, list[AddTrackletEvent]]:
    """Group add events by the frame where new tracklets are created.

    The grouped events are converted into ``previous_new_tracklets`` after the
    MaskManager call for that frame. They are then consumed by MaskManager on
    the next frame, matching the original McByte timing.
    """
    filenames = [path.name for path in frame_paths]
    grouped: dict[str, list[AddTrackletEvent]] = defaultdict(list)

    for event in events:
        source_index = filenames.index(event.frame_file)
        target_file = filenames[source_index]
        grouped[target_file].append(event)

    return dict(grouped)


def group_remove_events(events: list[RemoveTrackletEvent]) -> dict[str, list[int]]:
    """Group remove events by the frame where tracklets are terminated.

    The grouped events are converted into ``previous_removed_tracklet_ids`` after
    the MaskManager call for that frame. They are then consumed by MaskManager on
    the next frame, matching the original McByte timing.
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    for event in events:
        grouped[event.frame_file].append(event.tracker_id)
    return dict(grouped)


def validate_lifecycle_events(
    frame_paths: list[Path],
    add_events: list[AddTrackletEvent],
    remove_events: list[RemoveTrackletEvent],
) -> None:
    """Validate that add/remove events are compatible with the selected frame range."""
    filenames = [path.name for path in frame_paths]
    filename_set = set(filenames)

    for add_event in add_events:
        if add_event.frame_file not in filename_set:
            raise ValueError(f"Add event frame is outside selected range: {add_event.frame_file}")
        if add_event.frame_file == filenames[-1]:
            raise ValueError("Add events cannot be scheduled on the last selected frame.")

    for remove_event in remove_events:
        if remove_event.frame_file not in filename_set:
            raise ValueError(f"Remove event frame is outside selected range: {remove_event.frame_file}")
        if remove_event.frame_file == filenames[0]:
            raise ValueError("Remove events cannot be scheduled on the first selected frame.")


def color_from_id(object_id: int) -> np.ndarray:
    """Return a deterministic RGB color for a stable tracklet ID."""
    rng = np.random.default_rng(object_id)
    return rng.integers(0, 255, size=3, dtype=np.uint8)


def overlay_masks(
    image: np.ndarray,
    masks: np.ndarray,
    tracklet_ids: list[int],
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay binary masks on an RGB image.

    Each tracklet is rendered using a deterministic color derived from its
    tracklet ID. The ordering of ``tracklet_ids`` must match the ordering of
    ``masks``.
    """
    if masks.shape[0] != len(tracklet_ids):
        raise ValueError(
            "Number of masks must match number of tracklet IDs. "
            f"Got {masks.shape[0]} masks and {len(tracklet_ids)} tracklet IDs."
        )

    output = image.copy()
    for mask, tracklet_id in zip(masks, tracklet_ids):
        color = color_from_id(tracklet_id)
        colored_mask = np.zeros_like(output)
        colored_mask[mask] = color

        output = np.where(
            mask[..., None],
            (alpha * colored_mask + (1.0 - alpha) * output).astype(np.uint8),
            output,
        )

    return output


def get_mask_tracklet_ids_in_order(tracklet_mask_dict: dict[int, int]) -> list[int]:
    """Return tracklet IDs in the same order as ``MaskOutput.masks``."""
    return [
        tracklet_id
        for tracklet_id, _ in sorted(
            tracklet_mask_dict.items(),
            key=lambda item: item[1],
        )
    ]


def draw_boxes(image: np.ndarray, tracklets: list[TrackletSnapshot]) -> np.ndarray:
    """Draw tracklet bounding boxes and IDs on an RGB image."""
    output = image.copy()

    for tracklet in tracklets:
        x1, y1, x2, y2 = tracklet.xyxy.astype(int)
        cv2.rectangle(output, (x1, y1), (x2, y2), color=(91, 10, 145), thickness=2)
        cv2.putText(
            output,
            str(tracklet.tracker_id),
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def draw_frame_label(image: np.ndarray, frame_name: str, status: str) -> np.ndarray:
    """Draw the frame filename and lifecycle status in the top-left corner."""
    output = image.copy()
    label = f"{frame_name} | {status}"

    cv2.putText(output, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(output, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)

    return output


def visualize_output(
    frame: np.ndarray,
    frame_name: str,
    mask_output: MaskOutput | None,
    tracklets: list[TrackletSnapshot],
    output_path: Path,
    status: str,
) -> None:
    """Overlay masks, optional tracklet boxes, and frame status, then save the result."""
    visual = frame.copy()

    if mask_output is not None and mask_output.masks is not None:
        tracklet_ids = get_mask_tracklet_ids_in_order(mask_output.tracklet_mask_dict)
        visual = overlay_masks(
            image=visual,
            masks=mask_output.masks,
            tracklet_ids=tracklet_ids,
        )

    visual = draw_boxes(visual, tracklets)
    visual = draw_frame_label(visual, frame_name, status)
    save_rgb_image(visual, output_path)


def run_manual_mode(
    image_dir: Path,
    start_file: str,
    end_file: str,
    box: list[tuple[float, float, float, float]],
    output_root: Path,
    add_at: list[str] | None = None,
    remove_at: list[str] | None = None,
    device: str = "cuda",
    sam_model_type: str = "vit_b",
    cutie_model_type: str = "base-mega",
    cutie_config_path: Path | None = None,
    cutie_config_name: str = "eval_config",
) -> int:
    """Run MaskManager visual validation and save per-frame outputs.

    The script follows the original McByte execution order.

    For frame ``t``:

        1. MaskManager produces masks for frame ``t`` using tracker outputs
           from frame ``t-1``.
        2. The tracker (emulated by this script) produces tracklets for frame
           ``t`` using the masks from frame ``t`` (step 1).
        3. Newly created and removed tracklets are stored and become lifecycle
           events consumed by MaskManager on frame ``t+1``.

    This mirrors the timing used by the original McByte implementation.

    Every option is also accepted with hyphens in place of underscores, so
    ``--start-file`` and ``--start_file`` name the same thing.

    Args:
        image_dir: Directory holding the frame images.
        start_file: Filename of the first frame to process.
        end_file: Filename of the last frame to process.
        box: Initial tracklet boxes on the first selected frame, given as one
            list rather than a repeated option: ``--box='[[x1,y1,x2,y2]]'`` for
            one box, ``--box='[[10,20,110,220],[30,40,130,240]]'`` for two, and
            ``--box+='[[30,40,130,240]]'`` to append to boxes already given.
        add_at: New tracklets to add from the given frame boxes, each in
            ``filename:x1,y1,x2,y2`` format. A box is treated as a tracker
            output on that frame and added by MaskManager before propagating to
            the next frame. Given as one list,
            ``--add_at='["frame.jpg:10,20,110,220"]'``, or appended one at a
            time with ``--add_at+ frame.jpg:10,20,110,220``.
        remove_at: Tracklets to remove before propagating to the given frame,
            each in ``filename:tracklet_id`` format. Given as one list,
            ``--remove_at='["frame.jpg:3"]'``, or appended one at a time with
            ``--remove_at+ frame.jpg:3``.
        output_root: Directory holding one timestamped run directory per run.
        device: Device used by SAM and Cutie, for example ``cuda`` or ``cpu``.
        sam_model_type: SAM model type.
        cutie_model_type: Cutie model type.
        cutie_config_path: Directory holding the Cutie config; ``None`` uses the
            config shipped with the installed Cutie package.
        cutie_config_name: Name of the Cutie config to load.

    Returns:
        Exit code: ``0`` on success, ``1`` on validation error.
    """
    try:
        add_events = [parse_add_tracklet_event(event) for event in add_at or []]
        remove_events = [parse_remove_tracklet_event(event) for event in remove_at or []]

        frame_paths = list_selected_frame_paths(
            image_dir=image_dir,
            start_file=start_file,
            end_file=end_file,
        )
        if len(frame_paths) < 2:
            raise ValueError("At least two frames are required for MaskManager validation.")

        validate_lifecycle_events(
            frame_paths=frame_paths,
            add_events=add_events,
            remove_events=remove_events,
        )

        add_events_by_file = group_add_events_by_source_frame(frame_paths, add_events)
        remove_events_by_file = group_remove_events(remove_events)

        output_dir = timestamped_run_dir(output_root)

        resolved_device = validate_device(device, label="SAM/Cutie")
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    from trackers.core.masks.cutie import CutieMaskPropagator
    from trackers.core.masks.manager import MaskManager
    from trackers.core.masks.sam import SAMBoxMaskGenerator

    mask_manager = MaskManager(
        mask_generator=SAMBoxMaskGenerator(
            model_type=sam_model_type,
            device=resolved_device,
        ),
        mask_propagator=CutieMaskPropagator(
            model_type=cutie_model_type,
            config_path=cutie_config_path,
            config_name=cutie_config_name,
            device=resolved_device,
        ),
    )

    try:
        first_frame = load_rgb_image(frame_paths[0])

        active_tracklets: dict[int, TrackletSnapshot] = {
            index + 1: TrackletSnapshot(
                tracker_id=index + 1,
                xyxy=validate_and_clip_xyxy_box(
                    box=single_box,
                    image_shape=first_frame.shape[:2],
                ),
            )
            for index, single_box in enumerate(box)
        }
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    next_tracklet_id = len(active_tracklets) + 1

    print(f"Selected {len(frame_paths)} frames.")
    print(f"Saving outputs to {output_dir}")
    print("Initial tracklets:")
    for tracklet in active_tracklets.values():
        print(f"  {tracklet.tracker_id}: {tracklet.xyxy.tolist()}")

    # Previous-frame tracker state consumed by MaskManager.
    # These variables emulate the data flow between the tracker and MaskManager
    # in the original McByte pipeline.
    previous_frame: np.ndarray | None = None
    previous_tracklets: list[TrackletSnapshot] = []
    previous_new_tracklets: list[TrackletSnapshot] = []
    previous_removed_tracklet_ids: list[int] = []

    for frame_index, frame_path in enumerate(frame_paths):
        frame = load_rgb_image(frame_path)

        mask_output = mask_manager.get_updated_masks(
            frame=frame,
            previous_frame=previous_frame,
            previous_tracklets=previous_tracklets,
            new_tracklets=previous_new_tracklets,
            removed_tracklet_ids=previous_removed_tracklet_ids,
        )

        status_parts = []
        if mask_output is None:
            status_parts.append("no masks")
        else:
            status_parts.append(f"{0 if mask_output.masks is None else mask_output.masks.shape[0]} masks")

        if len(previous_new_tracklets) > 0:
            status_parts.append(f"added {[tracklet.tracker_id for tracklet in previous_new_tracklets]}")
        if len(previous_removed_tracklet_ids) > 0:
            status_parts.append(f"removed {previous_removed_tracklet_ids}")

        status = ", ".join(status_parts)
        print(f"Saved {frame_path.name}: {status}")

        new_tracklets_for_next_frame: list[TrackletSnapshot] = []
        removed_tracklet_ids_for_next_frame: list[int] = []

        frame_add_events = add_events_by_file.get(frame_path.name, [])
        for event in frame_add_events:
            tracklet_id = next_tracklet_id
            next_tracklet_id += 1

            tracklet = TrackletSnapshot(
                tracker_id=tracklet_id,
                xyxy=validate_and_clip_xyxy_box(
                    box=event.xyxy,
                    image_shape=frame.shape[:2],
                ),
            )
            active_tracklets[tracklet_id] = tracklet
            new_tracklets_for_next_frame.append(tracklet)

            print(f"Scheduled add from {frame_path.name}: tracklet {tracklet_id}, box {tracklet.xyxy.tolist()}")

        remove_tracklet_ids = remove_events_by_file.get(frame_path.name, [])
        for tracklet_id in remove_tracklet_ids:
            active_tracklets.pop(tracklet_id, None)
            removed_tracklet_ids_for_next_frame.append(tracklet_id)
            print(f"Scheduled removal from {frame_path.name}: tracklet {tracklet_id}")

        previous_frame = frame
        previous_tracklets = list(active_tracklets.values())
        previous_new_tracklets = new_tracklets_for_next_frame
        previous_removed_tracklet_ids = removed_tracklet_ids_for_next_frame

        tracklets_vis = list(active_tracklets.values()) if frame_index == 0 else previous_new_tracklets
        visualize_output(
            frame=frame,
            frame_name=frame_path.name,
            mask_output=mask_output,
            tracklets=tracklets_vis,
            output_path=output_dir / frame_path.name,
            status=status,
        )

    print(f"Done. Outputs saved to {output_dir.resolve()}")
    return 0


def _raise_for_mode_option_conflict(mode: MaskManagerMode, supplied: Mapping[str, Any]) -> None:
    """Reject options belonging to the mode that was not selected.

    Both modes are one command, so every option is registered on one parser and
    argparse cannot tell that ``--gt_file`` is meaningless under ``--mode
    manual``. This check restores that error, and runs before any model is
    loaded so a wrong invocation fails in milliseconds rather than after SAM and
    Cutie weights are on the device.

    Args:
        mode: The selected mode.
        supplied: Mapping of option name to the value the caller gave.

    Raises:
        ValueError: If an option belongs to the other mode, or a required option
            for the selected mode is missing.

    Examples:
        >>> _raise_for_mode_option_conflict("manual", {"gt_file": "gt.txt"})
        Traceback (most recent call last):
        ...
        ValueError: --gt_file is a --mode gt option and cannot be used with --mode manual.
    """
    foreign = _GT_ONLY_OPTIONS if mode == "manual" else _MANUAL_ONLY_OPTIONS
    other_mode = "gt" if mode == "manual" else "manual"
    for option in foreign:
        if supplied.get(option) is not None:
            raise ValueError(f"--{option} is a --mode {other_mode} option and cannot be used with --mode {mode}.")

    required = ("start_file", "end_file", "box") if mode == "manual" else ("gt_file", "start_frame", "end_frame")
    missing = [option for option in required if supplied.get(option) is None]
    if missing:
        joined = ", ".join(f"--{option}" for option in missing)
        raise ValueError(f"--mode {mode} requires {joined}.")


def mask_manager_command(
    image_dir: Path,
    mode: MaskManagerMode,
    start_file: str | None = None,
    end_file: str | None = None,
    box: list[tuple[float, float, float, float]] | None = None,
    add_at: list[str] | None = None,
    remove_at: list[str] | None = None,
    gt_file: Path | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    tracklet_id: list[PositiveInt | Literal["all"]] | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    device: str = "cuda",
    sam_model_type: str = "vit_b",
    cutie_model_type: str = "base-mega",
    cutie_config_path: Path | None = None,
    cutie_config_name: str = "eval_config",
    mask_creation_bbox_overlap_threshold: float = 0.6,
) -> int:
    """Run MaskManager over a frame range and save per-frame visualizations.

    Both modes follow the original McByte execution order. For frame ``t``:

        1. MaskManager produces masks for frame ``t`` using tracker outputs
           from frame ``t-1``.
        2. Tracklets for frame ``t`` are produced, from command-line boxes in
           ``manual`` mode or from the ground-truth file in ``gt`` mode.
        3. Newly created and removed tracklets become lifecycle events consumed
           by MaskManager on frame ``t+1``.

    Options belonging to the mode that was not selected are rejected rather than
    ignored, and the check runs before any model loads.

    Every option is also accepted with hyphens in place of underscores, so
    ``--start-file`` and ``--start_file`` name the same thing.

    Args:
        image_dir: Directory holding the frame images.
        mode: ``manual`` to drive MaskManager from command-line boxes and
            lifecycle events, ``gt`` to replay tracklets from a ground-truth
            file.
        start_file: ``manual`` mode. Filename of the first frame to process.
        end_file: ``manual`` mode. Filename of the last frame to process.
        box: ``manual`` mode. Initial tracklet boxes on the first selected
            frame, given as one list rather than a repeated option:
            ``--box='[[x1,y1,x2,y2]]'`` for one box,
            ``--box='[[10,20,110,220],[30,40,130,240]]'`` for two, and
            ``--box+='[[30,40,130,240]]'`` to append to boxes already given.
        add_at: ``manual`` mode. New tracklets to add from the given frame
            boxes, each in ``filename:x1,y1,x2,y2`` format. A box is treated as
            a tracker output on that frame and added by MaskManager before
            propagating to the next frame. Given as one list,
            ``--add_at='["frame.jpg:10,20,110,220"]'``, or appended one at a
            time with ``--add_at+ frame.jpg:10,20,110,220``.
        remove_at: ``manual`` mode. Tracklets to remove before propagating to
            the given frame, each in ``filename:tracklet_id`` format. Given as
            one list, ``--remove_at='["frame.jpg:3"]'``, or appended one at a
            time with ``--remove_at+ frame.jpg:3``.
        gt_file: ``gt`` mode. MOT-format ground-truth file to replay.
        start_frame: ``gt`` mode. First frame number to replay.
        end_frame: ``gt`` mode. Last frame number to replay.
        tracklet_id: ``gt`` mode. Tracklet IDs to replay, for example
            ``--tracklet_id='[3,7]'``. Omit the option, or pass
            ``--tracklet_id='[all]'``, to replay every tracklet.
        output_root: Directory holding one timestamped run directory per run.
        device: Device used by SAM and Cutie, for example ``cuda`` or ``cpu``.
        sam_model_type: SAM model type.
        cutie_model_type: Cutie model type.
        cutie_config_path: Directory holding the Cutie config; ``None`` uses the
            config shipped with the installed Cutie package.
        cutie_config_name: Name of the Cutie config to load.
        mask_creation_bbox_overlap_threshold: ``gt`` mode. Overlap threshold
            above which mask creation is delayed.

    Returns:
        Exit code: ``0`` on success, ``1`` on validation error.

    Examples:
        A foreign option is reported on stderr and exits non-zero, so only the
        return value shows up here.

        >>> mask_manager_command(Path("frames"), mode="manual", gt_file=Path("gt.txt"))
        1
    """
    supplied = {
        "start_file": start_file,
        "end_file": end_file,
        "box": box,
        "add_at": add_at,
        "remove_at": remove_at,
        "gt_file": gt_file,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "tracklet_id": tracklet_id,
    }
    try:
        _raise_for_mode_option_conflict(mode, supplied)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if mode == "gt":
        return run_gt_mode(
            image_dir=image_dir,
            gt_file=gt_file,  # type: ignore[arg-type]
            start_frame=start_frame,  # type: ignore[arg-type]
            end_frame=end_frame,  # type: ignore[arg-type]
            output_root=output_root,
            tracklet_id=tracklet_id,
            device=device,
            sam_model_type=sam_model_type,
            cutie_model_type=cutie_model_type,
            cutie_config_path=cutie_config_path,
            cutie_config_name=cutie_config_name,
            mask_creation_bbox_overlap_threshold=mask_creation_bbox_overlap_threshold,
        )

    return run_manual_mode(
        image_dir=image_dir,
        start_file=start_file,  # type: ignore[arg-type]
        end_file=end_file,  # type: ignore[arg-type]
        box=box,  # type: ignore[arg-type]
        output_root=output_root,
        add_at=add_at,
        remove_at=remove_at,
        device=device,
        sam_model_type=sam_model_type,
        cutie_model_type=cutie_model_type,
        cutie_config_path=cutie_config_path,
        cutie_config_name=cutie_config_name,
    )
