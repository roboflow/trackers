# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Replay detections through McByte++ with annotated, reproducible outputs."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
from jsonargparse import ArgumentParser
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from trackers.cli._detections import build_detections, find_frame_path, load_rgb_frame, read_detection_file
from trackers.cli._mcbyteplusplus import (
    CMCOptions,
    TrackerOptions,
    _add_reid_arguments,
    _HubAuthNoticeFilter,
    _load_mcbyteplusplus_reid,
    _reid_provenance,
    _tracker_kwargs,
    _validate_reid_options,
    visualize_mcbyteplusplus,
)
from trackers.cli._parser import register_argument_adder
from trackers.cli._reid_output import _ReIDResultWriter, _sort_mot_results
from trackers.cli._tracking import SequenceOptions as TrackingSequenceOptions
from trackers.cli._tracking import append_mot_results
from trackers.cli.inspect._common import (
    INSPECT_OUTPUT_ROOT,
    save_rgb_image,
    timestamped_run_dir,
)
from trackers.cli.inspect.edgetam import _versions
from trackers.cli.track import ReIDModelConfig
from trackers.core.masks.edgetam import EdgeTAMMaskBackend
from trackers.core.mcbyteplusplus.tracker import McBytePlusPlusMaskConfig, McBytePlusPlusTracker


@dataclass
class SequenceOptions(TrackingSequenceOptions):
    """Sequence selection with an optional detection-confidence override.

    Attributes:
        confidence_override: Replace file confidences, for example 1.0 for
            SoccerNet detections whose confidence column is a placeholder.
    """

    confidence_override: float | None = None


def mcbyteplusplus_inspection(
    sequence: SequenceOptions,
    cmc: CMCOptions | None = None,
    edgetam: McBytePlusPlusMaskConfig | None = None,
    tracker: TrackerOptions | None = None,
    output_dir: Path = INSPECT_OUTPUT_ROOT / "mcbyteplusplus",
    reid: ReIDModelConfig | None = None,
) -> int:
    """Inspect McByte++ with EdgeTAM masks on selected detection frames.

    Always enables the EdgeTAM mask pipeline. Camera motion compensation is
    enabled by default with ORB at downscale four; use cmc.enable to disable it.
    Saves RGB annotated frames, MOT results, a reusable config and model,
    dependency and frame provenance. Input identity columns are ignored.
    Frames enter the tracker as BGR, which converts them for its RGB mask
    backend. Only this inspector enumerates selected paths in advance. A progress
    bar counts fully saved frames. Runtime warnings and the Hub authentication
    notice are retained in run.json instead of printed to the terminal; errors
    and other logged diagnostics remain visible. A terminal message points to
    the saved warnings when any were captured.

    With re-ID enabled, collect current finalized identities and previously
    pending observations automatically for MOT output, including the final
    records returned by ``finish()``. Live frames show pending identities as
    ``-1``; historical identity resolution does not redraw earlier frames.

    Args:
        sequence: Input frames, detection file and inclusive frame range.
        cmc: Camera-motion compensation; defaults to ORB at downscale four.
        edgetam: EdgeTAM model, device and bounded-session settings.
        tracker: Association, confirmation, mask lifecycle and re-ID joining settings.
        output_dir: Root for timestamped output directories.
        reid: Optional encoder source and device. McByte++ requires an explicit
            model; use tracker.reid_joining_config to customize joining.

    Returns:
        Zero on success, one on invalid input or inference failure, or 130 on
        keyboard interruption. Unfinished runs retain a manifest and completed outputs.
    """
    cmc = cmc or CMCOptions()
    edgetam = edgetam or McBytePlusPlusMaskConfig()
    tracker_options = tracker or TrackerOptions()
    reid = reid or ReIDModelConfig()
    run_dir = None
    tracker_obj = None
    manifest: dict[str, Any] = {"status": "failed", "tracker": "mcbyteplusplus", "frames": []}
    exit_code = 0
    hub_logger = logging.getLogger("huggingface_hub.utils._http")
    hub_notice_filter = _HubAuthNoticeFilter()
    hub_logger.addFilter(hub_notice_filter)
    with warnings.catch_warnings(record=True) as caught_warnings:
        try:
            _validate_reid_options(reid, tracker_options.reid_joining_config)
            if sequence.start_frame <= 0 or sequence.end_frame < sequence.start_frame:
                raise ValueError("sequence frame range must be positive and inclusive.")
            if not math.isfinite(sequence.frame_rate) or sequence.frame_rate <= 0 or cmc.downscale <= 0:
                raise ValueError("sequence.frame_rate and cmc.downscale must be positive.")
            if sequence.confidence_override is not None and (
                not math.isfinite(sequence.confidence_override) or not 0 <= sequence.confidence_override <= 1
            ):
                raise ValueError("sequence.confidence_override must be between zero and one.")
            frame_paths = [
                find_frame_path(sequence.image_dir, number)
                for number in range(sequence.start_frame, sequence.end_frame + 1)
            ]
            detections = read_detection_file(sequence.det_file, sequence.det_format, sequence.confidence_override)
            run_dir = timestamped_run_dir(output_dir)
            manifest.update(requested_revision=edgetam.revision, versions=_versions())
            config: dict[str, Any] = {
                "sequence": {
                    **asdict(sequence),
                    "image_dir": str(sequence.image_dir.resolve()),
                    "det_file": str(sequence.det_file.resolve()),
                },
                "tracker": asdict(tracker_options),
                "cmc": asdict(cmc),
                "edgetam": asdict(edgetam),
                "reid": asdict(reid),
                "output_dir": str(output_dir.resolve()),
            }
            (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            manifest["config"] = config
            manifest["detection_sha256"] = hashlib.sha256(sequence.det_file.read_bytes()).hexdigest()
            reid_model = _load_mcbyteplusplus_reid(reid)
            manifest["reid"] = _reid_provenance(reid, reid_model)
            manifest["versions"]["reid"] = manifest["reid"]["reid_version"]
            if reid_model is not None:
                config["reid"]["device"] = manifest["reid"]["device"]
            tracker_obj = McBytePlusPlusTracker(
                frame_rate=sequence.frame_rate,
                enable_cmc=cmc.enable,
                cmc_method=cmc.method,
                cmc_downscale=cmc.downscale,
                enable_conditional_cmc=cmc.conditional,
                cmc_minimum_matches=cmc.minimum_matches,
                cmc_minimum_scale=cmc.minimum_scale,
                cmc_maximum_scale=cmc.maximum_scale,
                enable_mask_manager=True,
                edgetam_config=edgetam,
                reid_model=reid_model,
                **_tracker_kwargs(asdict(tracker_options)),
            )
            if tracker_obj.reid_joiner is not None:
                config["tracker"]["reid_joining_config"] = asdict(tracker_obj.reid_joiner.config)
            if tracker_obj.mask_manager is None:
                raise RuntimeError("McByte++ inspection requires its mask manager.")
            backend = cast(EdgeTAMMaskBackend, tracker_obj.mask_manager.mask_propagator)
            manifest.update(resolved_revision=backend.model_revision, device=str(backend.device))
            config["edgetam"].update(revision=backend.model_revision or edgetam.revision, device=str(backend.device))
            (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            processing_error: Exception | None = None
            try:
                with (
                    (run_dir / "results.txt").open("w", encoding="utf-8") as results_file,
                    Progress(
                        TextColumn("{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        MofNCompleteColumn(),
                        TimeElapsedColumn(),
                        TimeRemainingColumn(),
                        console=Console(stderr=True),
                    ) as progress,
                ):
                    result_writer = _ReIDResultWriter(
                        tracker_obj,
                        lambda number, values: append_mot_results(
                            results_file=results_file, frame_number=number, tracked_detections=values
                        ),
                    )
                    task = progress.add_task("McByte++", total=len(frame_paths))
                    interrupted = False
                    try:
                        for number, frame_path in zip(range(sequence.start_frame, sequence.end_frame + 1), frame_paths):
                            rgb = load_rgb_frame(frame_path)
                            tracked = tracker_obj.update(
                                build_detections(detections.get(number, [])), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            )
                            result_writer.write_current(number, tracked)
                            save_rgb_image(
                                visualize_mcbyteplusplus(rgb, number, tracked, tracker_obj),
                                run_dir / "frames" / frame_path.name,
                            )
                            manifest["frames"].append(
                                {
                                    "frame": number,
                                    "file": frame_path.name,
                                    "sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                                    "session_index": backend.session_index,
                                    "session_frame_count": backend.session_frame_count,
                                    "output_ids": [] if tracked.tracker_id is None else tracked.tracker_id.tolist(),
                                    "excluded_mask_ids": sorted(tracker_obj.untrusted_mask_ids),
                                }
                            )
                            progress.advance(task)
                    except KeyboardInterrupt:
                        interrupted = True
                    except Exception as error:
                        processing_error = error
                    result_writer.finish(
                        completed_only=interrupted or processing_error is not None, processing_error=processing_error
                    )
                    manifest["reid_finalized_rows"] = result_writer.historical_rows
                    if processing_error is not None:
                        raise processing_error
            except Exception as cleanup_error:
                # Close files before sorting recovered rows, retaining the original
                # processing failure even if output cleanup raises another one.
                if processing_error is None:
                    raise
                if cleanup_error is not processing_error:
                    print(f"Warning: Output cleanup also failed: {cleanup_error}", file=sys.stderr)
            if result_writer.historical_rows:
                _sort_mot_results(run_dir / "results.txt", processing_error=processing_error)
            if processing_error is not None:
                raise processing_error
            if interrupted:
                raise KeyboardInterrupt
            manifest["status"] = "completed"
        except KeyboardInterrupt:
            manifest.update(status="interrupted", error="Keyboard interruption")
            exit_code = 130
        except (OSError, ImportError, ValueError, RuntimeError) as error:
            manifest["error"] = str(error)
            print(f"Error: {error}", file=sys.stderr)
            exit_code = 1
        finally:
            hub_logger.removeFilter(hub_notice_filter)
            if tracker_obj is not None:
                try:
                    tracker_obj.reset()
                except Exception as error:
                    if manifest["status"] == "completed":
                        raise
                    print(f"Warning: Tracker cleanup also failed: {error}", file=sys.stderr)
            manifest["warnings"] = [
                {
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                    "file": warning.filename,
                    "line": warning.lineno,
                }
                for warning in caught_warnings
            ] + [{"category": "HuggingFaceHubWarning", "message": message} for message in hub_notice_filter.messages]
            if run_dir is not None:
                manifest_path = run_dir / "run.json"
                try:
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                except OSError as error:
                    print(f"Error: Could not write run manifest: {error}", file=sys.stderr)
                    exit_code = 1
                else:
                    if manifest["warnings"]:
                        print(f"Warnings were captured. See {manifest_path} (warnings field).", file=sys.stderr)
    if not exit_code:
        print(f"Saved McByte++ inspection to {run_dir}")
    return exit_code


def _add_inspect_arguments(parser: ArgumentParser) -> list[str]:
    """Register the nested config groups used by this command."""
    added: list[str] = []
    groups = (
        (SequenceOptions, "sequence"),
        (CMCOptions, "cmc"),
        (McBytePlusPlusMaskConfig, "edgetam"),
        (TrackerOptions, "tracker"),
    )
    for option_class, key in groups:
        added.extend(parser.add_class_arguments(option_class, key))
    added.extend(_add_reid_arguments(parser))
    parser.add_argument("--output_dir", type=Path, default=INSPECT_OUTPUT_ROOT / "mcbyteplusplus")
    return [*added, "output_dir"]


register_argument_adder(mcbyteplusplus_inspection, _add_inspect_arguments)
