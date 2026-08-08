# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Compare locked-IoU and mask-conditioned McByte on one sequence.

A supported inspection command, in beta while the ``inspect`` group settles.

Two detection-file layouts are supported:

``mot_tlwh``:
    ``frame,id,left,top,width,height,confidence,...``

``xyxy``:
    ``frame,x1,y1,x2,y2,confidence``

For ``mot_tlwh``, the input identity column is ignored because tracker
identities are produced by McByte. Both formats are converted internally to
``xyxy`` bounding boxes.

The script can be used with datasets such as MOT17, DanceTrack, SportsMOT,
and SoccerNet-tracking, provided their frame filenames follow one of the
supported naming conventions.

The script runs the same detections through two McByte configurations:

1. ``locked_iou``:
   McByte clear-match locking and reduced assignment, without MaskManager.
2. ``mask_conditioned``:
   Full McByte using SAM mask initialization, Cutie propagation, clear-match
   locking, and mask-conditioned association.

Both runs save:

- per-frame visualizations;
- MOTChallenge-style tracking results.

The full mask-conditioned run additionally overlays the propagated masks and
displays mask lifecycle information.

Related options are grouped, so each one is spelled with a dotted prefix, for
example ``--sequence.image_dir`` and ``--mask.device``. Both separators are
accepted in the option name: ``--sequence.image-dir`` and
``--sequence.image_dir`` select the same option.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import supervision as sv
from jsonargparse import ArgumentParser

from trackers.cli._annotate import annotate_masks, annotate_tracklet_boxes, draw_status_lines
from trackers.cli._detections import (
    DetectionRecord,
    build_detections,
    find_frame_path,
    load_rgb_frame,
    read_detection_file,
)
from trackers.cli._parser import register_argument_adder
from trackers.cli.inspect._common import (
    INSPECT_OUTPUT_ROOT,
    get_mask_tracklet_ids_in_order,
    get_masked_tracklet_ids,
    require_torch,
    save_rgb_image,
    timestamped_run_dir,
)
from trackers.core.mcbyte.tracker import McByteMaskConfig, McByteTracker
from trackers.utils.cmc import CMCMethod
from trackers.utils.device import _validate_device

DEFAULT_OUTPUT_DIR = INSPECT_OUTPUT_ROOT / "mcbyte"

RunMode = Literal["locked_iou", "mask_conditioned"]
RUN_MODES: tuple[RunMode, ...] = ("locked_iou", "mask_conditioned")

DetectionFileFormat = Literal["mot_tlwh", "xyxy"]


@dataclass
class SequenceOptions:
    """Frames, detections, and the frame range to process.

    Attributes:
        image_dir: Directory containing sequence frames.
        det_file: Path to the detection file.
        start_frame: First frame number to process, inclusive.
        end_frame: Last frame number to process, inclusive.
        det_format: Detection-file column format. ``mot_tlwh`` expects
            ``frame,id,left,top,width,height,confidence,...``; ``xyxy`` expects
            ``frame,x1,y1,x2,y2,confidence``.
        frame_rate: Sequence frame rate used to scale the lost-track buffer.
    """

    image_dir: Path
    det_file: Path
    start_frame: int
    end_frame: int
    det_format: DetectionFileFormat = "mot_tlwh"
    frame_rate: float = 30.0


@dataclass
class CMCOptions:
    """Camera-motion compensation settings shared by both runs.

    Attributes:
        enable: Enable camera motion compensation in both runs.
        method: Camera-motion compensation method.
        downscale: Image downscale factor used by CMC.
    """

    enable: bool = False
    method: CMCMethod = "sparseOptFlow"
    downscale: int = 6


@dataclass
class MaskOptions:
    """Settings that only the mask-conditioned run reads.

    Attributes:
        device: Device used by SAM and Cutie in the mask-conditioned run. The
            default ``auto`` resolves to CUDA when available, otherwise CPU.
        enable_isolated_matching: Allow mask evidence to rescue isolated
            positive-IoU pairs below the normal association threshold.
    """

    device: str = "auto"
    enable_isolated_matching: bool = False


def validate_options(
    sequence: SequenceOptions,
    cmc: CMCOptions,
    mask: MaskOptions,
) -> str | None:
    """Validate paths, frame range, device, and numeric options.

    The checks are held in one table and reported in order, so the caller can
    print the first problem and exit instead of raising at the user.

    Args:
        sequence: Frames, detections, and the frame range to process.
        cmc: Camera-motion compensation settings.
        mask: Settings that only the mask-conditioned run reads.

    Returns:
        Message describing the first failed check, or ``None`` when every
        option is usable.
    """
    checks: tuple[tuple[bool, str], ...] = (
        (not sequence.image_dir.is_dir(), f"Image directory does not exist: {sequence.image_dir}"),
        (not sequence.det_file.is_file(), f"Detection file does not exist: {sequence.det_file}"),
        (sequence.start_frame <= 0, "sequence.start_frame must be positive."),
        (
            sequence.end_frame < sequence.start_frame,
            "sequence.end_frame must be greater than or equal to sequence.start_frame.",
        ),
        (sequence.frame_rate <= 0, "sequence.frame_rate must be positive."),
        (cmc.downscale <= 0, "cmc.downscale must be positive."),
    )

    for failed, message in checks:
        if failed:
            return message

    # Kept out of the table above, which evaluates every condition eagerly:
    # the shared device check raises rather than returning a flag. require_torch
    # runs first so a missing mask extra still reports the pip hint.
    require_torch()
    try:
        _validate_device(mask.device, label="SAM/Cutie")
    except RuntimeError as error:
        return str(error)

    return None


def create_tracker(
    *,
    use_masks: bool,
    frame_rate: float,
    device: str,
    enable_cmc: bool,
    cmc_method: CMCMethod,
    cmc_downscale: int,
    enable_isolated_mask_matching: bool,
) -> McByteTracker:
    """Create mask-free locked-IoU or full mask-conditioned McByte."""
    if use_masks:
        return McByteTracker(
            frame_rate=frame_rate,
            enable_cmc=enable_cmc,
            cmc_method=cmc_method,
            cmc_downscale=cmc_downscale,
            enable_mask_manager=True,
            mask_config=McByteMaskConfig(
                device=device,
            ),
            enable_isolated_mask_matching=(enable_isolated_mask_matching),
        )

    return McByteTracker(
        frame_rate=frame_rate,
        enable_cmc=enable_cmc,
        cmc_method=cmc_method,
        cmc_downscale=cmc_downscale,
        enable_mask_manager=False,
        enable_isolated_mask_matching=False,
    )


def prepare_run_directory(
    output_root: Path,
    mode_name: str,
) -> tuple[Path, Path]:
    """Create the output directory for one comparison mode.

    ``output_root`` is a fresh run directory, so the mode directory starts empty
    without deleting anything. Frames are named after the frame number, and a
    shorter re-run into a populated directory would leave stale frames from a
    longer previous run interleaved with the new ones.
    """
    run_dir = output_root / mode_name
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.txt"
    return frames_dir, results_path


def get_pending_tracklet_ids(
    tracker: McByteTracker,
) -> set[int]:
    """Return pending mask IDs for visual debugging.

    This visual test intentionally inspects MaskManager's internal lifecycle state.
    """
    if tracker.mask_manager is None:
        return set()

    return set(tracker.mask_manager._pending_tracklet_ids)


def visualize_frame(
    *,
    frame: np.ndarray,
    frame_number: int,
    mode_name: str,
    input_detection_count: int,
    tracked_detections: sv.Detections,
    tracker: McByteTracker,
    use_masks: bool,
) -> np.ndarray:
    """Create one complete comparison visualization frame."""
    visual = frame.copy()

    mask_output = tracker._last_mask_output
    if use_masks and mask_output is not None and mask_output.masks is not None:
        visual = annotate_masks(
            visual,
            mask_output.masks,
            get_mask_tracklet_ids_in_order(mask_output.tracklet_mask_dict),
        )

    tracker_ids = tracked_detections.tracker_id
    if tracker_ids is not None and len(tracked_detections) > 0:
        visual = annotate_tracklet_boxes(
            visual,
            tracked_detections.xyxy,
            [int(tracker_id) for tracker_id in tracker_ids],
        )

    valid_tracker_ids = (
        [] if tracker_ids is None else sorted(int(tracker_id) for tracker_id in tracker_ids if tracker_id >= 0)
    )

    status_lines = [
        f"Mode: {mode_name}",
        f"Frame: {frame_number}",
        f"Input detections: {input_detection_count}",
        f"Output IDs: {valid_tracker_ids}",
    ]

    if use_masks:
        status_lines.extend(
            [
                (f"Masks: {sorted(get_masked_tracklet_ids(tracker._last_mask_output))}"),
                (f"Pending: {sorted(get_pending_tracklet_ids(tracker))}"),
            ]
        )

    return draw_status_lines(visual, status_lines)


def append_mot_results(
    *,
    results_file: TextIO,
    frame_number: int,
    tracked_detections: sv.Detections,
) -> None:
    """Append confirmed tracker outputs in MOTChallenge result format.

    Detections with negative tracker IDs are omitted.
    """
    tracker_ids = tracked_detections.tracker_id
    if tracker_ids is None:
        return

    confidences = tracked_detections.confidence
    if confidences is None:
        confidences = np.ones(len(tracked_detections), dtype=np.float32)

    for xyxy, tracker_id_value, confidence_value in zip(
        tracked_detections.xyxy,
        tracker_ids,
        confidences,
    ):
        tracker_id = int(tracker_id_value)
        if tracker_id < 0:
            continue

        left, top, right, bottom = map(float, xyxy)
        width = right - left
        height = bottom - top
        confidence = float(confidence_value)

        results_file.write(
            f"{frame_number},{tracker_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},{confidence:.6f},-1,-1,-1\n"
        )


def run_mode(
    *,
    mode_name: str,
    detections_by_frame: dict[int, list[DetectionRecord]],
    image_dir: Path,
    start_frame: int,
    end_frame: int,
    output_root: Path,
    frame_rate: float,
    device: str,
    enable_cmc: bool,
    cmc_method: CMCMethod,
    cmc_downscale: int,
    enable_isolated_mask_matching: bool,
) -> None:
    """Run one McByte configuration over the requested inclusive frame range.

    Tracking results are written in MOTChallenge format and one annotated image is
    saved for every processed frame.
    """
    use_masks = mode_name == "mask_conditioned"

    print(f"\nRunning mode: {mode_name}")

    frames_dir, results_path = prepare_run_directory(
        output_root=output_root,
        mode_name=mode_name,
    )

    tracker = create_tracker(
        use_masks=use_masks,
        frame_rate=frame_rate,
        device=device,
        enable_cmc=enable_cmc,
        cmc_method=cmc_method,
        cmc_downscale=cmc_downscale,
        enable_isolated_mask_matching=enable_isolated_mask_matching,
    )

    total_frames = end_frame - start_frame + 1

    with results_path.open("w", encoding="utf-8") as results_file:
        for processed_index, frame_number in enumerate(
            range(start_frame, end_frame + 1),
            start=1,
        ):
            frame_path = find_frame_path(
                image_dir=image_dir,
                frame_number=frame_number,
            )
            frame = load_rgb_frame(frame_path)

            input_detections = build_detections(detections_by_frame.get(frame_number, []))

            tracked_detections = tracker.update(
                detections=input_detections,
                frame=frame,
            )

            append_mot_results(
                results_file=results_file,
                frame_number=frame_number,
                tracked_detections=tracked_detections,
            )

            visual = visualize_frame(
                frame=frame,
                frame_number=frame_number,
                mode_name=mode_name,
                input_detection_count=len(input_detections),
                tracked_detections=tracked_detections,
                tracker=tracker,
                use_masks=use_masks,
            )

            save_rgb_image(
                image_rgb=visual,
                output_path=frames_dir / f"{frame_number:06d}.jpg",
            )

            if processed_index == 1 or processed_index == total_frames or processed_index % 25 == 0:
                print(f"[{mode_name}] frame {frame_number} ({processed_index}/{total_frames})")

    tracker.reset()

    print(f"[{mode_name}] Frames: {frames_dir}")
    print(f"[{mode_name}] Results: {results_path}")


def mcbyte_inspection(
    sequence: SequenceOptions,
    cmc: CMCOptions | None = None,
    mask: MaskOptions | None = None,
    modes: list[RunMode] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> int:
    """Run locked-IoU and full mask-conditioned McByte sequentially.

    Args:
        sequence: Frames, detections, and the frame range to process.
        cmc: Camera-motion compensation settings shared by both runs.
        mask: Settings that only the mask-conditioned run reads.
        modes: Tracker configurations to run, given in bracket syntax such as
            ``--modes=[locked_iou]``. By default, both the mask-free locked-IoU
            baseline and full mask-conditioned McByte are run.
        output_dir: Directory holding one timestamped run directory per
            invocation. Both comparison runs write inside it.

    Returns:
        Exit code: ``0`` on success, ``1`` on a validation error.
    """
    if cmc is None:
        cmc = CMCOptions()
    if mask is None:
        mask = MaskOptions()
    modes = modes or list(RUN_MODES)

    error = validate_options(sequence, cmc, mask)
    if error is not None:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        detections_by_frame = read_detection_file(
            det_file=sequence.det_file,
            detection_format=sequence.det_format,
        )
    except ValueError as parse_error:
        print(f"Error: {parse_error}", file=sys.stderr)
        return 1

    # One run directory per invocation, with a subdirectory per mode inside it,
    # so both modes of a single comparison stay together while a re-run never
    # writes into the outputs of an earlier one.
    run_root = timestamped_run_dir(output_dir)

    print(f"Image directory: {sequence.image_dir}")
    print(f"Detection file: {sequence.det_file}")
    print(f"Detection format: {sequence.det_format}")
    print(f"Frame range: {sequence.start_frame} to {sequence.end_frame} (inclusive)")
    print(f"Output root: {run_root}")

    for mode_name in modes:
        run_mode(
            mode_name=mode_name,
            detections_by_frame=detections_by_frame,
            image_dir=sequence.image_dir,
            start_frame=sequence.start_frame,
            end_frame=sequence.end_frame,
            output_root=run_root,
            frame_rate=sequence.frame_rate,
            device=mask.device,
            enable_cmc=cmc.enable,
            cmc_method=cmc.method,
            cmc_downscale=cmc.downscale,
            enable_isolated_mask_matching=mask.enable_isolated_matching,
        )

    print("\nFinished selected McByte comparison run(s).")
    return 0


# Option dataclasses paired with the nested CLI key each is registered under, so
# a dotted option path cannot drift from the dataclass that defines it.
_OPTION_GROUPS: tuple[tuple[type, str], ...] = (
    (SequenceOptions, "sequence"),
    (CMCOptions, "cmc"),
    (MaskOptions, "mask"),
)


def _add_compare_arguments(parser: ArgumentParser) -> list[str]:
    """Register the comparison arguments under their nested dataclass paths."""
    added_args: list[str] = []
    for option_class, nested_key in _OPTION_GROUPS:
        added_args.extend(parser.add_class_arguments(option_class, nested_key))
    parser.add_argument(
        "--modes",
        type=list[RunMode] | None,
        default=None,
        help=(
            "Tracker configurations to run, in bracket syntax, for example "
            "--modes=[locked_iou]. By default, both the mask-free locked-IoU "
            "baseline and full mask-conditioned McByte are run."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=("Directory holding one timestamped run directory per invocation. Both comparison runs write inside it."),
    )
    added_args.extend(["modes", "output_dir"])
    return added_args


# ``_add_compare_arguments`` is dispatched from ``trackers.cli._parser._CLIParser``
# rather than from a parser subclass of its own. Every subcommand shares one
# ``parser_class`` once the commands are nested under ``trackers inspect``, so a
# local subclass would never be constructed. Registering from here rather than
# being named in ``_parser`` is what keeps this module, and the cv2 and
# supervision imports above it, out of the parser build for every other command.
register_argument_adder(mcbyte_inspection, _add_compare_arguments)
