# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Compare locked-IoU and mask-conditioned McByte on one sequence.

The script supports two detection-file layouts:

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

import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

import cv2
import numpy as np
import supervision as sv
import torch
from jsonargparse import CLI, ArgumentParser

from trackers.cli.__main__ import _CLIParser, _normalise_option
from trackers.core.mcbyte.masks.base import MaskOutput
from trackers.core.mcbyte.tracker import McByteMaskConfig, McByteTracker
from trackers.utils.cmc import CMCMethod

DEFAULT_OUTPUT_DIR = Path("visual_tests/outputs/visualize_mcbyte_from_detections")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

RunMode = Literal["locked_iou", "mask_conditioned"]
RUN_MODES: tuple[RunMode, ...] = ("locked_iou", "mask_conditioned")

DetectionFileFormat = Literal["mot_tlwh", "xyxy"]


@dataclass(frozen=True)
class DetectionRecord:
    """One detection parsed from a MOT-style detection file."""

    xyxy: np.ndarray
    confidence: float


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
        device: Device used by SAM and Cutie in the mask-conditioned run.
        enable_isolated_matching: Allow mask evidence to rescue isolated
            positive-IoU pairs below the normal association threshold.
    """

    device: str = "cuda"
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
        (
            mask.device.startswith("cuda") and not torch.cuda.is_available(),
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Use --mask.device cpu or install CUDA-enabled PyTorch.",
        ),
    )

    for failed, message in checks:
        if failed:
            return message

    return None


def read_detection_file(
    det_file: Path,
    detection_format: DetectionFileFormat,
) -> dict[int, list[DetectionRecord]]:
    """Read detections and group them by frame number.

    Supported formats are:

    ``mot_tlwh``:
        ``frame,id,left,top,width,height,confidence,...``

        The input identity column is ignored. Bounding boxes are interpreted as
        top-left coordinates plus width and height.

    ``xyxy``:
        ``frame,x1,y1,x2,y2,confidence``

        Bounding boxes are interpreted directly as top-left and bottom-right
        coordinates.

    Detection confidence is preserved and used by McByte's high- and
    low-confidence association stages.
    """
    detections_by_frame: dict[int, list[DetectionRecord]] = defaultdict(list)

    minimum_column_count = 7 if detection_format == "mot_tlwh" else 6

    with det_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            values = [value.strip() for value in line.split(",")]

            if len(values) < minimum_column_count:
                raise ValueError(
                    f"Detection format {detection_format!r} requires at least "
                    f"{minimum_column_count} columns. Invalid line "
                    f"{line_number}: {line}"
                )

            try:
                frame_number = int(float(values[0]))

                if detection_format == "mot_tlwh":
                    left = float(values[2])
                    top = float(values[3])
                    width = float(values[4])
                    height = float(values[5])
                    confidence = float(values[6])

                    if width <= 0 or height <= 0:
                        continue

                    right = left + width
                    bottom = top + height

                else:
                    left = float(values[1])
                    top = float(values[2])
                    right = float(values[3])
                    bottom = float(values[4])
                    confidence = float(values[5])

                    if right <= left or bottom <= top:
                        continue

            except ValueError as exc:
                raise ValueError(
                    f"Could not parse detection line {line_number} using format {detection_format!r}: {line}"
                ) from exc

            if frame_number <= 0:
                raise ValueError(f"Invalid non-positive frame number on line {line_number}.")

            numeric_values = np.array(
                [left, top, right, bottom, confidence],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(numeric_values)):
                raise ValueError(f"Non-finite value on detection line {line_number}: {line}")

            xyxy = np.array(
                [left, top, right, bottom],
                dtype=np.float32,
            )

            detections_by_frame[frame_number].append(
                DetectionRecord(
                    xyxy=xyxy,
                    confidence=confidence,
                )
            )

    return dict(detections_by_frame)


def find_frame_path(
    image_dir: Path,
    frame_number: int,
) -> Path:
    """Find a frame using common MOT filename widths and image extensions."""
    filename_stems = (
        f"{frame_number:06d}",
        f"{frame_number:08d}",
    )

    for stem in filename_stems:
        for extension in IMAGE_EXTENSIONS:
            frame_path = image_dir / f"{stem}{extension}"
            if frame_path.is_file():
                return frame_path

    attempted_names = [f"{stem}{extension}" for stem in filename_stems for extension in IMAGE_EXTENSIONS]
    raise FileNotFoundError(f"Could not find frame {frame_number} in {image_dir}. Tried: {attempted_names}")


def load_rgb_frame(frame_path: Path) -> np.ndarray:
    """Load one image and convert it from OpenCV BGR to RGB."""
    frame_bgr = cv2.imread(str(frame_path))
    if frame_bgr is None:
        raise RuntimeError(f"cv2.imread failed for frame: {frame_path}")

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def build_detections(
    records: list[DetectionRecord],
) -> sv.Detections:
    """Convert parsed detection records into ``sv.Detections``."""
    if not records:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.stack([record.xyxy for record in records]).astype(np.float32),
        confidence=np.array(
            [record.confidence for record in records],
            dtype=np.float32,
        ),
    )


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
    """Recreate the output directory for one comparison mode.

    Any existing directory for the selected mode is removed before new outputs are
    written.
    """
    run_dir = output_root / mode_name
    if run_dir.exists():
        shutil.rmtree(run_dir)

    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.txt"
    return frames_dir, results_path


def color_from_id(tracker_id: int) -> tuple[int, int, int]:
    """Return a deterministic RGB color for a tracker ID."""
    if tracker_id < 0:
        return 180, 180, 180

    rng = np.random.default_rng(tracker_id)
    color = rng.integers(40, 256, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def get_mask_tracklet_ids_in_order(
    tracklet_mask_dict: dict[int, int],
) -> list[int]:
    """Return tracklet IDs ordered according to ``MaskOutput.masks``."""
    return [
        tracklet_id
        for tracklet_id, _ in sorted(
            tracklet_mask_dict.items(),
            key=lambda item: item[1],
        )
    ]


def overlay_masks(
    frame: np.ndarray,
    mask_output: MaskOutput | None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay propagated masks using deterministic tracklet colors."""
    if mask_output is None or mask_output.masks is None:
        return frame.copy()

    tracklet_ids = get_mask_tracklet_ids_in_order(mask_output.tracklet_mask_dict)

    if len(tracklet_ids) != mask_output.masks.shape[0]:
        raise ValueError(
            "The mask count does not match tracklet_mask_dict. "
            f"Got {mask_output.masks.shape[0]} masks and "
            f"{len(tracklet_ids)} mapped tracklet IDs."
        )

    output = frame.copy()

    for mask, tracker_id in zip(mask_output.masks, tracklet_ids):
        mask_bool = mask.astype(bool, copy=False)
        if not np.any(mask_bool):
            continue

        color = np.array(color_from_id(tracker_id), dtype=np.uint8)
        colored_mask = np.zeros_like(output)
        colored_mask[mask_bool] = color

        output = np.where(
            mask_bool[..., None],
            (alpha * colored_mask + (1.0 - alpha) * output).astype(np.uint8),
            output,
        )

    return output


def draw_tracking_boxes(
    frame: np.ndarray,
    tracked_detections: sv.Detections,
) -> np.ndarray:
    """Draw tracker boxes and stable IDs on an RGB frame."""
    output = frame.copy()

    tracker_ids = tracked_detections.tracker_id
    if tracker_ids is None:
        tracker_ids = np.full(len(tracked_detections), -1, dtype=int)

    for xyxy, tracker_id_value in zip(
        tracked_detections.xyxy,
        tracker_ids,
    ):
        tracker_id = int(tracker_id_value)
        x1, y1, x2, y2 = np.rint(xyxy).astype(int)

        color = color_from_id(tracker_id)
        label = str(tracker_id) if tracker_id >= 0 else "unmatched"

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color=color,
            thickness=2,
        )
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def draw_text_panel(
    frame: np.ndarray,
    lines: list[str],
) -> np.ndarray:
    """Draw readable status text in the top-left corner."""
    output = frame.copy()
    x = 20
    initial_y = 30
    line_height = 24

    for line_index, line in enumerate(lines):
        y = initial_y + line_index * line_height

        cv2.putText(
            output,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return output


def get_pending_tracklet_ids(
    tracker: McByteTracker,
) -> set[int]:
    """Return pending mask IDs for visual debugging.

    This visual test intentionally inspects MaskManager's internal lifecycle state.
    """
    if tracker.mask_manager is None:
        return set()

    return set(tracker.mask_manager._pending_tracklet_ids)


def get_masked_tracklet_ids(
    mask_output: MaskOutput | None,
) -> set[int]:
    """Return tracklet IDs represented in the current mask output."""
    if mask_output is None or mask_output.masks is None:
        return set()

    return set(mask_output.tracklet_mask_dict)


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

    if use_masks:
        visual = overlay_masks(
            frame=visual,
            mask_output=tracker._last_mask_output,
        )

    visual = draw_tracking_boxes(
        frame=visual,
        tracked_detections=tracked_detections,
    )

    tracker_ids = tracked_detections.tracker_id
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

    return draw_text_panel(
        frame=visual,
        lines=status_lines,
    )


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


def save_rgb_frame(
    frame: np.ndarray,
    output_path: Path,
) -> None:
    """Save an RGB frame through OpenCV."""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if not cv2.imwrite(str(output_path), frame_bgr):
        raise RuntimeError(f"Could not save visualization: {output_path}")


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

            save_rgb_frame(
                frame=visual,
                output_path=frames_dir / f"{frame_number:06d}.jpg",
            )

            if processed_index == 1 or processed_index == total_frames or processed_index % 25 == 0:
                print(f"[{mode_name}] frame {frame_number} ({processed_index}/{total_frames})")

    tracker.reset()

    print(f"[{mode_name}] Frames: {frames_dir}")
    print(f"[{mode_name}] Results: {results_path}")


def compare_mcbyte_command(
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
        output_dir: Root directory for both comparison runs.

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

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Image directory: {sequence.image_dir}")
    print(f"Detection file: {sequence.det_file}")
    print(f"Detection format: {sequence.det_format}")
    print(f"Frame range: {sequence.start_frame} to {sequence.end_frame} (inclusive)")
    print(f"Output root: {output_dir}")

    for mode_name in modes:
        run_mode(
            mode_name=mode_name,
            detections_by_frame=detections_by_frame,
            image_dir=sequence.image_dir,
            start_frame=sequence.start_frame,
            end_frame=sequence.end_frame,
            output_root=output_dir,
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
        help="Root directory for both comparison runs.",
    )
    added_args.extend(["modes", "output_dir"])
    return added_args


class _ComparisonParser(_CLIParser):
    """Expose the option dataclasses while keeping the shared boolean syntax."""

    def add_function_arguments(self, function, *args, **kwargs):  # type: ignore[override]
        if function is compare_mcbyte_command:
            return _add_compare_arguments(self)
        return super().add_function_arguments(function, *args, **kwargs)


def main() -> int:
    """Parse the command line and run the requested McByte comparisons."""
    args = [_normalise_option(arg) for arg in sys.argv[1:]]
    rc = CLI(
        compare_mcbyte_command,
        args=args,
        as_positional=False,
        prog="python visual_tests/visualize_mcbyte_from_detections.py",
        description="Compare locked-IoU and mask-conditioned McByte on one sequence.",
        parser_class=_ComparisonParser,
    )
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
