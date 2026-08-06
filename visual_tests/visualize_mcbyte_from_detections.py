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
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO, cast

import cv2
import numpy as np
import supervision as sv
import torch

from trackers.core.mcbyte.masks.base import MaskOutput
from trackers.core.mcbyte.tracker import McByteMaskConfig, McByteTracker
from trackers.utils.cmc import CMCMethod

DEFAULT_OUTPUT_DIR = Path("visual_tests/outputs/visualize_mcbyte_from_detections")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
RUN_MODES = ("locked_iou", "mask_conditioned")
SUPPORTED_CMC_METHODS = ("orb", "sift", "sparseOptFlow", "ecc")

DetectionFileFormat = Literal["mot_tlwh", "xyxy"]
SUPPORTED_DETECTION_FORMATS = ("mot_tlwh", "xyxy")


@dataclass(frozen=True)
class DetectionRecord:
    """One detection parsed from a MOT-style detection file."""

    xyxy: np.ndarray
    confidence: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=("Compare locked-IoU and mask-conditioned McByte on one sequence."))
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing sequence frames.",
    )
    parser.add_argument(
        "--det-file",
        type=Path,
        required=True,
        help="Path to the detection file.",
    )
    parser.add_argument(
        "--det-format",
        choices=SUPPORTED_DETECTION_FORMATS,
        default="mot_tlwh",
        help=(
            "Detection-file column format. "
            "'mot_tlwh' expects "
            "frame,id,left,top,width,height,confidence,...; "
            "'xyxy' expects frame,x1,y1,x2,y2,confidence."
        ),
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        required=True,
        help="First frame number to process, inclusive.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        required=True,
        help="Last frame number to process, inclusive.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=30.0,
        help="Sequence frame rate used to scale the lost-track buffer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used by SAM and Cutie in the mask-conditioned run.",
    )
    parser.add_argument(
        "--enable-cmc",
        action="store_true",
        help="Enable camera motion compensation in both runs.",
    )
    parser.add_argument(
        "--cmc-method",
        type=str,
        default="sparseOptFlow",
        choices=SUPPORTED_CMC_METHODS,
        help="Camera-motion compensation method.",
    )
    parser.add_argument(
        "--cmc-downscale",
        type=int,
        default=6,
        help="Image downscale factor used by CMC.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=RUN_MODES,
        default=list(RUN_MODES),
        help=(
            "Tracker configurations to run. By default, both the mask-free "
            "locked-IoU baseline and full mask-conditioned McByte are run."
        ),
    )
    parser.add_argument(
        "--enable-isolated-mask-matching",
        action="store_true",
        help=("Allow mask evidence to rescue isolated positive-IoU pairs below the normal association threshold."),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for both comparison runs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate paths, frame range, device, and numeric arguments."""
    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {args.image_dir}")

    if not args.det_file.is_file():
        raise FileNotFoundError(f"Detection file does not exist: {args.det_file}")

    if args.start_frame <= 0:
        raise ValueError("start-frame must be positive.")

    if args.end_frame < args.start_frame:
        raise ValueError("end-frame must be greater than or equal to start-frame.")

    if args.frame_rate <= 0:
        raise ValueError("frame-rate must be positive.")

    if args.cmc_downscale <= 0:
        raise ValueError("cmc-downscale must be positive.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Use --device cpu or install CUDA-enabled PyTorch."
        )


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


def main() -> None:
    """Run locked-IoU and full mask-conditioned McByte sequentially."""
    args = parse_args()
    validate_args(args)

    detections_by_frame = read_detection_file(
        det_file=args.det_file,
        detection_format=cast(DetectionFileFormat, args.det_format),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Image directory: {args.image_dir}")
    print(f"Detection file: {args.det_file}")
    print(f"Detection format: {args.det_format}")
    print(f"Frame range: {args.start_frame} to {args.end_frame} (inclusive)")
    print(f"Output root: {args.output_dir}")

    for mode_name in args.modes:
        run_mode(
            mode_name=mode_name,
            detections_by_frame=detections_by_frame,
            image_dir=args.image_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            output_root=args.output_dir,
            frame_rate=args.frame_rate,
            device=args.device,
            enable_cmc=args.enable_cmc,
            cmc_method=args.cmc_method,
            cmc_downscale=args.cmc_downscale,
            enable_isolated_mask_matching=(args.enable_isolated_mask_matching),
        )

    print("\nFinished selected McByte comparison run(s).")


if __name__ == "__main__":
    main()
