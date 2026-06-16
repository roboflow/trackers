# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Visual sanity check for CutieMaskPropagator.

This script is intended for local/manual validation only. It does not bundle any
image assets and does not run as part of the test suite. The user provides an
image directory, a frame range, and one or more bounding boxes. The script uses
SAM to initialize masks on the first selected frame, then uses Cutie to propagate
those masks over the remaining selected frames.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from trackers.core.mcbyte.masks.base import TrackletSnapshot
from trackers.core.mcbyte.masks.cutie import CutieMaskPropagator
from trackers.core.mcbyte.masks.sam import SAMBoxMaskGenerator

DEFAULT_OUTPUT_ROOT = Path("visual_tests/outputs/cutie")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class AddMaskEvent:
    """Manual add-mask event parsed from ``--add-at``.

    ``frame_file`` is the frame where the provided box is valid. The script
    applies the mask on that frame and propagates it to the next frame.
    """

    frame_file: str
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class RemoveMaskEvent:
    """Manual remove-mask event parsed from ``--remove-at``.

    Removal happens before propagating to ``frame_file``.
    """

    frame_file: str
    tracker_id: int


def parse_xyxy_box(box: str) -> tuple[float, float, float, float]:
    """Parse one command-line bounding box in ``x1,y1,x2,y2`` format."""
    values = [float(value) for value in box.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Each box must contain exactly 4 comma-separated values: x1,y1,x2,y2.")
    return values[0], values[1], values[2], values[3]


def parse_add_mask_event(event: str) -> AddMaskEvent:
    """Parse ``filename:x1,y1,x2,y2`` add-mask event."""
    parts = event.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Add event must have format filename:x1,y1,x2,y2.")

    frame_file, box_str = parts
    return AddMaskEvent(
        frame_file=frame_file,
        xyxy=parse_xyxy_box(box_str),
    )


def parse_remove_mask_event(event: str) -> RemoveMaskEvent:
    """Parse ``filename:manual_mask_id`` remove-mask event."""
    parts = event.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Remove event must have format filename:manual_mask_id.")

    frame_file, tracker_id_str = parts
    return RemoveMaskEvent(
        frame_file=frame_file,
        tracker_id=int(tracker_id_str),
    )


def group_add_events_by_next_frame(
    frame_paths: list[Path],
    events: list[AddMaskEvent],
) -> dict[str, list[AddMaskEvent]]:
    """Group add events by the frame they should be applied before.

    The CLI frame is the frame where the box is valid. Internally, McByte-style
    timing applies the mask on that frame, then propagates to the next frame.
    """
    filenames = [path.name for path in frame_paths]
    grouped: dict[str, list[AddMaskEvent]] = defaultdict(list)

    for event in events:
        source_index = filenames.index(event.frame_file)
        target_file = filenames[source_index + 1]
        grouped[target_file].append(event)

    return dict(grouped)


def group_remove_events(events: list[RemoveMaskEvent]) -> dict[str, list[int]]:
    """Group remove events by the frame before which they should be applied."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for event in events:
        grouped[event.frame_file].append(event.tracker_id)
    return dict(grouped)


def validate_lifecycle_events(
    frame_paths: list[Path],
    add_events: list[AddMaskEvent],
    remove_events: list[RemoveMaskEvent],
) -> None:
    """Validate that add/remove lifecycle events are compatible with the frame range.

    Add events refer to the source frame where the box is valid and are internally
    shifted to the next frame. Therefore, they cannot be scheduled on the last
    selected frame. Remove events are applied before propagation to their target
    frame, so they cannot be scheduled on the first selected frame.
    """
    filenames = [path.name for path in frame_paths]
    filename_set = set(filenames)
    first_file = filenames[0]

    for event in add_events:
        if event.frame_file not in filename_set:
            raise ValueError(f"Add event frame is outside selected range: {event.frame_file}")
        if event.frame_file == filenames[-1]:
            raise ValueError(
                "Add events cannot be scheduled on the last selected frame, because "
                "they are applied on that frame and propagated to the next one."
            )

    for event in remove_events:
        if event.frame_file not in filename_set:
            raise ValueError(f"Remove event frame is outside selected range: {event.frame_file}")
        if event.frame_file == first_file:
            raise ValueError("Remove events cannot be scheduled on the first selected frame.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the visualization script."""
    parser = argparse.ArgumentParser(description="Visualize SAM initialization and Cutie mask propagation.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing input frames.",
    )
    parser.add_argument(
        "--start-file",
        type=str,
        required=True,
        help="First frame filename, included in the selected frame range.",
    )
    parser.add_argument(
        "--end-file",
        type=str,
        required=True,
        help="Last frame filename, included in the selected frame range.",
    )
    parser.add_argument(
        "--box",
        type=parse_xyxy_box,
        action="append",
        required=True,
        help=(
            "Bounding box on the first selected frame in xyxy format: x1,y1,x2,y2. "
            "Pass this argument multiple times for multiple boxes."
        ),
    )
    parser.add_argument(
        "--add-at",
        type=parse_add_mask_event,
        action="append",
        default=[],
        help=(
            "Add a new mask using a box on the given frame. Format: "
            "filename:x1,y1,x2,y2. Can be passed multiple times. "
            "The box is applied on this frame, then Cutie propagates to the next "
            "frame, following McByte timing. "
            "In this standalone script, every --add-at event is treated as a new "
            "object. Do not add a mask for an object that already has one. In the full "
            "McByte pipeline this is handled by the tracker, but this visual script "
            "does not know object identity beyond the manual IDs."
        ),
    )
    parser.add_argument(
        "--remove-at",
        type=parse_remove_mask_event,
        action="append",
        default=[],
        help=(
            "Remove a mask before propagating to the given frame. Format: "
            "filename:manual_mask_id. Can be passed multiple times. "
            "Manual mask IDs are assigned automatically."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for timestamped outputs. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device used by SAM and Cutie, for example "cpu" or "cuda".',
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_b",
        help='SAM model type. Defaults to "vit_b".',
    )
    parser.add_argument(
        "--cutie-model-type",
        type=str,
        default="base-mega",
        help='Cutie model type. Defaults to "base-mega".',
    )
    parser.add_argument(
        "--cutie-config-path",
        type=Path,
        default=None,
        help="Optional path to Cutie's Hydra config directory.",
    )
    parser.add_argument(
        "--cutie-config-name",
        type=str,
        default="eval_config",
        help='Cutie Hydra config name. Defaults to "eval_config".',
    )
    return parser.parse_args()


def list_selected_frame_paths(
    image_dir: Path,
    start_file: str,
    end_file: str,
) -> list[Path]:
    """List sorted frame paths from ``start_file`` to ``end_file`` inclusive."""
    frame_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    filenames = [path.name for path in frame_paths]

    if start_file not in filenames:
        raise FileNotFoundError(f"Start file not found in {image_dir}: {start_file}")
    if end_file not in filenames:
        raise FileNotFoundError(f"End file not found in {image_dir}: {end_file}")

    start_index = filenames.index(start_file)
    end_index = filenames.index(end_file)
    if end_index < start_index:
        raise ValueError(f"end-file must not come before start-file. Got {start_file=} and {end_file=}.")

    return frame_paths[start_index : end_index + 1]


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image from disk and return it in RGB format."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


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


def color_from_id(object_id: int) -> np.ndarray:
    """Return a deterministic RGB color for a stable object/manual ID."""
    rng = np.random.default_rng(object_id)
    return rng.integers(0, 255, size=3, dtype=np.uint8)


def overlay_masks(
    image: np.ndarray,
    masks: np.ndarray,
    object_ids: list[int],
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay binary masks on an RGB image using stable object-ID colors."""
    if masks.shape[0] != len(object_ids):
        raise ValueError(
            "Number of masks must match number of object IDs. "
            f"Got {masks.shape[0]} masks and {len(object_ids)} object IDs."
        )

    output = image.copy()

    for mask, object_id in zip(masks, object_ids):
        color = color_from_id(object_id)
        colored_mask = np.zeros_like(output)
        colored_mask[mask] = color

        output = np.where(
            mask[..., None],
            (alpha * colored_mask + (1.0 - alpha) * output).astype(np.uint8),
            output,
        )

    return output


# In CutieMaskPropagator, tracklet_mask_dict maps manual/tracklet IDs to immutable
# Cutie object IDs, and masks are produced in _object_ids order.
def get_mask_object_ids_in_order(tracklet_mask_dict: dict[int, int]) -> list[int]:
    """Return object IDs in the same order as MaskOutput.masks."""
    return [
        object_id
        for _, object_id in sorted(
            tracklet_mask_dict.items(),
            key=lambda item: item[1],
        )
    ]


def draw_boxes(
    image: np.ndarray,
    tracklets: list[TrackletSnapshot],
) -> np.ndarray:
    """Draw tracklet bounding boxes and tracker IDs on an RGB image."""
    output = image.copy()

    for tracklet in tracklets:
        x1, y1, x2, y2 = tracklet.xyxy.astype(int)
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color=(91, 10, 145),
            thickness=2,
        )
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


def draw_frame_label(
    image: np.ndarray,
    frame_name: str,
) -> np.ndarray:
    """Draw the frame filename in the top-left corner."""
    output = image.copy()

    cv2.putText(
        output,
        frame_name,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,  # white outline
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        frame_name,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        1,  # black fill
        cv2.LINE_AA,
    )

    return output


def save_rgb_image(
    image_rgb: np.ndarray,
    output_path: Path,
) -> None:
    """Save an RGB image to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)


def validate_device(
    device: str,
    label: str,
) -> str:
    """Validate requested execution device."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested for {label}, but torch.cuda.is_available() is False. "
            "Use a CPU device or install a CUDA-enabled PyTorch build."
        )
    return device


def print_device_info(
    label: str,
    device: torch.device,
) -> None:
    """Print execution device information."""
    print(f"{label} device: {device}")
    if device.type == "cuda":
        print(f"{label} GPU: {torch.cuda.get_device_name(device)} (CUDA {torch.version.cuda})")
    else:
        print(f"{label} GPU: N/A (running on CPU)")


def main() -> None:
    """Run SAM initialization, Cutie propagation, and save visualizations."""
    args = parse_args()

    frame_paths = list_selected_frame_paths(
        image_dir=args.image_dir,
        start_file=args.start_file,
        end_file=args.end_file,
    )
    if len(frame_paths) < 2:
        raise ValueError("At least two frames are required for Cutie propagation.")

    validate_lifecycle_events(
        frame_paths=frame_paths,
        add_events=args.add_at,
        remove_events=args.remove_at,
    )
    add_events_by_file = group_add_events_by_next_frame(
        frame_paths=frame_paths,
        events=args.add_at,
    )
    remove_events_by_file = group_remove_events(args.remove_at)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_frame = load_rgb_image(frame_paths[0])

    tracklets = [
        TrackletSnapshot(
            tracker_id=index + 1,
            xyxy=validate_and_clip_xyxy_box(
                box=box,
                image_shape=initial_frame.shape[:2],
            ),
        )
        for index, box in enumerate(args.box)
    ]
    next_manual_tracklet_id = len(tracklets) + 1
    print("Initial manual mask IDs:")
    for tracklet in tracklets:
        print(f"  {tracklet.tracker_id}: initial box {tracklet.xyxy.tolist()}")
    print(
        "Note: each --add-at event is treated as a new object. "
        "Do not add another mask for an already initialized object."
    )

    device = validate_device(args.device, label="SAM/Cutie")

    sam_generator = SAMBoxMaskGenerator(
        model_type=args.sam_model_type,
        device=device,
    )
    cutie_propagator = CutieMaskPropagator(
        model_type=args.cutie_model_type,
        config_path=args.cutie_config_path,
        config_name=args.cutie_config_name,
        device=device,
    )

    print_device_info("SAM/Cutie", sam_generator.device)
    print(f"Selected {len(frame_paths)} frames.")
    print(f"Saving outputs to {output_dir}")

    initial_mask_output = sam_generator.generate(
        frame=initial_frame,
        tracklets=tracklets,
    )
    if initial_mask_output.masks is None:
        raise RuntimeError("SAM did not return initialization masks.")

    cutie_propagator.initialize(
        frame=initial_frame,
        mask_output=initial_mask_output,
    )

    initial_visual = overlay_masks(
        image=initial_frame,
        masks=initial_mask_output.masks,
        # Use manual IDs for the first-frame SAM masks so their colors match the later
        # Cutie outputs. SAM local mask indices start at 0, while Cutie object IDs start
        # at 1.
        object_ids=[tracklet.tracker_id for tracklet in tracklets],
    )
    initial_visual = draw_boxes(initial_visual, tracklets)
    initial_visual = draw_frame_label(
        initial_visual,
        frame_paths[0].name,
    )

    save_rgb_image(initial_visual, output_dir / frame_paths[0].name)
    print(f"Saved {frame_paths[0].name} (SAM)")

    previous_frame = initial_frame
    previous_frame_path = frame_paths[0]

    for frame_path in frame_paths[1:]:
        frame = load_rgb_image(frame_path)

        add_events = add_events_by_file.get(frame_path.name, [])
        if len(add_events) > 0:
            add_tracklets = []
            for event in add_events:
                manual_tracklet_id = next_manual_tracklet_id
                next_manual_tracklet_id += 1

                tracklet = TrackletSnapshot(
                    tracker_id=manual_tracklet_id,
                    xyxy=validate_and_clip_xyxy_box(
                        box=event.xyxy,
                        image_shape=previous_frame.shape[:2],
                    ),
                )
                add_tracklets.append(tracklet)

                print(
                    f"Scheduled add from {previous_frame_path.name} before propagating "
                    f"to {frame_path.name}: manual ID {manual_tracklet_id}, "
                    f"box {tracklet.xyxy.tolist()}"
                )

            add_mask_output = sam_generator.generate(
                frame=previous_frame,
                tracklets=add_tracklets,
            )
            cutie_propagator.add_masks(
                frame=previous_frame,
                mask_output=add_mask_output,
            )

            print(
                f"Added {len(add_tracklets)} mask(s) before {frame_path.name} "
                f"using previous frame {previous_frame_path.name}."
            )

        remove_tracklet_ids = remove_events_by_file.get(frame_path.name, [])
        if len(remove_tracklet_ids) > 0:
            cutie_propagator.remove_masks(remove_tracklet_ids)
            print(f"Removed mask(s) for manual IDs {remove_tracklet_ids} before {frame_path.name}")

        propagated_mask_output = cutie_propagator.propagate(frame)

        if propagated_mask_output is None or propagated_mask_output.masks is None:
            raise RuntimeError(f"Cutie did not return masks for frame: {frame_path}")

        visual = overlay_masks(
            image=frame,
            masks=propagated_mask_output.masks,
            object_ids=get_mask_object_ids_in_order(propagated_mask_output.tracklet_mask_dict),
        )
        visual = draw_frame_label(visual, frame_path.name)
        save_rgb_image(visual, output_dir / frame_path.name)

        print(
            f"Saved {frame_path.name} (Cutie); "
            f"tracklet_mask_dict={propagated_mask_output.tracklet_mask_dict}; "
            f"mask_avg_prob_dict={propagated_mask_output.mask_avg_prob_dict}"
        )

        previous_frame = frame
        previous_frame_path = frame_path

    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
