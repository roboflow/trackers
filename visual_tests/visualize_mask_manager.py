# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Visual sanity check for McByte MaskManager with SAM + Cutie.

This script validates the real MaskManager orchestration:
- initialize masks from previous-frame tracklets,
- propagate masks to the current frame,
- add masks for new tracklets,
- remove masks for terminated tracklets.

It intentionally calls MaskManager.get_updated_masks(), not Cutie directly.
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

from trackers.core.mcbyte.mask_manager import MaskManager
from trackers.core.mcbyte.masks.base import MaskOutput, TrackletSnapshot
from trackers.core.mcbyte.masks.cutie import CutieMaskPropagator
from trackers.core.mcbyte.masks.sam import SAMBoxMaskGenerator

DEFAULT_OUTPUT_ROOT = Path("visual_tests/outputs/mask_manager")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class AddTrackletEvent:
    """Manual new-tracklet event parsed from ``--add-at``."""

    frame_file: str
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class RemoveTrackletEvent:
    """Manual removed-tracklet event parsed from ``--remove-at``."""

    frame_file: str
    tracker_id: int


def parse_xyxy_box(box: str) -> tuple[float, float, float, float]:
    """Parse one command-line bounding box in ``x1,y1,x2,y2`` format."""
    values = [float(value) for value in box.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Each box must contain exactly 4 comma-separated values: x1,y1,x2,y2.")
    return values[0], values[1], values[2], values[3]


def parse_add_tracklet_event(event: str) -> AddTrackletEvent:
    """Parse ``filename:x1,y1,x2,y2`` add-tracklet event."""
    parts = event.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Add event must have format filename:x1,y1,x2,y2.")

    frame_file, box_str = parts
    try:
        xyxy = parse_xyxy_box(box_str)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("Add event must have format filename:x1,y1,x2,y2.") from exc

    return AddTrackletEvent(frame_file=frame_file, xyxy=xyxy)


def parse_remove_tracklet_event(event: str) -> RemoveTrackletEvent:
    """Parse ``filename:tracklet_id`` remove-tracklet event."""
    parts = event.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Remove event must have format filename:tracklet_id.")

    frame_file, tracker_id_str = parts
    try:
        tracker_id = int(tracker_id_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Remove event must have format filename:tracklet_id.") from exc

    if tracker_id <= 0:
        raise argparse.ArgumentTypeError("tracklet_id must be a positive integer.")

    return RemoveTrackletEvent(frame_file=frame_file, tracker_id=tracker_id)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the MaskManager visualizer."""
    parser = argparse.ArgumentParser(description="Visualize McByte MaskManager with SAM + Cutie.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--start-file", type=str, required=True)
    parser.add_argument("--end-file", type=str, required=True)
    parser.add_argument(
        "--box",
        type=parse_xyxy_box,
        action="append",
        required=True,
        help="Initial tracklet box on the first selected frame: x1,y1,x2,y2. Can be repeated.",
    )
    parser.add_argument(
        "--add-at",
        type=parse_add_tracklet_event,
        action="append",
        default=[],
        help=(
            "Add a new tracklet from the given frame box. Format: filename:x1,y1,x2,y2. "
            "The box is treated as a tracker output on that frame and added by MaskManager "
            "before propagating to the next frame."
        ),
    )
    parser.add_argument(
        "--remove-at",
        type=parse_remove_tracklet_event,
        action="append",
        default=[],
        help=(
            "Remove a tracklet before propagating to the given frame. "
            "Format: filename:tracklet_id."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--cutie-model-type", type=str, default="base-mega")
    parser.add_argument("--cutie-config-path", type=Path, default=None)
    parser.add_argument("--cutie-config-name", type=str, default="eval_config")
    return parser.parse_args()


def list_selected_frame_paths(image_dir: Path, start_file: str, end_file: str) -> list[Path]:
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


def validate_device(device: str) -> str:
    """Validate that the requested execution device is available."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Use --device cpu or install CUDA-enabled PyTorch."
        )
    return device


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


def save_rgb_image(image_rgb: np.ndarray, output_path: Path) -> None:
    """Save an RGB image to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)


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


def main() -> None:
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
    """
    args = parse_args()

    frame_paths = list_selected_frame_paths(
        image_dir=args.image_dir,
        start_file=args.start_file,
        end_file=args.end_file,
    )
    if len(frame_paths) < 2:
        raise ValueError("At least two frames are required for MaskManager validation.")

    validate_lifecycle_events(
        frame_paths=frame_paths,
        add_events=args.add_at,
        remove_events=args.remove_at,
    )

    add_events_by_file = group_add_events_by_source_frame(frame_paths, args.add_at)
    remove_events_by_file = group_remove_events(args.remove_at)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    device = validate_device(args.device)

    mask_manager = MaskManager(
        mask_generator=SAMBoxMaskGenerator(
            model_type=args.sam_model_type,
            device=device,
        ),
        mask_propagator=CutieMaskPropagator(
            model_type=args.cutie_model_type,
            config_path=args.cutie_config_path,
            config_name=args.cutie_config_name,
            device=device,
        ),
    )

    first_frame = load_rgb_image(frame_paths[0])

    active_tracklets: dict[int, TrackletSnapshot] = {
        index + 1: TrackletSnapshot(
            tracker_id=index + 1,
            xyxy=validate_and_clip_xyxy_box(
                box=box,
                image_shape=first_frame.shape[:2],
            ),
        )
        for index, box in enumerate(args.box)
    }
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

        add_events = add_events_by_file.get(frame_path.name, [])
        for event in add_events:
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

            print(
                f"Scheduled add from {frame_path.name}: "
                f"tracklet {tracklet_id}, box {tracklet.xyxy.tolist()}"
            )


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

    print(f"Done. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
