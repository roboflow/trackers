# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Ground-truth replay mode for ``trackers inspect mask-manager``.

Private implementation module. The user-facing command lives in
:mod:`trackers.cli.inspect.mask_manager`, which dispatches here when
``--mode gt`` is given.

Per-frame tracklet boxes are replayed from a MOT-style ground-truth file and fed
into :class:`MaskManager` with McByte timing:

1. MaskManager produces masks for frame t from tracker state at frame t-1.
2. GT tracklets for frame t are read and treated as tracker output.
3. Newly visible and disappeared GT tracklets are stored as lifecycle events for
   the next frame.

Intended for manual validation of delayed mask creation, pending tracklets, SAM
initialization/addition, Cutie propagation, and removal behavior.

The rendering helpers here deliberately duplicate names used in
:mod:`trackers.cli.inspect.mask_manager`. They are not the same functions: this
mode draws lifecycle-aware box colors (new / pending / masked / plain) because
GT replay knows that state, while manual mode draws one flat color. Unifying
them would change what each mode's output looks like.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from trackers.cli.inspect._common import (
    load_rgb_image,
    save_rgb_image,
    timestamped_run_dir,
    validate_device,
)
from trackers.core.masks.base import MaskOutput, TrackletSnapshot

if TYPE_CHECKING:
    from jsonargparse.typing import PositiveInt


@dataclass(frozen=True)
class FrameTrackletState:
    """Tracklet lifecycle state produced for one replayed frame."""

    tracklets: list[TrackletSnapshot]
    new_tracklets: list[TrackletSnapshot]
    removed_tracklet_ids: list[int]


def frame_number_to_filename(frame_number: int) -> str:
    """Convert integer frame number to SoccerNet/MOT image filename."""
    return f"{frame_number:06d}.jpg"


def validate_frame_range(start_frame: int, end_frame: int) -> None:
    """Validate requested frame range."""
    if start_frame <= 0:
        raise ValueError("start-frame must be positive.")
    if end_frame < start_frame:
        raise ValueError("end-frame must not be smaller than start-frame.")


def parse_selected_tracklet_ids(tracklet_ids: list[int | str] | None) -> set[int] | None:
    """Return selected tracklet IDs, or ``None`` when all should be selected."""
    if tracklet_ids is None:
        return None

    if "all" in tracklet_ids:
        return None

    return {tracklet_id for tracklet_id in tracklet_ids if isinstance(tracklet_id, int)}


def read_gt_tracklets(
    gt_file: Path,
    selected_tracklet_ids: set[int] | None,
) -> dict[int, list[TrackletSnapshot]]:
    """Read MOT-style GT annotations grouped by frame number.

    Expected line format:
    ``frame_no,tracklet_id,left,top,width,height,...``.
    """
    tracklets_by_frame: dict[int, list[TrackletSnapshot]] = defaultdict(list)

    with gt_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            values = line.split(",")
            if len(values) < 6:
                raise ValueError(f"Invalid GT line with fewer than 6 columns: {line}")

            frame_number = int(values[0])
            tracklet_id = int(values[1])

            if selected_tracklet_ids is not None and tracklet_id not in selected_tracklet_ids:
                continue

            left = float(values[2])
            top = float(values[3])
            width = float(values[4])
            height = float(values[5])

            xyxy = np.array(
                [
                    left,
                    top,
                    left + width,
                    top + height,
                ],
                dtype=np.float32,
            )

            tracklets_by_frame[frame_number].append(
                TrackletSnapshot(
                    tracker_id=tracklet_id,
                    xyxy=xyxy,
                )
            )

    return dict(tracklets_by_frame)


def build_frame_tracklet_state(
    current_tracklets: list[TrackletSnapshot],
    previous_visible_tracklet_ids: set[int],
) -> FrameTrackletState:
    """Build visible/new/removed lifecycle state for one replayed frame."""
    current_tracklet_ids = {tracklet.tracker_id for tracklet in current_tracklets}
    new_tracklet_ids = current_tracklet_ids - previous_visible_tracklet_ids
    removed_tracklet_ids = sorted(previous_visible_tracklet_ids - current_tracklet_ids)

    new_tracklets = [tracklet for tracklet in current_tracklets if tracklet.tracker_id in new_tracklet_ids]

    return FrameTrackletState(
        tracklets=current_tracklets,
        new_tracklets=new_tracklets,
        removed_tracklet_ids=removed_tracklet_ids,
    )


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

    The ordering of ``tracklet_ids`` must match the ordering of ``masks``.
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


def draw_boxes(
    image: np.ndarray,
    tracklets: list[TrackletSnapshot],
    pending_tracklet_ids: set[int],
    new_tracklet_ids: set[int],
    masked_tracklet_ids: set[int],
) -> np.ndarray:
    """Draw GT tracklet boxes with lifecycle-aware colors.

    Colors in RGB:
    - blue: newly visible tracklet,
    - yellow: pending mask creation,
    - green: tracklet already has a mask,
    - purple: visible GT tracklet that is not currently masked, pending, or new
    """
    output = image.copy()

    for tracklet in tracklets:
        x1, y1, x2, y2 = tracklet.xyxy.astype(int)

        if tracklet.tracker_id in new_tracklet_ids:
            color = (0, 128, 255)
        elif tracklet.tracker_id in pending_tracklet_ids:
            color = (255, 255, 0)
        elif tracklet.tracker_id in masked_tracklet_ids:
            color = (0, 255, 0)
        else:
            color = (180, 80, 200)

        cv2.rectangle(output, (x1, y1), (x2, y2), color=color, thickness=2)
        cv2.putText(
            output,
            str(tracklet.tracker_id),
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def draw_text_panel(
    image: np.ndarray,
    lines: list[str],
) -> np.ndarray:
    """Draw a semi-readable status panel in the top-left corner."""
    output = image.copy()
    x = 20
    y = 30
    line_height = 24

    for line_index, line in enumerate(lines):
        y_position = y + line_index * line_height
        cv2.putText(output, line, (x, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(output, line, (x, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

    return output


def get_masked_tracklet_ids(mask_output: MaskOutput | None) -> set[int]:
    """Return tracklet IDs currently represented in a mask output."""
    if mask_output is None or mask_output.masks is None:
        return set()
    return set(mask_output.tracklet_mask_dict)


def visualize_output(
    frame: np.ndarray,
    frame_number: int,
    mask_output: MaskOutput | None,
    current_tracklets: list[TrackletSnapshot],
    current_new_tracklets: list[TrackletSnapshot],
    pending_tracklet_ids: set[int],
    removed_tracklet_ids_from_previous_frame: list[int],
    output_path: Path,
) -> None:
    """Overlay masks, GT boxes, lifecycle status, and save the frame."""
    visual = frame.copy()

    masked_tracklet_ids = get_masked_tracklet_ids(mask_output)
    if mask_output is not None and mask_output.masks is not None:
        tracklet_ids = get_mask_tracklet_ids_in_order(mask_output.tracklet_mask_dict)
        visual = overlay_masks(
            image=visual,
            masks=mask_output.masks,
            tracklet_ids=tracklet_ids,
        )

    current_tracklet_ids = [tracklet.tracker_id for tracklet in current_tracklets]
    current_new_tracklet_ids = {tracklet.tracker_id for tracklet in current_new_tracklets}

    visual = draw_boxes(
        image=visual,
        tracklets=current_tracklets,
        pending_tracklet_ids=pending_tracklet_ids,
        new_tracklet_ids=current_new_tracklet_ids,
        masked_tracklet_ids=masked_tracklet_ids,
    )

    status_lines = [
        f"Frame: {frame_number}",
        f"Visible: {current_tracklet_ids}",
        f"New: {sorted(current_new_tracklet_ids)}",
        f"Removed prev: {removed_tracklet_ids_from_previous_frame}",
        f"Pending: {sorted(pending_tracklet_ids)}",
        f"Masks: {sorted(masked_tracklet_ids)}",
    ]
    visual = draw_text_panel(visual, status_lines)
    save_rgb_image(visual, output_path)


def run_gt_mode(
    image_dir: Path,
    gt_file: Path,
    start_frame: int,
    end_frame: int,
    output_root: Path,
    tracklet_id: list[PositiveInt | Literal["all"]] | None = None,
    device: str = "cuda",
    sam_model_type: str = "vit_b",
    cutie_model_type: str = "base-mega",
    cutie_config_path: Path | None = None,
    cutie_config_name: str = "eval_config",
    mask_creation_bbox_overlap_threshold: float = 0.6,
) -> int:
    """Replay GT tracklets through MaskManager and save per-frame outputs.

    The script follows the original McByte execution order.

    For frame ``t``:

        1. MaskManager produces masks for frame ``t`` using tracker outputs
           from frame ``t-1``.
        2. The tracker is emulated by reading ground-truth tracklets for frame
           ``t`` and treating them as tracker outputs.
        3. Newly visible and disappeared GT tracklets are stored as lifecycle
           events consumed by MaskManager on frame ``t+1``.

    This mirrors the timing used by the original McByte implementation while
    using ground-truth boxes and IDs to make delayed mask creation easier to
    inspect visually.

    Every option is spelled with underscores here, but hyphens work just as
    well on the command line: ``--image-dir`` and ``--image_dir`` are the same
    option.

    ``tracklet_id`` is a list, so it takes the bracket syntax rather than a
    repeated option: ``--tracklet_id=[3,7]`` selects two tracklets and
    ``--tracklet_id+=9`` appends a third.

    Args:
        image_dir: Directory containing input frames.
        gt_file: MOT-style GT file.
        start_frame: First frame number, inclusive.
        end_frame: Last frame number, inclusive.
        tracklet_id: Tracklet IDs to replay, e.g. ``[3,7]``. Omit this option,
            or pass ``--tracklet_id=[all]``, to replay all tracklets.
        output_root: Root directory for timestamped outputs.
        device: Device used by SAM and Cutie, for example ``cpu`` or ``cuda``.
        sam_model_type: SAM model type.
        cutie_model_type: Cutie model type.
        cutie_config_path: Optional path to Cutie's Hydra config directory.
        cutie_config_name: Cutie Hydra config name.
        mask_creation_bbox_overlap_threshold: Overlap threshold above which
            mask creation is delayed.

    Returns:
        Exit code: ``0`` on success, ``1`` on a validation error.
    """
    try:
        validate_frame_range(start_frame, end_frame)
        selected_tracklet_ids = parse_selected_tracklet_ids(tracklet_id)
        tracklets_by_frame = read_gt_tracklets(
            gt_file=gt_file,
            selected_tracklet_ids=selected_tracklet_ids,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    output_dir = timestamped_run_dir(output_root)

    try:
        device = validate_device(device, label="SAM/Cutie")
    except (ImportError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    from trackers.core.masks.cutie import CutieMaskPropagator
    from trackers.core.masks.manager import MaskManager
    from trackers.core.masks.sam import SAMBoxMaskGenerator

    mask_manager = MaskManager(
        mask_generator=SAMBoxMaskGenerator(
            model_type=sam_model_type,
            device=device,
        ),
        mask_propagator=CutieMaskPropagator(
            model_type=cutie_model_type,
            config_path=cutie_config_path,
            config_name=cutie_config_name,
            device=device,
        ),
        mask_creation_bbox_overlap_threshold=mask_creation_bbox_overlap_threshold,
    )

    print(f"Selected tracklets: {'all' if selected_tracklet_ids is None else sorted(selected_tracklet_ids)}")
    print(f"Selected frame range: {start_frame} to {end_frame}")
    print(f"Saving outputs to {output_dir}")

    previous_frame: np.ndarray | None = None
    previous_tracklets: list[TrackletSnapshot] = []
    previous_new_tracklets: list[TrackletSnapshot] = []
    previous_removed_tracklet_ids: list[int] = []
    previous_visible_tracklet_ids: set[int] = set()

    for frame_number in range(start_frame, end_frame + 1):
        frame_path = image_dir / frame_number_to_filename(frame_number)
        frame = load_rgb_image(frame_path)

        mask_output = mask_manager.get_updated_masks(
            frame=frame,
            previous_frame=previous_frame,
            previous_tracklets=previous_tracklets,
            new_tracklets=previous_new_tracklets,
            removed_tracklet_ids=previous_removed_tracklet_ids,
        )

        current_tracklets = tracklets_by_frame.get(frame_number, [])
        current_state = build_frame_tracklet_state(
            current_tracklets=current_tracklets,
            previous_visible_tracklet_ids=previous_visible_tracklet_ids,
        )

        visualize_output(
            frame=frame,
            frame_number=frame_number,
            mask_output=mask_output,
            current_tracklets=current_state.tracklets,
            current_new_tracklets=current_state.new_tracklets,
            pending_tracklet_ids=mask_manager._pending_tracklet_ids,
            removed_tracklet_ids_from_previous_frame=previous_removed_tracklet_ids,
            output_path=output_dir / frame_path.name,
        )

        print(
            f"Frame {frame_number}: "
            f"visible={sorted(tracklet.tracker_id for tracklet in current_state.tracklets)}, "
            f"new={sorted(tracklet.tracker_id for tracklet in current_state.new_tracklets)}, "
            f"removed={current_state.removed_tracklet_ids}, "
            f"pending={sorted(mask_manager._pending_tracklet_ids)}, "
            f"masks={sorted(get_masked_tracklet_ids(mask_output))}"
        )

        previous_frame = frame
        previous_tracklets = current_state.tracklets
        previous_new_tracklets = current_state.new_tracklets
        previous_removed_tracklet_ids = current_state.removed_tracklet_ids
        previous_visible_tracklet_ids = {tracklet.tracker_id for tracklet in current_state.tracklets}

    print(f"Done. Outputs saved to {output_dir.resolve()}")
    return 0
