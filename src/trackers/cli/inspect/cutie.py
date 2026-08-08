# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Visual sanity check for :class:`CutieMaskPropagator`.

Intended for local validation only. No image assets are bundled and nothing here
runs as part of the test suite. The caller supplies an image directory, a frame
range, and one or more bounding boxes. SAM initializes masks on the first
selected frame, then Cutie propagates them over the remaining selected frames.

Usage
-----

::

    trackers inspect cutie \\
        --image_dir frames --start_file 000001.jpg --end_file 000010.jpg \\
        --box='[[10,20,110,220]]'

Options come from the :func:`cutie_command` signature, parsed with jsonargparse
through the shared ``trackers`` parser, so the shared conventions hold:
``--image-dir`` and ``--image_dir`` are the same option, and ``--config
run.yaml`` supplies the same keys from a file. The repeatable options are
lists: boxes are ``--box='[[x1,y1,x2,y2]]'``, while lifecycle events are
``--add_at='["frame.jpg:x1,y1,x2,y2"]'`` and ``--remove_at='["frame.jpg:3"]'``.
Every one of them also appends with ``+``, as in
``--add_at+ frame.jpg:10,20,110,220``.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from trackers.cli.inspect._common import (
    INSPECT_OUTPUT_ROOT,
    list_selected_frame_paths,
    load_rgb_image,
    parse_xyxy_box,
    print_device_info,
    save_rgb_image,
    timestamped_run_dir,
    validate_device,
)
from trackers.core.masks.base import TrackletSnapshot

if TYPE_CHECKING:
    from trackers.core.masks.cutie import CutieMaskPropagator
    from trackers.core.masks.sam import SAMBoxMaskGenerator

DEFAULT_OUTPUT_ROOT = INSPECT_OUTPUT_ROOT / "cutie"


@dataclass(frozen=True)
class AddMaskEvent:
    """Manual add-mask event supplied through ``--add_at``.

    ``frame_file`` is the frame where the provided box is valid. The script
    applies the mask on that frame and propagates it to the next frame.

    Attributes:
        frame_file: Frame filename on which the box is valid.
        xyxy: Bounding box on that frame as ``[x1,y1,x2,y2]``.
    """

    frame_file: str
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class RemoveMaskEvent:
    """Manual remove-mask event supplied through ``--remove_at``.

    Removal happens before propagating to ``frame_file``.

    Attributes:
        frame_file: Frame filename before which the mask is removed.
        tracker_id: Manual mask ID to remove. Must be positive.
    """

    frame_file: str
    tracker_id: int


def parse_add_mask_event(event: str) -> AddMaskEvent:
    """Parse ``filename:x1,y1,x2,y2`` add-mask event."""
    parts = event.split(":")
    if len(parts) != 2:
        raise ValueError("Add event must have format filename:x1,y1,x2,y2.")

    frame_file, box_str = parts

    try:
        xyxy = parse_xyxy_box(box_str)
    except ValueError as exc:
        raise ValueError("Add event must have format filename:x1,y1,x2,y2.") from exc

    return AddMaskEvent(
        frame_file=frame_file,
        xyxy=xyxy,
    )


def parse_remove_mask_event(event: str) -> RemoveMaskEvent:
    """Parse ``filename:manual_mask_id`` remove-mask event."""
    parts = event.split(":")
    if len(parts) != 2:
        raise ValueError("Remove event must have format filename:manual_mask_id.")

    frame_file, tracker_id_str = parts

    try:
        tracker_id = int(tracker_id_str)
    except ValueError as exc:
        raise ValueError("Remove event must have format filename:manual_mask_id.") from exc

    if tracker_id <= 0:
        raise ValueError("manual_mask_id must be a positive integer.")

    return RemoveMaskEvent(
        frame_file=frame_file,
        tracker_id=tracker_id,
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

    for add_event in add_events:
        if add_event.frame_file not in filename_set:
            raise ValueError(f"Add event frame is outside selected range: {add_event.frame_file}")
        if add_event.frame_file == filenames[-1]:
            raise ValueError(
                "Add events cannot be scheduled on the last selected frame, because "
                "they are applied on that frame and propagated to the next one."
            )

    for remove_event in remove_events:
        if remove_event.frame_file not in filename_set:
            raise ValueError(f"Remove event frame is outside selected range: {remove_event.frame_file}")
        if remove_event.frame_file == first_file:
            raise ValueError("Remove events cannot be scheduled on the first selected frame.")


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


def get_mask_tracklet_ids_in_order(tracklet_mask_dict: dict[int, int]) -> list[int]:
    """Return tracklet/manual IDs in the same order as MaskOutput.masks."""
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


def apply_add_events(
    *,
    sam_generator: SAMBoxMaskGenerator,
    cutie_propagator: CutieMaskPropagator,
    add_events: list[AddMaskEvent],
    previous_frame: np.ndarray,
    previous_frame_path: Path,
    frame_path: Path,
    next_manual_tracklet_id: int,
) -> int:
    """Apply scheduled add-mask events on the previous frame.

    Each event is assigned the next free manual tracklet ID, turned into a SAM
    mask on ``previous_frame``, and handed to Cutie so the mask is available
    when propagation reaches ``frame_path``.

    Args:
        sam_generator: SAM mask generator used to create the new masks.
        cutie_propagator: Cutie propagator receiving the new masks.
        add_events: Events scheduled before ``frame_path``.
        previous_frame: RGB frame the event boxes are valid on.
        previous_frame_path: Path of ``previous_frame``, used for logging.
        frame_path: Frame the new masks are propagated to.
        next_manual_tracklet_id: First unused manual tracklet ID.

    Returns:
        The next unused manual tracklet ID after assigning one per event.
    """
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

    add_mask_output = sam_generator.generate(frame=previous_frame, tracklets=add_tracklets)
    cutie_propagator.add_masks(frame=previous_frame, mask_output=add_mask_output)

    print(
        f"Added {len(add_tracklets)} mask(s) before {frame_path.name} using previous frame {previous_frame_path.name}."
    )
    return next_manual_tracklet_id


def initialize_first_frame(
    *,
    sam_generator: SAMBoxMaskGenerator,
    cutie_propagator: CutieMaskPropagator,
    initial_frame: np.ndarray,
    tracklets: list[TrackletSnapshot],
    output_path: Path,
) -> None:
    """Seed Cutie with SAM masks on the first frame and save its visualization.

    Args:
        sam_generator: SAM mask generator producing the initial masks.
        cutie_propagator: Cutie propagator seeded with those masks.
        initial_frame: First selected RGB frame.
        tracklets: Manually supplied tracklets on the first frame.
        output_path: File the annotated first frame is written to.

    Raises:
        RuntimeError: If SAM returns no initialization masks.
    """
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
        output_path.name,
    )

    save_rgb_image(initial_visual, output_path)
    print(f"Saved {output_path.name} (SAM)")


def propagate_and_save_frame(
    *,
    cutie_propagator: CutieMaskPropagator,
    frame: np.ndarray,
    frame_path: Path,
    remove_events_by_file: dict[str, list[int]],
    output_path: Path,
) -> None:
    """Apply scheduled removals, propagate masks to one frame, and save it.

    Args:
        cutie_propagator: Cutie propagator holding the current mask state.
        frame: RGB frame to propagate the masks onto.
        frame_path: Path of ``frame``, used for logging and event lookup.
        remove_events_by_file: Manual mask IDs to remove, keyed by the frame
            they are removed before.
        output_path: File the annotated frame is written to.

    Raises:
        RuntimeError: If Cutie returns no masks for the frame.
    """
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
        object_ids=get_mask_tracklet_ids_in_order(propagated_mask_output.tracklet_mask_dict),
    )
    visual = draw_frame_label(visual, frame_path.name)
    save_rgb_image(visual, output_path)

    print(
        f"Saved {frame_path.name} (Cutie); "
        f"tracklet_mask_dict={propagated_mask_output.tracklet_mask_dict}; "
        f"mask_avg_prob_dict={propagated_mask_output.mask_avg_prob_dict}"
    )


def cutie_command(
    image_dir: Path,
    start_file: str,
    end_file: str,
    box: list[tuple[float, float, float, float]],
    add_at: list[str] | None = None,
    remove_at: list[str] | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    device: str = "cuda",
    sam_model_type: str = "vit_b",
    cutie_model_type: str = "base-mega",
    cutie_config_path: Path | None = None,
    cutie_config_name: str = "eval_config",
) -> int:
    """Run SAM initialization, Cutie propagation, and save visualizations.

    Every option is spelled with underscores here, but hyphens work just as
    well on the command line: ``--image-dir`` and ``--image_dir`` are the same
    option.

    Args:
        image_dir: Directory containing input frames.
        start_file: First frame filename, included in the selected frame range.
        end_file: Last frame filename, included in the selected frame range.
        box: Bounding boxes on the first selected frame in xyxy format, given
            as one list rather than a repeated option:
            ``--box='[[x1,y1,x2,y2]]'`` for one box,
            ``--box='[[10,20,110,220],[30,40,130,240]]'`` for two, and
            ``--box+='[[30,40,130,240]]'`` to append to boxes already given.
        add_at: Masks to add using a box on the given frame, each in
            ``filename:x1,y1,x2,y2`` format. The box is applied on that frame,
            then Cutie propagates to the next frame, following McByte timing.
            In this standalone script, every add event is treated as a new
            object. Do not add a mask for an object that already has one. In
            the full McByte pipeline this is handled by the tracker, but this
            visual script does not know object identity beyond the manual IDs.
            Given as one list, ``--add_at='["frame.jpg:10,20,110,220"]'``, or
            appended one at a time with ``--add_at+ frame.jpg:10,20,110,220``.
        remove_at: Masks to remove before propagating to the given frame, each
            in ``filename:manual_mask_id`` format. Manual mask IDs are assigned
            automatically. Given as one list, ``--remove_at='["frame.jpg:3"]'``,
            or appended one at a time with ``--remove_at+ frame.jpg:3``.
        output_root: Root directory for timestamped outputs.
        device: Device used by SAM and Cutie, for example ``cpu`` or ``cuda``.
        sam_model_type: SAM model type.
        cutie_model_type: Cutie model type.
        cutie_config_path: Optional path to Cutie's Hydra config directory.
        cutie_config_name: Cutie Hydra config name.

    Returns:
        Exit code: ``0`` on success, ``1`` on a validation error.
    """
    try:
        add_events = [parse_add_mask_event(event) for event in add_at or []]
        remove_events = [parse_remove_mask_event(event) for event in remove_at or []]

        frame_paths = list_selected_frame_paths(
            image_dir=image_dir,
            start_file=start_file,
            end_file=end_file,
        )
        if len(frame_paths) < 2:
            raise ValueError("At least two frames are required for Cutie propagation.")

        validate_lifecycle_events(
            frame_paths=frame_paths,
            add_events=add_events,
            remove_events=remove_events,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    add_events_by_file = group_add_events_by_next_frame(
        frame_paths=frame_paths,
        events=add_events,
    )
    remove_events_by_file = group_remove_events(remove_events)

    output_dir = timestamped_run_dir(output_root)

    initial_frame = load_rgb_image(frame_paths[0])

    try:
        tracklets = [
            TrackletSnapshot(
                tracker_id=index + 1,
                xyxy=validate_and_clip_xyxy_box(
                    box=initial_box,
                    image_shape=initial_frame.shape[:2],
                ),
            )
            for index, initial_box in enumerate(box)
        ]
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    next_manual_tracklet_id = len(tracklets) + 1
    print("Initial manual mask IDs:")
    for tracklet in tracklets:
        print(f"  {tracklet.tracker_id}: initial box {tracklet.xyxy.tolist()}")
    print(
        "Note: each --add_at event is treated as a new object. "
        "Do not add another mask for an already initialized object."
    )

    try:
        device = validate_device(device, label="SAM/Cutie")
    except (ImportError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    from trackers.core.masks.cutie import CutieMaskPropagator
    from trackers.core.masks.sam import SAMBoxMaskGenerator

    sam_generator = SAMBoxMaskGenerator(
        model_type=sam_model_type,
        device=device,
    )
    cutie_propagator = CutieMaskPropagator(
        model_type=cutie_model_type,
        config_path=cutie_config_path,
        config_name=cutie_config_name,
        device=device,
    )

    print_device_info(sam_generator.device, label="SAM/Cutie")
    print(f"Selected {len(frame_paths)} frames.")
    print(f"Saving outputs to {output_dir}")

    initialize_first_frame(
        sam_generator=sam_generator,
        cutie_propagator=cutie_propagator,
        initial_frame=initial_frame,
        tracklets=tracklets,
        output_path=output_dir / frame_paths[0].name,
    )

    previous_frame = initial_frame
    previous_frame_path = frame_paths[0]

    for frame_path in frame_paths[1:]:
        frame = load_rgb_image(frame_path)

        add_events = add_events_by_file.get(frame_path.name, [])
        if len(add_events) > 0:
            next_manual_tracklet_id = apply_add_events(
                sam_generator=sam_generator,
                cutie_propagator=cutie_propagator,
                add_events=add_events,
                previous_frame=previous_frame,
                previous_frame_path=previous_frame_path,
                frame_path=frame_path,
                next_manual_tracklet_id=next_manual_tracklet_id,
            )

        propagate_and_save_frame(
            cutie_propagator=cutie_propagator,
            frame=frame,
            frame_path=frame_path,
            remove_events_by_file=remove_events_by_file,
            output_path=output_dir / frame_path.name,
        )

        previous_frame = frame
        previous_frame_path = frame_path

    print(f"Saved visualizations to {output_dir.resolve()}")
    return 0
