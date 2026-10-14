# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared parsing and scheduling for manual mask lifecycle inspection."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from trackers.cli.inspect._common import parse_xyxy_box


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

    The CLI frame is the frame where the box is valid. Internally, McByte-style timing applies the mask on that frame,
    then propagates to the next frame.
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

    Add events refer to the source frame where the box is valid and are internally shifted to the next frame. Therefore,
    they cannot be scheduled on the last selected frame. Remove events are applied before propagation to their target
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
