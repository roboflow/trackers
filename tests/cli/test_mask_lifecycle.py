# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Characterize event parsing and frame timing shared by mask inspection commands."""

from pathlib import Path

import pytest

from trackers.cli.inspect._mask_lifecycle import (
    AddMaskEvent,
    RemoveMaskEvent,
    group_add_events_by_next_frame,
    group_remove_events,
    parse_add_mask_event,
    parse_remove_mask_event,
    validate_lifecycle_events,
)


def test_events_follow_selected_frames_and_preserve_multiple_entries() -> None:
    """Filename gaps do not change when additions and removals take effect."""
    frames = [Path(name) for name in ("000001.png", "000008.png", "000050.png")]
    additions = [
        parse_add_mask_event("000008.png:1,2,3,4"),
        parse_add_mask_event("000008.png:5.5,6,7,8"),
    ]
    removals = [parse_remove_mask_event("000050.png:300"), parse_remove_mask_event("000050.png:1")]
    validate_lifecycle_events(frames, additions, removals)
    assert additions[0] == AddMaskEvent("000008.png", (1.0, 2.0, 3.0, 4.0))
    assert additions[1].xyxy == (5.5, 6.0, 7.0, 8.0)
    assert removals[0] == RemoveMaskEvent("000050.png", 300)
    assert group_add_events_by_next_frame(frames, additions) == {"000050.png": additions}
    assert group_remove_events(removals) == {"000050.png": [300, 1]}


@pytest.mark.parametrize("event", ["a.png", "a.png:1,2,3", "a.png:1,2,3,4,5", "a.png:1,2,x,4", "a:b:1,2,3,4"])
def test_malformed_addition_names_the_expected_format(event: str) -> None:
    """Malformed box events produce the CLI's actionable format error."""
    with pytest.raises(ValueError, match="Add event must have format"):
        parse_add_mask_event(event)


@pytest.mark.parametrize("event", ["a.png", "a.png:x", "a.png:1.5", "a:b:1"])
def test_malformed_removal_names_the_expected_format(event: str) -> None:
    """Removal IDs must parse as integers, without silently rounding them."""
    with pytest.raises(ValueError, match="Remove event must have format"):
        parse_remove_mask_event(event)


@pytest.mark.parametrize("tracker_id", [0, -1])
def test_manual_removal_requires_positive_id(tracker_id: int) -> None:
    """Manual inspection IDs start at one, independently of backend ID support."""
    with pytest.raises(ValueError, match="positive integer"):
        parse_remove_mask_event(f"a.png:{tracker_id}")


@pytest.mark.parametrize(
    ("additions", "removals", "message"),
    [
        ([AddMaskEvent("missing.png", (1, 2, 3, 4))], [], "Add event frame is outside"),
        ([AddMaskEvent("b.png", (1, 2, 3, 4))], [], "last selected frame"),
        ([], [RemoveMaskEvent("missing.png", 1)], "Remove event frame is outside"),
        ([], [RemoveMaskEvent("a.png", 1)], "first selected frame"),
    ],
)
def test_events_outside_supported_timing_are_rejected(
    additions: list[AddMaskEvent], removals: list[RemoveMaskEvent], message: str
) -> None:
    """Reject unavailable frames and operations with no valid propagation step."""
    with pytest.raises(ValueError, match=message):
        validate_lifecycle_events([Path("a.png"), Path("b.png")], additions, removals)


def test_replay_without_lifecycle_events_is_valid() -> None:
    """Both commands can inspect propagation without any scheduled mutations."""
    frames = [Path("a.png"), Path("b.png")]
    validate_lifecycle_events(frames, [], [])
    assert group_add_events_by_next_frame(frames, []) == {}
    assert group_remove_events([]) == {}
