# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for tracker pruning across detection gaps.

``time_since_update`` is incremented inside ``predict()``, which every
tracker calls at the top of its update loop for all live tracks.
Unmatched tracks therefore have their missed-frame clock advanced
automatically without any explicit call from the tracker.

These tests pin the contract for every concrete tracker:

1. A confirmed track is pruned after ``lost_track_buffer + N`` empty
   frames.
2. ``time_since_update`` actually advances when frames are missed.
3. A confirmed track survives a short occlusion (a few empty frames
   well below the lost-track buffer).
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from trackers.core.base import BaseTracker
from trackers.core.bytetrack.tracker import ByteTrackTracker

_TRACKER_IDS = ["sort", "bytetrack", "ocsort"]


def _detection(xyxy: tuple[float, float, float, float]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([0.95], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )


def _instantiate(tracker_id: str) -> BaseTracker:
    """Instantiate a tracker by id via the BaseTracker registry.

    The registry is populated by ``__init_subclass__`` hooks that fire
    when each concrete tracker module is imported. Importing the
    ``trackers`` package eagerly triggers those imports so the lookup
    below succeeds regardless of test collection order.
    """
    import trackers

    _ = trackers  # triggers tracker auto-registration
    info = BaseTracker._lookup_tracker(tracker_id)
    assert info is not None, f"tracker {tracker_id} not registered"
    return info.tracker_class()


@pytest.mark.parametrize("tracker_id", _TRACKER_IDS)
def test_track_expires_after_buffer(tracker_id: str) -> None:
    """A confirmed track is pruned after `lost_track_buffer + N` empty frames."""
    tracker = _instantiate(tracker_id)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_detection(bbox))
    assert len(tracker.tracks) == 1, "track should be alive after warmup"

    buffer = tracker.maximum_frames_without_update
    for _ in range(buffer + 5):
        tracker.update(sv.Detections.empty())

    assert len(tracker.tracks) == 0, (
        "track should be pruned after maximum_frames_without_update empty frames"
    )


@pytest.mark.parametrize("tracker_id", _TRACKER_IDS)
def test_time_since_update_advances_for_unmatched(tracker_id: str) -> None:
    """`predict()` advances `time_since_update` for unmatched tracks."""
    tracker = _instantiate(tracker_id)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_detection(bbox))

    for _ in range(5):
        tracker.update(sv.Detections.empty())

    assert len(tracker.tracks) == 1, "track is still within lost buffer"
    assert tracker.tracks[0].time_since_update == 5, (
        "time_since_update should reflect 5 missed frames, not 0"
    )


@pytest.mark.parametrize("tracker_id", _TRACKER_IDS)
def test_track_survives_short_occlusion(tracker_id: str) -> None:
    """A confirmed track stays alive across a short detection gap."""
    tracker = _instantiate(tracker_id)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_detection(bbox))
    confirmed_id = tracker.tracks[0].tracker_id
    assert confirmed_id != -1, "track should be confirmed after warmup"

    for _ in range(3):
        tracker.update(sv.Detections.empty())

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].tracker_id == confirmed_id, (
        "confirmed track must survive a short gap"
    )


def test_bytetrack_consecutive_counter_resets_on_miss() -> None:
    """ByteTrack resets the consecutive counter after a missed frame."""
    tracker = ByteTrackTracker(minimum_consecutive_frames=2, lost_track_buffer=30)
    bbox = (100.0, 100.0, 200.0, 200.0)

    tracker.update(_detection(bbox))
    assert tracker.tracks[0].number_of_successful_consecutive_updates == 1
    assert tracker.tracks[0].tracker_id == -1

    tracker.tracks[0].predict()

    tracker.update(_detection(bbox))

    assert tracker.tracks[0].number_of_successful_consecutive_updates == 1
    assert tracker.tracks[0].tracker_id == -1
