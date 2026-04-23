# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ``tracked_objects`` property (issue #105).

Verifies that all three concrete trackers expose alive-but-unmatched tracks
(e.g. occluded) via Kalman-predicted boxes, keyed by a stable ``tracker_id``,
and that those tracks drop out once ``lost_track_buffer`` is exceeded.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from trackers.core.base import BaseTracker


def _one_detection(xyxy: tuple[float, float, float, float]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([0.95], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )


def _instantiate(tracker_id: str) -> BaseTracker:
    import trackers

    _ = trackers  # triggers tracker auto-registration
    info = BaseTracker._lookup_tracker(tracker_id)
    assert info is not None, f"tracker {tracker_id} not registered"
    return info.tracker_class()


_TRACKER_IDS = ["sort", "bytetrack", "ocsort"]


@pytest.mark.parametrize("tracker_id", _TRACKER_IDS)
def test_tracked_objects_exposes_mature_track(tracker_id: str) -> None:
    """After enough consistent frames the track is mature and visible."""
    tracker = _instantiate(tracker_id)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_one_detection(bbox))

    exposed = tracker.tracked_objects
    assert len(exposed) == 1
    assert exposed.tracker_id[0] != -1
    pred = exposed.xyxy[0]
    assert np.allclose(pred, np.array(bbox), atol=10.0), (
        f"predicted box {pred} drifted far from input {bbox}"
    )


@pytest.mark.parametrize("tracker_id", _TRACKER_IDS)
def test_tracked_objects_survives_occlusion(tracker_id: str) -> None:
    """A mature track stays exposed during a short detection gap."""
    tracker = _instantiate(tracker_id)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_one_detection(bbox))

    baseline = tracker.tracked_objects
    assert len(baseline) == 1
    original_id = int(baseline.tracker_id[0])

    for _ in range(3):
        tracker.update(sv.Detections.empty())
        occluded = tracker.tracked_objects
        assert len(occluded) == 1, "track should remain alive through short occlusion"
        assert int(occluded.tracker_id[0]) == original_id


@pytest.mark.parametrize("tracker_id", _TRACKER_IDS)
def test_tracked_objects_drops_after_expiry(tracker_id: str) -> None:
    """After ``lost_track_buffer`` empty frames, the track is pruned."""
    tracker = _instantiate(tracker_id)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_one_detection(bbox))
    assert len(tracker.tracked_objects) == 1

    buffer = tracker.maximum_frames_without_update  # type: ignore[attr-defined]
    for _ in range(buffer + 5):
        tracker.update(sv.Detections.empty())

    expired = tracker.tracked_objects
    assert len(expired) == 0
    assert expired.tracker_id.size == 0


@pytest.mark.parametrize("tracker_id", _TRACKER_IDS)
def test_tracked_objects_empty_before_update(tracker_id: str) -> None:
    """Before the first update, no tracked objects are exposed."""
    tracker = _instantiate(tracker_id)

    assert len(tracker.tracked_objects) == 0
    assert tracker.tracked_objects.tracker_id.size == 0
