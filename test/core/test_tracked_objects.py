# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ``tracked_objects`` property (issue #105).

Verifies that all three concrete trackers expose alive tracks via the
``tracked_objects`` property as ``sv.Detections`` with Kalman-predicted
boxes and stable tracker IDs.

The pruning-and-occlusion behaviour itself (track expires after the lost
buffer, track survives a short gap, ``time_since_update`` advances on a
miss) is exercised at the tracker level in ``test_tracker_pruning.py``;
this file focuses on what ``tracked_objects`` exposes to callers.
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
def test_tracked_objects_empty_before_update(tracker_id: str) -> None:
    """Before the first update, no tracked objects are exposed."""
    tracker = _instantiate(tracker_id)

    assert len(tracker.tracked_objects) == 0
    assert tracker.tracked_objects.tracker_id.size == 0


@pytest.mark.parametrize("tracker_id", _TRACKER_IDS)
def test_tracked_objects_multiple_simultaneous_tracks(tracker_id: str) -> None:
    """Two mature, simultaneous tracks are both exposed with valid IDs."""
    tracker = _instantiate(tracker_id)

    detections = sv.Detections(
        xyxy=np.array(
            [
                [10.0, 10.0, 50.0, 50.0],
                [200.0, 200.0, 300.0, 300.0],
            ],
            dtype=np.float32,
        ),
        confidence=np.array([0.95, 0.95], dtype=np.float32),
        class_id=np.array([0, 0], dtype=int),
    )

    for _ in range(6):
        tracker.update(detections)

    exposed = tracker.tracked_objects
    assert len(exposed) == 2
    assert exposed.xyxy.shape == (2, 4)

    tracker_ids = exposed.tracker_id
    assert tracker_ids.shape == (2,)
    assert np.all(tracker_ids >= 0)
    assert len(set(map(int, tracker_ids))) == 2
