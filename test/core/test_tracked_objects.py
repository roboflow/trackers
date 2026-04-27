# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ``tracked_objects`` property (issue #105).

Verifies that all three concrete trackers expose alive tracks via the
``tracked_objects`` property as ``sv.Detections`` with Kalman-predicted
boxes and stable tracker IDs, including the critical occlusion-survival
scenario that motivates the feature.
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


def _instantiate(tracker_id: str, **kwargs: object) -> BaseTracker:
    import trackers

    _ = trackers  # triggers tracker auto-registration
    info = BaseTracker._lookup_tracker(tracker_id)
    assert info is not None, f"tracker {tracker_id} not registered"
    return info.tracker_class(**kwargs)


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
    assert exposed.confidence is None
    assert exposed.class_id is None
    assert exposed.xyxy.dtype == np.float32
    assert exposed.tracker_id.dtype.kind == "i"


@pytest.mark.parametrize("tracker_id", _TRACKER_IDS)
def test_tracked_objects_empty_before_update(tracker_id: str) -> None:
    """Before the first update, no tracked objects are exposed."""
    tracker = _instantiate(tracker_id)

    result = tracker.tracked_objects
    assert len(result) == 0
    assert result.tracker_id is not None
    assert result.tracker_id.size == 0
    assert result.tracker_id.dtype.kind == "i"
    assert result.confidence is None
    assert result.class_id is None


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


@pytest.mark.parametrize("tracker_id", _TRACKER_IDS)
def test_tracked_objects_survives_occlusion(tracker_id: str) -> None:
    """Track stays in tracked_objects while absent from update() during occlusion."""
    tracker = _instantiate(tracker_id, lost_track_buffer=5, frame_rate=30)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_one_detection(bbox))

    # One missed frame — update() sees nothing, tracked_objects still holds it.
    update_result = tracker.update(sv.Detections.empty())
    assert len(update_result) == 0, "update() must not return occluded track"

    exposed = tracker.tracked_objects
    assert len(exposed) == 1, "tracked_objects must keep alive-but-occluded track"
    assert exposed.tracker_id[0] != -1
    assert np.all(np.isfinite(exposed.xyxy)), "Kalman prediction must be finite"


@pytest.mark.parametrize("tracker_id", _TRACKER_IDS)
def test_tracked_objects_expires_after_buffer(tracker_id: str) -> None:
    """Track is removed from tracked_objects once lost_track_buffer is exceeded."""
    tracker = _instantiate(tracker_id, lost_track_buffer=3, frame_rate=30)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_one_detection(bbox))

    # Feed max_frames + 1 empty updates — forces expiry for all tracker variants.
    max_f = tracker.maximum_frames_without_update
    for _ in range(max_f + 1):
        tracker.update(sv.Detections.empty())

    assert len(tracker.tracked_objects) == 0, "track must expire after buffer exceeded"
