# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from trackers.core.bytetrack.tracklet import ByteTrackTracklet
from trackers.core.sort.tracker import SORTTracker
from trackers.core.sort.tracklet import SORTTracklet
from trackers.utils.state_representations import (
    XCYCSRStateEstimator,
    XYXYStateEstimator,
)


@pytest.fixture
def bbox() -> np.ndarray:
    return np.array([10.0, 20.0, 30.0, 40.0])


@pytest.fixture
def detections(bbox: np.ndarray) -> sv.Detections:
    detections = sv.Detections(xyxy=bbox.reshape(1, 4))
    detections.confidence = np.array([0.9])
    return detections


def test_sort_tracklet_update_none_increments_time_without_changing_bbox(
    bbox: np.ndarray,
) -> None:
    tracklet = SORTTracklet(bbox)
    initial_bbox = tracklet.get_state_bbox().copy()

    tracklet.update(None)

    assert tracklet.time_since_update == 1
    np.testing.assert_allclose(tracklet.get_state_bbox(), initial_bbox)


def test_sort_tracklet_configures_different_noise_for_state_estimators(
    bbox: np.ndarray,
) -> None:
    xcycsr_tracklet = SORTTracklet(bbox, state_estimator_class=XCYCSRStateEstimator)
    xyxy_tracklet = SORTTracklet(bbox, state_estimator_class=XYXYStateEstimator)

    xcycsr_kf = xcycsr_tracklet.state_estimator.kf
    xyxy_kf = xyxy_tracklet.state_estimator.kf

    noise_differs = (
        not np.array_equal(xcycsr_kf.R, xyxy_kf.R)
        or xcycsr_kf.Q.shape != xyxy_kf.Q.shape
        or xcycsr_kf.P.shape != xyxy_kf.P.shape
    )
    assert noise_differs


def test_bytetrack_tracklet_starts_with_one_successful_consecutive_update(
    bbox: np.ndarray,
) -> None:
    tracklet = ByteTrackTracklet(bbox)

    assert tracklet.number_of_successful_consecutive_updates == 1


def test_tracklet_id_counter_increments_per_subclass() -> None:
    sort_count_id = SORTTracklet.count_id
    bytetrack_count_id = ByteTrackTracklet.count_id
    try:
        SORTTracklet.count_id = 0
        ByteTrackTracklet.count_id = 0

        assert SORTTracklet.get_next_tracker_id() == 0
        assert SORTTracklet.get_next_tracker_id() == 1
        assert ByteTrackTracklet.get_next_tracker_id() == 0
        assert ByteTrackTracklet.get_next_tracker_id() == 1
        assert SORTTracklet.get_next_tracker_id() == 2
    finally:
        SORTTracklet.count_id = sort_count_id
        ByteTrackTracklet.count_id = bytetrack_count_id


def test_sort_tracker_trackers_alias_returns_tracks_with_warning(
    detections: sv.Detections,
) -> None:
    tracker = SORTTracker(minimum_consecutive_frames=1)
    tracker.update(detections)

    with pytest.warns((DeprecationWarning, FutureWarning)):
        trackers = tracker.trackers

    assert trackers is tracker.tracks
