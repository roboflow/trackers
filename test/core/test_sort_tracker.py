# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for SORTTracker.update input-mutation contract."""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from trackers.core.sort.tracker import SORTTracker


@pytest.mark.parametrize(
    "xyxy, confidence",
    [
        (
            np.array([[10.0, 20.0, 30.0, 40.0]]),
            np.array([0.9]),
        ),
        (
            np.array([[0.0, 0.0, 100.0, 100.0], [50.0, 50.0, 150.0, 150.0]]),
            np.array([0.8, 0.85]),
        ),
    ],
    ids=["single_detection", "two_detections"],
)
def test_sort_update_does_not_mutate_input(
    xyxy: np.ndarray, confidence: np.ndarray
) -> None:
    """update() must not modify the caller's sv.Detections; must return fresh object."""
    tracker = SORTTracker(minimum_consecutive_frames=1)
    dets = sv.Detections(xyxy=xyxy)
    dets.confidence = confidence
    assert dets.tracker_id is None

    result = tracker.update(dets)

    assert dets.tracker_id is None, "update() must not assign tracker_id on input"
    assert result is not dets, "update() must return a new sv.Detections instance"


def test_sort_update_empty_does_not_mutate_input() -> None:
    """update() on empty detections (no active trackers) must not mutate input."""
    tracker = SORTTracker(minimum_consecutive_frames=1)
    dets = sv.Detections(xyxy=np.zeros((0, 4), dtype=float))
    dets.confidence = np.array([], dtype=float)
    assert dets.tracker_id is None

    result = tracker.update(dets)

    assert dets.tracker_id is None, "update() must not assign tracker_id on empty input"
    assert result is not dets, "update() must return a new sv.Detections instance"
