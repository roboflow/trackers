# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Lightweight smoke tests for ByteTrackTracker and the shared get_iou_matrix utility.

These tests run without any external data downloads and without the integration
test marker, so they execute on every guard invocation during the campaign loop.

Goals:
- Catch bugs in ByteTrackTracker.update() before the expensive ~7s metric eval runs.
- Guard get_iou_matrix against cross-tracker regressions: both ByteTrack and SORT
  depend on this shared utility, so a ByteTrack-motivated change that breaks it
  would silently regress SORT with no other test catching it.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from trackers import ByteTrackTracker
from trackers.core.bytetrack.kalman import ByteTrackKalmanBoxTracker
from trackers.core.sort.kalman import SORTKalmanBoxTracker
from trackers.core.sort.utils import get_iou_matrix


def _make_dets(boxes: np.ndarray, confidence: float = 0.9) -> sv.Detections:
    return sv.Detections(
        xyxy=boxes.astype(np.float32),
        confidence=np.full(len(boxes), confidence, dtype=np.float32),
    )


class TestByteTrackTrackerUpdate:
    """End-to-end update() smoke tests — exercises the full predict/associate cycle."""

    def test_stable_detections_produce_confirmed_track_ids(self) -> None:
        """Objects detected consistently across frames should receive confirmed IDs."""
        tracker = ByteTrackTracker(minimum_consecutive_frames=2)
        boxes = np.array([[10.0, 20.0, 50.0, 60.0], [100.0, 100.0, 150.0, 150.0]])

        result = None
        for _ in range(5):
            result = tracker.update(_make_dets(boxes))

        assert result is not None
        assert result.tracker_id is not None, (
            "tracker_id must be set after stable detections"
        )
        assert any(tid >= 0 for tid in result.tracker_id), (
            "at least one track should be confirmed (tid >= 0)"
        )

    def test_update_returns_sv_detections(self) -> None:
        """update() return type must be sv.Detections regardless of input."""
        tracker = ByteTrackTracker()
        boxes = np.array([[0.0, 0.0, 100.0, 100.0]])
        result = tracker.update(_make_dets(boxes))
        assert isinstance(result, sv.Detections)

    def test_empty_detections_do_not_crash(self) -> None:
        """update() with empty Detections must not raise, must return sv.Detections."""
        tracker = ByteTrackTracker()
        for _ in range(3):
            result = tracker.update(sv.Detections.empty())
        assert isinstance(result, sv.Detections)

    def test_reset_clears_tracks(self) -> None:
        """reset() must discard all active tracks; next update starts fresh."""
        tracker = ByteTrackTracker(minimum_consecutive_frames=1)
        boxes = np.array([[10.0, 20.0, 50.0, 60.0]])

        for _ in range(3):
            tracker.update(_make_dets(boxes))

        tracker.reset()
        result = tracker.update(_make_dets(boxes))
        # After reset + one frame, no track should be confirmed yet
        # (minimum_consecutive_frames=1 means confirmed immediately after 1 hit)
        assert result.tracker_id is not None

    @pytest.mark.parametrize("n_boxes", [1, 5, 20])
    def test_various_detection_counts(self, n_boxes: int) -> None:
        """update() must handle 1, 5, or 20 detections without raising."""
        tracker = ByteTrackTracker()
        rng = np.random.default_rng(seed=0)
        x1 = rng.uniform(0, 400, size=(n_boxes,))
        y1 = rng.uniform(0, 300, size=(n_boxes,))
        boxes = np.stack([x1, y1, x1 + 40, y1 + 40], axis=1)
        result = tracker.update(_make_dets(boxes))
        assert isinstance(result, sv.Detections)


class TestGetIouMatrix:
    """Unit tests for the shared get_iou_matrix utility.

    Both ByteTrackTracker and SORTTracker depend on this function. Any
    modification to it for ByteTrack purposes is caught here for SORT too.
    """

    def test_bytetrack_kalman_same_box_iou_is_one(self) -> None:
        """Predicted box == detection box should yield IoU ≈ 1.0."""
        box = np.array([10.0, 20.0, 50.0, 60.0])
        tracker_obj = ByteTrackKalmanBoxTracker(bbox=box)
        detection_boxes = np.array([[10.0, 20.0, 50.0, 60.0]])

        iou_mat = get_iou_matrix([tracker_obj], detection_boxes)

        assert iou_mat.shape == (1, 1)
        assert iou_mat[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_sort_kalman_same_box_iou_is_one(self) -> None:
        """SORT Kalman: predicted box == detection box should yield IoU ≈ 1.0."""
        box = np.array([10.0, 20.0, 50.0, 60.0])
        tracker_obj = SORTKalmanBoxTracker(bbox=box)
        detection_boxes = np.array([[10.0, 20.0, 50.0, 60.0]])

        iou_mat = get_iou_matrix([tracker_obj], detection_boxes)

        assert iou_mat.shape == (1, 1)
        assert iou_mat[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_non_overlapping_boxes_yield_non_positive_similarity(self) -> None:
        """Non-overlapping boxes must yield similarity ≤ 0 (DIoU penalises centre distance)."""  # noqa: E501
        box = np.array([0.0, 0.0, 10.0, 10.0])
        tracker_obj = ByteTrackKalmanBoxTracker(bbox=box)
        detection_boxes = np.array([[100.0, 100.0, 200.0, 200.0]])

        iou_mat = get_iou_matrix([tracker_obj], detection_boxes)

        assert iou_mat[0, 0] <= 0.0

    def test_empty_trackers_returns_zero_matrix(self) -> None:
        """Empty tracker list must return a (0, N) zero matrix."""
        detection_boxes = np.array([[10.0, 20.0, 50.0, 60.0]])
        iou_mat = get_iou_matrix([], detection_boxes)
        assert iou_mat.shape == (0, 1)

    def test_empty_detections_returns_zero_matrix(self) -> None:
        """Empty detection array must return a (N, 0) zero matrix."""
        box = np.array([10.0, 20.0, 50.0, 60.0])
        tracker_obj = ByteTrackKalmanBoxTracker(bbox=box)
        iou_mat = get_iou_matrix([tracker_obj], np.zeros((0, 4), dtype=np.float32))
        assert iou_mat.shape == (1, 0)

    def test_output_shape_multiple_trackers_and_detections(self) -> None:
        """Shape must be (n_trackers, n_detections) for arbitrary sizes."""
        boxes = np.array(
            [
                [0.0, 0.0, 40.0, 40.0],
                [50.0, 50.0, 90.0, 90.0],
                [200.0, 200.0, 300.0, 300.0],
            ]
        )
        trackers = [ByteTrackKalmanBoxTracker(bbox=b) for b in boxes]
        detection_boxes = np.array([[0.0, 0.0, 40.0, 40.0], [50.0, 50.0, 90.0, 90.0]])

        iou_mat = get_iou_matrix(trackers, detection_boxes)

        assert iou_mat.shape == (3, 2)
