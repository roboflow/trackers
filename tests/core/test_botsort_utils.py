# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for BoT-SORT utility helpers."""

from __future__ import annotations

import numpy as np

from trackers.core.botsort.tracklet import BoTSORTTracklet
from trackers.core.botsort.utils import _fuse_score, get_alive_tracklets

# -------------------------------------------------------------------
# _fuse_score
# -------------------------------------------------------------------


class TestFuseScore:
    """Tests for _fuse_score — IoU similarity fused with detection scores."""

    def test_fuse_score_empty_matrix_returns_empty(self) -> None:
        """Empty similarity matrix must return empty array unchanged."""
        iou = np.array([], dtype=np.float32).reshape(0, 0)
        scores = np.array([], dtype=np.float32)
        result = _fuse_score(iou, scores)
        assert result.shape == (0, 0)

    def test_fuse_score_identity(self) -> None:
        """Fusing with all-1 scores must leave IoU unchanged."""
        iou = np.array([[0.5, 0.8], [0.3, 0.9]], dtype=np.float32)
        scores = np.array([1.0, 1.0], dtype=np.float32)
        result = _fuse_score(iou, scores)
        np.testing.assert_array_almost_equal(result, iou)

    def test_fuse_score_halves_with_half_confidence(self) -> None:
        """Fusing with 0.5 scores must halve every IoU value."""
        iou = np.array([[0.6, 0.4], [0.2, 1.0]], dtype=np.float32)
        scores = np.array([0.5, 0.5], dtype=np.float32)
        result = _fuse_score(iou, scores)
        expected = iou * 0.5
        np.testing.assert_array_almost_equal(result, expected)

    def test_fuse_score_per_detection_weighting(self) -> None:
        """Each detection column must be scaled by its own score."""
        iou = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        result = _fuse_score(iou, scores)
        expected = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)


# -------------------------------------------------------------------
# _get_alive_tracklets
# -------------------------------------------------------------------


def _make_tracklet(success_updates: int = 1, time_since: int = 0) -> BoTSORTTracklet:
    """Build a minimal BoTSORTTracklet with controlled state."""
    tracklet = BoTSORTTracklet(
        initial_bbox=np.array([10.0, 10.0, 20.0, 20.0]),
    )
    tracklet.number_of_successful_updates = success_updates
    tracklet.time_since_update = time_since
    return tracklet


class TestGetAliveTrackers:
    """Tests for _get_alive_tracklets — tracklet lifecycle filtering."""

    def test_empty_list_returns_empty(self) -> None:
        """Empty input must return empty output."""
        assert get_alive_tracklets([], 3, 30) == []

    def test_active_immature_track_survives(self) -> None:
        """A track updated this frame survives even with 1 successful update."""
        track = _make_tracklet(success_updates=1, time_since=0)
        alive = get_alive_tracklets([track], minimum_consecutive_frames=3, maximum_frames_without_update=30)
        assert alive == [track]

    def test_mature_lost_track_survives_within_buffer(self) -> None:
        """A mature track lost for a few frames survives inside the buffer."""
        track = _make_tracklet(success_updates=3, time_since=5)
        alive = get_alive_tracklets([track], minimum_consecutive_frames=3, maximum_frames_without_update=30)
        assert alive == [track]

    def test_immature_lost_track_dies(self) -> None:
        """An immature track that has been lost must be removed."""
        track = _make_tracklet(success_updates=1, time_since=5)
        alive = get_alive_tracklets([track], minimum_consecutive_frames=3, maximum_frames_without_update=30)
        assert alive == []

    def test_mature_track_dies_past_buffer(self) -> None:
        """Even a mature track must die once it exceeds the lost buffer."""
        track = _make_tracklet(success_updates=5, time_since=31)
        alive = get_alive_tracklets([track], minimum_consecutive_frames=3, maximum_frames_without_update=30)
        assert alive == []

    def test_buffer_boundary_exactly_at_limit(self) -> None:
        """time_since_update == maximum_frames_without_update is past the limit."""
        track = _make_tracklet(success_updates=3, time_since=30)
        alive = get_alive_tracklets([track], minimum_consecutive_frames=3, maximum_frames_without_update=30)
        assert alive == []

    def test_buffer_boundary_one_under_limit(self) -> None:
        """time_since_update == maximum_frames_without_update - 1 survives."""
        track = _make_tracklet(success_updates=3, time_since=29)
        alive = get_alive_tracklets([track], minimum_consecutive_frames=3, maximum_frames_without_update=30)
        assert alive == [track]

    def test_mixed_tracks_partial_survival(self) -> None:
        """Mixed maturity and lost-time states — only valid tracks survive."""
        active_mature = _make_tracklet(success_updates=3, time_since=0)
        active_immature = _make_tracklet(success_updates=1, time_since=0)
        lost_mature = _make_tracklet(success_updates=3, time_since=10)
        lost_immature = _make_tracklet(success_updates=1, time_since=10)
        dead_mature = _make_tracklet(success_updates=3, time_since=35)

        tracks = [
            active_mature,
            active_immature,
            lost_mature,
            lost_immature,
            dead_mature,
        ]
        alive = get_alive_tracklets(tracks, minimum_consecutive_frames=3, maximum_frames_without_update=30)
        assert alive == [active_mature, active_immature, lost_mature]
