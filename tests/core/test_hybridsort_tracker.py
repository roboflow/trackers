# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Hybrid-SORT-specific tracker tests.

Generic lifecycle contracts are covered in test_trackers.py via ALL_TRACKER_IDS.
This file covers what Hybrid-SORT adds on top of OC-SORT:
  - HMIoU is the default association metric and stays pluggable.
  - Tracklet Confidence Modeling (TCM): the confidence Kalman filter, the linear
    confidence trend, and the confidence-consistency penalty deciding matches
    that IoU alone gets wrong.
  - The low-confidence (BYTE) stage: it recovers existing tracks, never spawns
    new ones, and ignores detections at or below the 0.1 floor.
  - Robust OCM: per-corner velocities and the corner direction-consistency matrix.
  - ORU replays a gap only between two matched observations (reference semantics).
  - The output contract: every input detection is returned exactly once.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from trackers.core.hybridsort.tracker import HybridSORTTracker
from trackers.core.hybridsort.tracklet import HybridSORTTracklet
from trackers.core.hybridsort.utils import _build_corner_direction_consistency_matrix
from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.utils.iou import HMIoU, IoU


def _detections(boxes: list[list[float]], confidences: list[float]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float64).reshape(-1, 4),
        confidence=np.array(confidences, dtype=np.float64),
    )


def _id_of(result: sv.Detections, box: list[float]) -> int:
    """Return the tracker_id assigned to the output row whose box equals ``box``."""
    assert result.tracker_id is not None
    matches = np.where(np.all(np.isclose(result.xyxy, np.array(box)), axis=1))[0]
    assert len(matches) == 1, f"box {box} not found exactly once in output"
    return int(result.tracker_id[matches[0]])


_BOX = [100.0, 100.0, 200.0, 300.0]


class TestAssociationMetric:
    def test_default_iou_is_hmiou(self) -> None:
        assert isinstance(HybridSORTTracker().iou, HMIoU)

    def test_iou_is_pluggable(self) -> None:
        tracker = HybridSORTTracker(iou=IoU())
        assert type(tracker.iou) is IoU


class TestOutputContract:
    def test_every_input_detection_is_returned_once(self) -> None:
        """High, low and below-floor detections all come back, exactly one row each."""
        tracker = HybridSORTTracker()
        boxes = [_BOX, [400.0, 100.0, 500.0, 300.0], [700.0, 100.0, 800.0, 300.0]]
        for confidences in ([0.9, 0.4, 0.05], [0.9, 0.4, 0.05], [0.3, 0.9, 0.08]):
            result = tracker.update(_detections(boxes, confidences))
            assert len(result) == len(boxes)
            np.testing.assert_allclose(np.sort(result.xyxy, axis=0), np.sort(np.array(boxes), axis=0))

    def test_empty_update_returns_empty_result(self) -> None:
        result = HybridSORTTracker().update(sv.Detections.empty())
        assert len(result) == 0
        assert result.tracker_id is not None


class TestLowConfidenceStage:
    def test_low_confidence_detection_recovers_confirmed_track(self) -> None:
        """A confirmed track whose detection dips below the high threshold keeps its ID.

        OC-SORT has no low-confidence stage, so the same dip returns ``tracker_id=-1`` there.
        """
        hybrid = HybridSORTTracker()
        ocsort = OCSORTTracker()
        for frame in range(4):
            box = [_BOX[0] + 2 * frame, _BOX[1], _BOX[2] + 2 * frame, _BOX[3]]
            hybrid_id = _id_of(hybrid.update(_detections([box], [0.9])), box)
            ocsort.update(_detections([box], [0.9]))
        assert hybrid_id != -1

        dipped = [_BOX[0] + 8, _BOX[1], _BOX[2] + 8, _BOX[3]]
        assert _id_of(hybrid.update(_detections([dipped], [0.4])), dipped) == hybrid_id
        assert _id_of(ocsort.update(_detections([dipped], [0.4])), dipped) == -1

    def test_low_confidence_detections_never_spawn_tracks(self) -> None:
        tracker = HybridSORTTracker()
        for _ in range(5):
            result = tracker.update(_detections([_BOX], [0.45]))
            assert result.tracker_id is not None
            assert result.tracker_id.tolist() == [-1]
        assert len(tracker.tracks) == 0

    def test_detection_at_floor_is_never_associated(self) -> None:
        tracker = HybridSORTTracker(minimum_consecutive_frames=1)
        for _ in range(3):
            tracker.update(_detections([_BOX], [0.9]))

        result = tracker.update(_detections([_BOX], [0.1]))

        assert result.tracker_id is not None
        assert result.tracker_id.tolist() == [-1]
        assert tracker.tracks[0].time_since_update == 1


class TestTrackletConfidenceModeling:
    @staticmethod
    def _two_overlapping_tracks(tracker: HybridSORTTracker) -> tuple[int, int]:
        """Confirm two overlapping tracks: A (confidence 0.95) and B (confidence 0.65)."""
        box_a, box_b = _BOX, [104.0, 100.0, 204.0, 300.0]
        for _ in range(3):
            result = tracker.update(_detections([box_a, box_b], [0.95, 0.65]))
        return _id_of(result, box_a), _id_of(result, box_b)

    def test_confidence_consistency_overrides_small_iou_advantage(self) -> None:
        """The detection overlaps A slightly more, but its confidence matches B's history."""
        tracker = HybridSORTTracker()
        id_a, id_b = self._two_overlapping_tracks(tracker)
        detection = [101.0, 100.0, 201.0, 300.0]

        result = tracker.update(_detections([detection], [0.66]))

        assert _id_of(result, detection) == id_b != id_a

    def test_without_confidence_weight_iou_decides(self) -> None:
        tracker = HybridSORTTracker(confidence_weight_first_assoc=0.0)
        id_a, _ = self._two_overlapping_tracks(tracker)
        detection = [101.0, 100.0, 201.0, 300.0]

        result = tracker.update(_detections([detection], [0.66]))

        assert _id_of(result, detection) == id_a

    def test_linear_confidence_trend(self) -> None:
        tracklet = HybridSORTTracklet(np.array(_BOX), confidence=0.9)
        assert tracklet.linear_confidence == pytest.approx(0.9)  # one observation: no trend yet
        tracklet.predict()
        tracklet.update(np.array(_BOX), confidence=0.8)
        assert tracklet.linear_confidence == pytest.approx(0.7)

        tracklet.break_confidence_trend()

        assert tracklet.previous_confidence is None
        assert tracklet.linear_confidence == pytest.approx(0.8)

    def test_unmatched_track_breaks_its_confidence_trend(self) -> None:
        tracker = HybridSORTTracker(minimum_consecutive_frames=1)
        for confidence in (0.9, 0.8):
            tracker.update(_detections([_BOX], [confidence]))
        assert tracker.tracks[0].previous_confidence == pytest.approx(0.9)

        tracker.update(sv.Detections.empty())

        assert tracker.tracks[0].previous_confidence is None

    def test_kalman_confidence_converges_to_observations(self) -> None:
        tracklet = HybridSORTTracklet(np.array(_BOX), confidence=0.9)
        for _ in range(30):
            tracklet.predict()
            tracklet.update(np.array(_BOX), confidence=0.5)
        tracklet.predict()
        assert tracklet.kalman_confidence == pytest.approx(0.5, abs=0.02)


class TestObservationCentricReUpdate:
    def test_track_lost_before_first_match_is_not_frozen(self) -> None:
        """Like the reference implementation, a never-matched track has no gap to replay."""
        tracklet = HybridSORTTracklet(np.array(_BOX), confidence=0.9)
        tracklet.predict()
        tracklet.predict()
        assert tracklet._frozen_state is None
        assert tracklet._frozen_confidence_state is None

    def test_confidence_filter_is_frozen_and_replayed_with_the_box(self) -> None:
        tracklet = HybridSORTTracklet(np.array(_BOX), confidence=0.9)
        tracklet.predict()
        tracklet.update(np.array(_BOX), confidence=0.9)
        tracklet.predict()
        tracklet.predict()  # the previous frame was a miss: both filters freeze
        assert tracklet._frozen_state is not None
        assert tracklet._frozen_confidence_state is not None

        tracklet.update(np.array(_BOX), confidence=0.3)

        assert tracklet._frozen_state is None
        assert tracklet._frozen_confidence_state is None
        assert 0.3 < tracklet.kalman_confidence < 0.9


class TestRobustObservationCentricMomentum:
    def test_corner_velocities_sum_directions_over_delta_t(self) -> None:
        tracklet = HybridSORTTracklet(np.array(_BOX), confidence=0.9)
        for step in range(1, 4):
            tracklet.predict()
            shifted = np.array([_BOX[0] + 10 * step, _BOX[1], _BOX[2] + 10 * step, _BOX[3]])
            tracklet.update(shifted, confidence=0.9)
            if step == 1:
                assert tracklet.corner_velocities is None

        # Third match: unit directions from the two stored observations, per corner, as [dy, dx].
        assert tracklet.corner_velocities is not None
        np.testing.assert_allclose(tracklet.corner_velocities, np.tile([0.0, 2.0], (4, 1)), atol=1e-4)

    def test_direction_consistency_prefers_detection_along_motion(self) -> None:
        moving_right = np.tile([0.0, 1.0], (4, 1))[np.newaxis]
        reference = np.array([[0.0, 0.0, 10.0, 10.0]])
        detections = np.array([[5.0, 0.0, 15.0, 10.0], [-5.0, 0.0, 5.0, 10.0]])

        scores = _build_corner_direction_consistency_matrix(moving_right, reference, detections, np.ones((1, 1)))
        masked = _build_corner_direction_consistency_matrix(moving_right, reference, detections, np.zeros((1, 1)))

        # Exact alignment scores +/-0.5; the reference 1e-6 norm guard keeps arccos a hair off 0.
        np.testing.assert_allclose(scores, [[0.5, -0.5]], atol=1e-3)
        np.testing.assert_array_equal(masked, np.zeros((1, 2)))


class TestReviewRegressions:
    def test_confidence_filter_gap_noise_is_calibrated_from_its_own_q(self) -> None:
        """Off-nominal frame steps scale the configured confidence noise ``diag(1, 1e-4)``, not a placeholder."""
        tracklet = HybridSORTTracklet(np.array(_BOX), confidence=0.9)

        gap_noise = tracklet._confidence_motion.process_noise.build_Q(2.0, 30.0)

        # DWNA at frame_step 2 with sigma_a^2 = 1e-4: [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] * 1e-4.
        np.testing.assert_allclose(gap_noise, np.full((2, 2), 4e-4))

    def test_tracklets_do_not_alias_caller_detection_arrays(self) -> None:
        tracker = HybridSORTTracker()
        tracker.update(_detections([[100.0, 100.0, 150.0, 220.0]], [0.9]))
        detections = _detections([[102.0, 100.0, 152.0, 220.0]], [0.9])
        tracker.update(detections)

        detections.xyxy[:] = 0.0

        np.testing.assert_array_equal(tracker.tracks[0].last_observation, [102.0, 100.0, 152.0, 220.0])

    def test_miss_breaks_confidence_trend_even_on_duplicate_timestamp(self) -> None:
        """An unmatched frame drops the trend immediately, as in the reference, not at the next predict."""
        tracker = HybridSORTTracker(minimum_consecutive_frames=1)
        box = [100.0, 100.0, 150.0, 220.0]
        for step, confidence in enumerate((0.9, 0.9, 0.62)):
            tracker.update(_detections([box], [confidence]), timestamp=step / 30.0)
        tracker.update(sv.Detections.empty(), timestamp=3 / 30.0)  # miss
        assert tracker.tracks[0].previous_confidence is None

        # Same instant again (predict is skipped): the stale trend 2 * 0.62 - 0.9 = 0.34 must not be used.
        shifted = [125.0, 100.0, 175.0, 220.0]
        with pytest.warns(UserWarning, match="duplicate timestamp"):
            result = tracker.update(_detections([shifted], [0.30]), timestamp=3 / 30.0)

        assert _id_of(result, shifted) == -1
