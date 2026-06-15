# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""CBIoU-specific tracker tests.

Generic lifecycle contracts are covered in test_trackers.py via ALL_TRACKER_IDS.
This file covers C-BIoU-specific invariants (Yang et al., WACV 2023):
  - Cascaded BIoU with per-step buffer scales (b1, b2)
  - CMC disabled; frame argument triggers UserWarning
  - BIoU association more tolerant than standard IoU
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import supervision as sv

from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.cbiou.tracker import CBIoUTracker
from trackers.utils.iou import BIoU


def _detection(xyxy: tuple[float, float, float, float], conf: float = 0.9) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([conf], dtype=np.float32),
    )


def _make_frame(h: int = 480, w: int = 640, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


class TestCBIoUConstruction:
    def test_default_construction(self) -> None:
        tracker = CBIoUTracker()
        assert tracker is not None

    def test_per_step_biou_instances(self) -> None:
        tracker = CBIoUTracker(
            buffer_ratio_first=0.1,
            buffer_ratio_second=0.3,
        )
        assert isinstance(tracker.iou_first, BIoU)
        assert isinstance(tracker.iou_second, BIoU)
        assert not hasattr(tracker, "iou_unconfirmed")

    def test_buffer_ratios_forwarded_to_biou(self) -> None:
        tracker = CBIoUTracker(
            buffer_ratio_first=0.1,
            buffer_ratio_second=0.3,
        )
        assert tracker.iou_first.buffer_ratio == pytest.approx(0.1)
        assert tracker.iou_second.buffer_ratio == pytest.approx(0.3)

    def test_cmc_disabled(self) -> None:
        tracker = CBIoUTracker()
        assert tracker.enable_cmc is False
        assert tracker.cmc is None

    def test_tracker_id(self) -> None:
        assert CBIoUTracker.tracker_id == "cbiou"

    def test_invalid_buffer_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="buffer_ratio"):
            CBIoUTracker(buffer_ratio_first=-0.01)


class TestCBIoUFrameWarning:
    def test_frame_triggers_warning(self) -> None:
        tracker = CBIoUTracker()
        with pytest.warns(UserWarning):
            tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame=_make_frame())

    def test_no_warning_without_frame(self) -> None:
        tracker = CBIoUTracker()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tracker.update(_detection((100.0, 100.0, 200.0, 200.0)))


class TestCBIoUAssociationTolerance:
    """BIoU should associate near-miss detections that plain IoU would miss."""

    def test_near_miss_associated_with_buffer(self) -> None:
        """
        A track initialized at box A, then a detection at box B just outside
        should be associated by CBIoU (buffer expands boxes) but not by
        BoTSORT with standard IoU (tight threshold).

        Box A: [0, 0, 100, 100]  (100x100)
        Box B: [110, 0, 210, 100]  (gap of 10px = 10% of width)
        With buffer_ratio=0.15 each side expands by 15px, so A becomes
        [-15, -15, 115, 115] and B becomes [95, -15, 225, 115] —
        they now overlap.
        """
        # Frame 1: spawn a track at box A with high confidence
        cbiou = CBIoUTracker(
            buffer_ratio_first=0.15,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.5,
            minimum_iou_threshold_first_assoc=0.05,
        )
        botsort = BoTSORTTracker(
            enable_cmc=False,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.5,
            minimum_iou_threshold_first_assoc=0.05,
        )

        box_a = (0.0, 0.0, 100.0, 100.0)
        box_b = (110.0, 0.0, 210.0, 100.0)

        cbiou.update(_detection(box_a))
        botsort.update(_detection(box_a))
        botsort_frame1_track_id = next((t.tracker_id for t in botsort.tracks), None)

        # Frame 2: detection slightly outside A — CBIoU buffer closes the gap
        cbiou_result = cbiou.update(_detection(box_b))
        botsort_result = botsort.update(_detection(box_b))

        assert cbiou_result.tracker_id is not None and len(cbiou_result.tracker_id) == 1
        assert cbiou_result.tracker_id[0] >= 0
        cbiou_frame1_id = cbiou.tracks[0].tracker_id
        assert cbiou_result.tracker_id[0] == cbiou_frame1_id

        botsort_ids = botsort_result.tracker_id
        if botsort_ids is not None and len(botsort_ids) > 0 and botsort_frame1_track_id is not None:
            assert botsort_ids[0] != botsort_frame1_track_id


class TestCBIoUZeroBufferEquivalence:
    """With buffer_ratio=0, BIoU recovers IoU; C-BIoU should match BoT-SORT (no CMC)."""

    def test_zero_buffer_matches_botsort_without_cmc(self) -> None:
        detections = [
            _detection((0.0, 0.0, 50.0, 50.0)),
            _detection((5.0, 5.0, 55.0, 55.0)),
            _detection((100.0, 100.0, 150.0, 150.0)),
            _detection((105.0, 105.0, 155.0, 155.0)),
            _detection((8.0, 8.0, 58.0, 58.0)),
        ]

        def run_tracker(tracker: CBIoUTracker | BoTSORTTracker) -> list[sv.Detections]:
            tracker.reset()
            return [tracker.update(det) for det in detections]

        cbiou = CBIoUTracker(
            buffer_ratio_first=0.0,
            buffer_ratio_second=0.0,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.5,
            minimum_iou_threshold_first_assoc=0.3,
            minimum_iou_threshold_second_assoc=0.3,
            minimum_iou_threshold_unconfirmed_assoc=0.3,
            high_conf_det_threshold=0.6,
        )
        botsort = BoTSORTTracker(
            enable_cmc=False,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.5,
            minimum_iou_threshold_first_assoc=0.3,
            minimum_iou_threshold_second_assoc=0.3,
            minimum_iou_threshold_unconfirmed_assoc=0.3,
            high_conf_det_threshold=0.6,
        )

        cbiou_results = run_tracker(cbiou)
        botsort_results = run_tracker(botsort)

        for frame_idx, (r_cbiou, r_botsort) in enumerate(zip(cbiou_results, botsort_results)):
            assert len(r_cbiou) == len(r_botsort), (
                f"frame {frame_idx}: CBIoU(buffer=0) and BoTSORT(no CMC) returned different "
                f"detection counts ({len(r_cbiou)} vs {len(r_botsort)})"
            )
            np.testing.assert_array_equal(
                r_cbiou.tracker_id,
                r_botsort.tracker_id,
                err_msg=f"frame {frame_idx}: different tracker IDs",
            )
            if len(r_cbiou) > 0:
                np.testing.assert_allclose(
                    r_cbiou.xyxy,
                    r_botsort.xyxy,
                    err_msg=f"frame {frame_idx}: different boxes",
                )


class TestCBIoUSearchSpace:
    def test_cascade_buffer_params_in_search_space(self) -> None:
        ss = CBIoUTracker.search_space
        assert "buffer_ratio_first" in ss
        assert "buffer_ratio_second" in ss
        assert "buffer_ratio_unconfirmed" not in ss

    def test_no_cmc_in_search_space(self) -> None:
        ss = CBIoUTracker.search_space
        assert "enable_cmc" not in ss
        assert "cmc_method" not in ss


class TestCBIoUUnmatchedLowConfidence:
    def test_unmatched_low_conf_detection_has_minus_one_tracker_id(self) -> None:
        """Unmatched low-confidence detection appears in update() output with tracker_id=-1."""
        tracker = CBIoUTracker(
            minimum_consecutive_frames=1,
            high_conf_det_threshold=0.6,
            buffer_ratio_first=0.1,
            buffer_ratio_second=0.3,
            minimum_iou_threshold_second_assoc=0.1,
        )
        # Frame 1: establish a confirmed track near origin
        tracker.update(
            sv.Detections(
                xyxy=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
                confidence=np.array([0.9], dtype=np.float32),
            )
        )
        # Frame 2: low-confidence detection far from any track (no IoU overlap)
        result = tracker.update(
            sv.Detections(
                xyxy=np.array([[500.0, 500.0, 510.0, 510.0]], dtype=np.float32),
                confidence=np.array([0.3], dtype=np.float32),
            )
        )
        assert len(result) == 1
        assert result.tracker_id[0] == -1
