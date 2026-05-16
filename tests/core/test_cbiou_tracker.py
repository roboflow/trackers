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
        [-15, -15, 115, 115] and B becomes [93.5, -15, 226.5, 115] —
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

        # Frame 2: detection slightly outside A — CBIoU buffer closes the gap
        cbiou_result = cbiou.update(_detection(box_b))
        botsort_result = botsort.update(_detection(box_b))

        assert cbiou_result.tracker_id is not None and len(cbiou_result.tracker_id) == 1
        assert cbiou_result.tracker_id[0] >= 0
        cbiou_frame1_id = cbiou.tracks[0].tracker_id
        assert cbiou_result.tracker_id[0] == cbiou_frame1_id

        botsort_ids = botsort_result.tracker_id
        if botsort_ids is not None and len(botsort_ids) > 0:
            assert botsort_ids[0] != cbiou_frame1_id or botsort_ids[0] == -1


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
