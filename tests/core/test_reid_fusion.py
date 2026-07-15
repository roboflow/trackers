# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for ReID fusion helpers."""

from __future__ import annotations

import numpy as np
import pytest

from trackers.core.botsort.fusion import fuse_botsort_reid_association
from trackers.core.botsort.tracker import BoTSORTTracker


class TestFuseBotsortReidAssociation:
    def test_appearance_can_win_when_proximity_passes(self) -> None:
        iou_raw = np.array([[0.7]], dtype=np.float32)
        iou_fused = np.array([[0.63]], dtype=np.float32)
        app_sim = np.array([[0.8]], dtype=np.float32)
        proximity_iou = np.array([[0.7]], dtype=np.float32)
        fused = fuse_botsort_reid_association(
            iou_raw,
            iou_fused,
            app_sim,
            proximity_iou_similarity=proximity_iou,
            proximity_threshold=0.5,
            appearance_threshold=0.25,
        )
        assert fused[0, 0] == pytest.approx(0.9)

    def test_low_proximity_zeros_appearance(self) -> None:
        iou_raw = np.array([[0.4]], dtype=np.float32)
        iou_fused = np.array([[0.36]], dtype=np.float32)
        app_sim = np.array([[0.9]], dtype=np.float32)
        proximity_iou = np.array([[0.4]], dtype=np.float32)
        fused = fuse_botsort_reid_association(
            iou_raw,
            iou_fused,
            app_sim,
            proximity_iou_similarity=proximity_iou,
            proximity_threshold=0.5,
            appearance_threshold=0.25,
        )
        assert fused[0, 0] == pytest.approx(0.36)

    def test_proximity_uses_standard_iou_not_giou(self) -> None:
        """GIoU can be high while standard IoU fails the proximity gate."""
        giou_raw = np.array([[0.85]], dtype=np.float32)
        giou_fused = np.array([[0.80]], dtype=np.float32)
        app_sim = np.array([[0.95]], dtype=np.float32)
        standard_iou = np.array([[0.35]], dtype=np.float32)
        fused = fuse_botsort_reid_association(
            giou_raw,
            giou_fused,
            app_sim,
            proximity_iou_similarity=standard_iou,
            proximity_threshold=0.5,
            appearance_threshold=0.25,
        )
        assert fused[0, 0] == pytest.approx(0.80)

    def test_no_appearance_information_keeps_fused_iou(self) -> None:
        iou_raw = np.array([[0.55, 0.2]], dtype=np.float32)
        iou_fused = np.array([[0.50, 0.18]], dtype=np.float32)
        app_sim = np.zeros((1, 2), dtype=np.float32)
        proximity_iou = np.array([[0.55, 0.2]], dtype=np.float32)
        fused = fuse_botsort_reid_association(
            iou_raw,
            iou_fused,
            app_sim,
            proximity_iou_similarity=proximity_iou,
            proximity_threshold=0.5,
            appearance_threshold=0.25,
        )
        np.testing.assert_allclose(fused, iou_fused)


class TestBoTSORTReidFusion:
    def test_tracker_delegates_to_reference_formula(self) -> None:
        tracker = BoTSORTTracker(enable_cmc=False)
        iou_raw = np.array([[0.7]], dtype=np.float32)
        iou_fused = np.array([[0.63]], dtype=np.float32)
        app_sim = np.array([[0.8]], dtype=np.float32)
        proximity_iou = np.array([[0.7]], dtype=np.float32)
        fused = tracker._fuse_botsort_reid(iou_raw, iou_fused, app_sim, proximity_iou)
        assert fused[0, 0] == pytest.approx(0.9)

    def test_reid_threshold_defaults(self) -> None:
        tracker = BoTSORTTracker(enable_cmc=False)
        assert tracker.appearance_threshold == 0.25
        assert tracker.proximity_threshold == 0.5
        assert tracker.reid_ema_alpha == 0.9

    def test_appearance_threshold_kwarg(self) -> None:
        tracker = BoTSORTTracker(enable_cmc=False, appearance_threshold=0.2)
        assert tracker.appearance_threshold == 0.2

    def test_proximity_threshold_kwarg(self) -> None:
        tracker = BoTSORTTracker(enable_cmc=False, proximity_threshold=0.4)
        assert tracker.proximity_threshold == 0.4
