# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for ReID fusion helpers."""

from __future__ import annotations

import numpy as np
import pytest

from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.reid.fusion_methods import fuse_weighted_first_stage


class TestFuseWeightedFirstStage:
    def test_weight_zero_returns_iou_only(self) -> None:
        iou = np.array([[0.8, 0.2]], dtype=np.float32)
        app = np.array([[0.1, 0.9]], dtype=np.float32)
        fused = fuse_weighted_first_stage(iou, app, weight=0.0)
        np.testing.assert_allclose(fused, iou)

    def test_weight_one_returns_appearance_only(self) -> None:
        iou = np.array([[0.8, 0.2]], dtype=np.float32)
        app = np.array([[0.1, 0.9]], dtype=np.float32)
        fused = fuse_weighted_first_stage(iou, app, weight=1.0)
        np.testing.assert_allclose(fused, app)

    def test_weighted_blend(self) -> None:
        iou = np.array([[1.0, 0.0]], dtype=np.float32)
        app = np.array([[0.0, 1.0]], dtype=np.float32)
        fused = fuse_weighted_first_stage(iou, app, weight=0.2)
        np.testing.assert_allclose(fused, np.array([[0.8, 0.2]], dtype=np.float32))

    def test_invalid_weight_raises(self) -> None:
        iou = np.zeros((1, 1), dtype=np.float32)
        app = np.zeros((1, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="weight must be in"):
            fuse_weighted_first_stage(iou, app, weight=1.5)


class TestBoTSORTGatedMinFusion:
    def test_gated_pair_uses_scaled_appearance(self) -> None:
        tracker = BoTSORTTracker(enable_cmc=False)
        iou_sim = np.array([[0.7]], dtype=np.float32)
        app_sim = np.array([[0.8]], dtype=np.float32)
        fused = tracker._fuse_botsort_gated_min(iou_sim, app_sim)
        # d_iou=0.3, d_app=0.2 -> gated d_app=0.1 -> min=0.1 -> sim=0.9
        assert fused.shape == (1, 1)
        assert fused[0, 0] == pytest.approx(0.9)

    def test_ungated_pair_falls_back_to_iou(self) -> None:
        tracker = BoTSORTTracker(enable_cmc=False)
        iou_sim = np.array([[0.4]], dtype=np.float32)
        app_sim = np.array([[0.9]], dtype=np.float32)
        fused = tracker._fuse_botsort_gated_min(iou_sim, app_sim)
        # d_iou=0.6 fails IoU gate -> appearance ignored -> sim stays 0.4
        assert fused[0, 0] == pytest.approx(0.4)

    def test_no_appearance_information_keeps_iou(self) -> None:
        tracker = BoTSORTTracker(enable_cmc=False)
        iou_sim = np.array([[0.55, 0.2]], dtype=np.float32)
        app_sim = np.zeros((1, 2), dtype=np.float32)
        fused = tracker._fuse_botsort_gated_min(iou_sim, app_sim)
        np.testing.assert_allclose(fused, iou_sim)
