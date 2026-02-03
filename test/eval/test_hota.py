# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for HOTA metrics computation."""

from __future__ import annotations

import numpy as np
import pytest

from trackers.eval.hota import (
    ALPHA_THRESHOLDS,
    aggregate_hota_metrics,
    compute_hota_metrics,
)


class TestComputeHotaMetrics:
    """Tests for compute_hota_metrics function."""

    def test_empty_gt_and_tracker(self) -> None:
        """Empty sequences should return valid structure with zero metrics."""
        result = compute_hota_metrics(
            gt_ids=[np.array([])],
            tracker_ids=[np.array([])],
            similarity_scores=[np.zeros((0, 0))],
        )

        assert "HOTA" in result
        assert "DetA" in result
        assert "AssA" in result
        assert "HOTA_TP" in result
        assert result["HOTA_TP"] == 0
        assert result["HOTA_FN"] == 0
        assert result["HOTA_FP"] == 0

    def test_no_tracker_detections(self) -> None:
        """No tracker detections should result in all FN."""
        result = compute_hota_metrics(
            gt_ids=[np.array([0, 1]), np.array([0, 1])],
            tracker_ids=[np.array([]), np.array([])],
            similarity_scores=[np.zeros((2, 0)), np.zeros((2, 0))],
        )

        assert result["HOTA"] == 0.0
        assert result["DetA"] == 0.0
        # 4 GT dets * 19 alphas = 76 total FN
        assert result["HOTA_FN"] == 4 * len(ALPHA_THRESHOLDS)
        assert result["HOTA_FP"] == 0
        assert result["HOTA_TP"] == 0

    def test_no_gt_detections(self) -> None:
        """No GT detections should result in all FP."""
        result = compute_hota_metrics(
            gt_ids=[np.array([]), np.array([])],
            tracker_ids=[np.array([10, 20]), np.array([10, 20])],
            similarity_scores=[np.zeros((0, 2)), np.zeros((0, 2))],
        )

        assert result["HOTA"] == 0.0
        assert result["DetA"] == 0.0
        # 4 tracker dets * 19 alphas = 76 total FP
        assert result["HOTA_FP"] == 4 * len(ALPHA_THRESHOLDS)
        assert result["HOTA_FN"] == 0
        assert result["HOTA_TP"] == 0

    def test_perfect_tracking(self) -> None:
        """Perfect tracking with high IoU should give high HOTA."""
        # Two frames, two objects, perfect tracking with IoU=0.9
        result = compute_hota_metrics(
            gt_ids=[np.array([0, 1]), np.array([0, 1])],
            tracker_ids=[np.array([10, 20]), np.array([10, 20])],
            similarity_scores=[
                np.array([[0.9, 0.0], [0.0, 0.9]]),
                np.array([[0.9, 0.0], [0.0, 0.9]]),
            ],
        )

        # Should have high HOTA since IoU=0.9 passes most thresholds
        assert result["HOTA"] > 0.8
        assert result["DetA"] > 0.8
        assert result["AssA"] > 0.9  # Perfect association
        # LocA is averaged over alphas where matches occurred
        assert result["LocA"] == pytest.approx(0.9, rel=0.01)

    def test_id_switch(self) -> None:
        """ID switch should reduce AssA."""
        # Frame 1: GT0->T10, GT1->T20
        # Frame 2: GT0->T20, GT1->T10 (switched!)
        result = compute_hota_metrics(
            gt_ids=[np.array([0, 1]), np.array([0, 1])],
            tracker_ids=[np.array([10, 20]), np.array([10, 20])],
            similarity_scores=[
                np.array([[0.8, 0.1], [0.1, 0.8]]),  # Normal matching
                np.array([[0.1, 0.8], [0.8, 0.1]]),  # Swapped!
            ],
        )

        # AssA should be lower due to ID switch
        # But DetA should still be reasonable
        assert result["DetA"] > 0.5
        assert result["AssA"] < 0.8  # Penalized for ID switch

    def test_low_iou_threshold_sensitivity(self) -> None:
        """Lower IoU should pass fewer alpha thresholds."""
        # With IoU=0.3, only alphas <= 0.3 will match
        result = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.3]])],
        )

        # Should have some TP at low alphas, but many FN at high alphas
        # Overall HOTA should be moderate
        assert 0 < result["HOTA"] < 0.5
        assert result["HOTA_TP"] > 0  # Some matches at low alphas
        assert result["HOTA_FN"] > 0  # Misses at high alphas

    def test_multiple_objects_partial_match(self) -> None:
        """Test with multiple objects where only some match."""
        result = compute_hota_metrics(
            gt_ids=[np.array([0, 1, 2])],
            tracker_ids=[np.array([10, 20])],  # Only 2 trackers for 3 GTs
            similarity_scores=[
                np.array(
                    [
                        [0.8, 0.0],
                        [0.0, 0.8],
                        [0.0, 0.0],  # GT2 has no match
                    ]
                )
            ],
        )

        # GT2 is unmatched at all alphas (1 FN per alpha)
        # GT0 and GT1 are matched at alphas <= 0.8 only
        # At alphas > 0.8, GT0 and GT1 also become FN
        assert result["HOTA_FN"] > len(ALPHA_THRESHOLDS)  # More than just 1 FN/alpha
        assert result["DetRe"] < 1.0  # Not all GTs matched
        assert result["DetPr"] < 1.0  # At high alphas, no matches

    def test_result_structure(self) -> None:
        """Verify all expected fields are present in result."""
        result = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.8]])],
        )

        # Float summary fields
        for field in [
            "HOTA",
            "DetA",
            "AssA",
            "DetRe",
            "DetPr",
            "AssRe",
            "AssPr",
            "LocA",
            "OWTA",
        ]:
            assert field in result
            assert isinstance(result[field], float)
            assert 0 <= result[field] <= 1

        # Integer summary fields
        for field in ["HOTA_TP", "HOTA_FN", "HOTA_FP"]:
            assert field in result
            assert isinstance(result[field], int)
            assert result[field] >= 0

        # Array fields for aggregation
        for field in [
            "HOTA_TP_array",
            "HOTA_FN_array",
            "HOTA_FP_array",
            "AssA_array",
            "AssRe_array",
            "AssPr_array",
            "LocA_array",
        ]:
            assert field in result
            assert isinstance(result[field], np.ndarray)
            assert len(result[field]) == len(ALPHA_THRESHOLDS)


class TestAggregateHotaMetrics:
    """Tests for aggregate_hota_metrics function."""

    def test_empty_list(self) -> None:
        """Empty sequence list should return zero metrics."""
        result = aggregate_hota_metrics([])

        assert result["HOTA"] == 0.0
        assert result["HOTA_TP"] == 0
        assert result["HOTA_FN"] == 0
        assert result["HOTA_FP"] == 0

    def test_single_sequence(self) -> None:
        """Single sequence aggregation should match original."""
        seq_result = compute_hota_metrics(
            gt_ids=[np.array([0, 1])],
            tracker_ids=[np.array([10, 20])],
            similarity_scores=[np.array([[0.8, 0.0], [0.0, 0.8]])],
        )

        agg_result = aggregate_hota_metrics([seq_result])

        # Should be very close to original (minor floating point differences)
        assert agg_result["HOTA"] == pytest.approx(seq_result["HOTA"], rel=1e-4)
        assert agg_result["DetA"] == pytest.approx(seq_result["DetA"], rel=1e-4)
        assert agg_result["AssA"] == pytest.approx(seq_result["AssA"], rel=1e-4)

    def test_multiple_sequences(self) -> None:
        """Multiple sequences should aggregate correctly."""
        # Two identical sequences
        seq_result1 = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.9]])],
        )
        seq_result2 = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.9]])],
        )

        agg_result = aggregate_hota_metrics([seq_result1, seq_result2])

        # TP/FN/FP should be summed
        expected_tp = seq_result1["HOTA_TP"] + seq_result2["HOTA_TP"]
        assert agg_result["HOTA_TP"] == expected_tp

        # HOTA should be similar to individual sequences since they're identical
        assert agg_result["HOTA"] == pytest.approx(seq_result1["HOTA"], rel=0.01)

    def test_aggregation_with_different_qualities(self) -> None:
        """Aggregation should weight by TP count."""
        # High quality sequence (many TPs)
        high_quality = compute_hota_metrics(
            gt_ids=[np.array([0, 1, 2, 3])],
            tracker_ids=[np.array([10, 20, 30, 40])],
            similarity_scores=[np.diag([0.9, 0.9, 0.9, 0.9])],
        )

        # Low quality sequence (few TPs)
        low_quality = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.3]])],
        )

        agg_result = aggregate_hota_metrics([high_quality, low_quality])

        # Result should be closer to high_quality since it has more TPs
        assert agg_result["HOTA"] > low_quality["HOTA"]
