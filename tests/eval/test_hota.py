# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trackers.eval.hota import (
    ALPHA_THRESHOLDS,
    aggregate_hota_metrics,
    compute_hota_metrics,
)


class TestComputeHOTAMetrics:
    """Per-sequence HOTA metric computation (DetA, AssA, LocA, HOTA)."""

    @pytest.mark.parametrize(
        (
            "gt_ids",
            "tracker_ids",
            "similarity_scores",
            "expected",
        ),
        [
            # Empty GT and tracker
            (
                [np.array([])],
                [np.array([])],
                [np.zeros((0, 0))],
                {"HOTA_TP": 0, "HOTA_FN": 0, "HOTA_FP": 0},
            ),
            # No tracker detections - all FN
            (
                [np.array([0, 1]), np.array([0, 1])],
                [np.array([]), np.array([])],
                [np.zeros((2, 0)), np.zeros((2, 0))],
                {
                    "HOTA": 0.0,
                    "DetA": 0.0,
                    "HOTA_FN": 4 * 19,
                    "HOTA_FP": 0,
                    "HOTA_TP": 0,
                },
            ),
            # No GT detections - all FP
            (
                [np.array([]), np.array([])],
                [np.array([10, 20]), np.array([10, 20])],
                [np.zeros((0, 2)), np.zeros((0, 2))],
                {
                    "HOTA": 0.0,
                    "DetA": 0.0,
                    "HOTA_FP": 4 * 19,
                    "HOTA_FN": 0,
                    "HOTA_TP": 0,
                },
            ),
            # Perfect tracking with high IoU
            (
                [np.array([0, 1]), np.array([0, 1])],
                [np.array([10, 20]), np.array([10, 20])],
                [
                    np.array([[0.9, 0.0], [0.0, 0.9]]),
                    np.array([[0.9, 0.0], [0.0, 0.9]]),
                ],
                {"HOTA_min": 0.8, "DetA_min": 0.8, "AssA_min": 0.9},
            ),
            # ID switch reduces AssA
            (
                [np.array([0, 1]), np.array([0, 1])],
                [np.array([10, 20]), np.array([10, 20])],
                [
                    np.array([[0.8, 0.1], [0.1, 0.8]]),  # Normal matching
                    np.array([[0.1, 0.8], [0.8, 0.1]]),  # Swapped!
                ],
                {"DetA_min": 0.5, "AssA_max": 0.8},
            ),
            # Low IoU passes fewer alpha thresholds
            (
                [np.array([0])],
                [np.array([10])],
                [np.array([[0.3]])],
                {"HOTA_min": 0.0, "HOTA_max": 0.5, "HOTA_TP_min": 1},
            ),
            # Multiple objects partial match
            (
                [np.array([0, 1, 2])],
                [np.array([10, 20])],  # Only 2 trackers for 3 GTs
                [
                    np.array(
                        [
                            [0.8, 0.0],
                            [0.0, 0.8],
                            [0.0, 0.0],  # GT2 has no match
                        ]
                    )
                ],
                {"HOTA_FN_min": 19, "DetRe_max": 1.0, "DetPr_max": 1.0},
            ),
            # Single frame perfect matching
            (
                [np.array([0])],
                [np.array([10])],
                [np.array([[0.8]])],
                {"HOTA_TP_min": 1, "LocA_min": 0.8},
            ),
        ],
    )
    def test_scenarios(
        self,
        gt_ids: list[np.ndarray[Any, np.dtype[Any]]],
        tracker_ids: list[np.ndarray[Any, np.dtype[Any]]],
        similarity_scores: list[np.ndarray[Any, np.dtype[Any]]],
        expected: dict[str, Any],
    ) -> None:
        """compute_hota_metrics produces expected HOTA fields across scenarios."""
        result = compute_hota_metrics(gt_ids, tracker_ids, similarity_scores)

        for field in [
            "HOTA",
            "DetA",
            "AssA",
            "DetRe",
            "DetPr",
            "AssRe",
            "AssPr",
            "LocA",
        ]:
            assert field in result
            assert isinstance(result[field], float)

        for field in ["HOTA_TP", "HOTA_FN", "HOTA_FP"]:
            assert field in result
            assert isinstance(result[field], int)

        for key, value in expected.items():
            if key.endswith("_min"):
                actual_key = key[:-4]
                assert result[actual_key] >= value, f"{actual_key} should be >= {value}, got {result[actual_key]}"
            elif key.endswith("_max"):
                actual_key = key[:-4]
                assert result[actual_key] <= value, f"{actual_key} should be <= {value}, got {result[actual_key]}"
            elif key.endswith("_approx"):
                actual_key = key[:-7]
                assert result[actual_key] == pytest.approx(value, rel=0.01), (
                    f"{actual_key} should be ~{value}, got {result[actual_key]}"
                )
            elif isinstance(value, float):
                assert result[key] == pytest.approx(value), f"Mismatch for {key}: {result[key]} != {value}"
            else:
                assert result[key] == value, f"Mismatch for {key}: {result[key]} != {value}"

    def test_result_structure(self) -> None:
        """Result contains all float summary, integer count, and array fields."""
        result = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.8]])],
        )

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

        for field in ["HOTA_TP", "HOTA_FN", "HOTA_FP"]:
            assert field in result
            assert isinstance(result[field], int)
            assert result[field] >= 0

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


class TestAggregateHOTAMetrics:
    """Multi-sequence aggregation of HOTA metrics."""

    def test_empty(self) -> None:
        """Empty sequence list returns all-zero HOTA metrics."""
        result = aggregate_hota_metrics([])
        assert result["HOTA"] == pytest.approx(0.0)
        assert result["HOTA_TP"] == 0
        assert result["HOTA_FN"] == 0
        assert result["HOTA_FP"] == 0

    def test_single_sequence(self) -> None:
        """Single-sequence aggregation reproduces per-sequence HOTA scores."""
        seq_result = compute_hota_metrics(
            gt_ids=[np.array([0, 1])],
            tracker_ids=[np.array([10, 20])],
            similarity_scores=[np.array([[0.8, 0.0], [0.0, 0.8]])],
        )

        agg_result = aggregate_hota_metrics([seq_result])

        assert agg_result["HOTA"] == pytest.approx(seq_result["HOTA"], rel=1e-4)
        assert agg_result["DetA"] == pytest.approx(seq_result["DetA"], rel=1e-4)
        assert agg_result["AssA"] == pytest.approx(seq_result["AssA"], rel=1e-4)

    def test_multiple_sequences(self) -> None:
        """Identical sequences aggregate to the same HOTA; TP counts are summed."""
        seq1 = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.9]])],
        )
        seq2 = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.9]])],
        )

        agg_result = aggregate_hota_metrics([seq1, seq2])

        assert agg_result["HOTA_TP"] == seq1["HOTA_TP"] + seq2["HOTA_TP"]
        assert agg_result["HOTA"] == pytest.approx(seq1["HOTA"], rel=0.01)

    def test_weighted_by_tp(self) -> None:
        """Aggregated HOTA skews toward the higher-TP sequence."""
        high_quality = compute_hota_metrics(
            gt_ids=[np.array([0, 1, 2, 3])],
            tracker_ids=[np.array([10, 20, 30, 40])],
            similarity_scores=[np.diag([0.9, 0.9, 0.9, 0.9])],
        )
        low_quality = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.3]])],
        )

        agg_result = aggregate_hota_metrics([high_quality, low_quality])

        assert agg_result["HOTA"] > low_quality["HOTA"]
