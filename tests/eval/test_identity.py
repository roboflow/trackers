# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trackers.eval.identity import (
    aggregate_identity_metrics,
    compute_identity_metrics,
)


class TestComputeIdentityMetrics:
    """Per-sequence Identity metric computation (IDTP, IDFN, IDFP, IDF1)."""

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
                {"IDTP": 0, "IDFN": 0, "IDFP": 0, "IDF1": 0.0},
            ),
            # No tracker detections - all FN
            (
                [np.array([0, 1]), np.array([0, 1])],
                [np.array([]), np.array([])],
                [np.zeros((2, 0)), np.zeros((2, 0))],
                {"IDTP": 0, "IDFN": 4, "IDFP": 0, "IDF1": 0.0, "IDR": 0.0},
            ),
            # No GT detections - all FP
            (
                [np.array([]), np.array([])],
                [np.array([10, 20]), np.array([10, 20])],
                [np.zeros((0, 2)), np.zeros((0, 2))],
                {"IDTP": 0, "IDFN": 0, "IDFP": 4, "IDF1": 0.0, "IDP": 0.0},
            ),
            # Perfect tracking - all detections matched with correct IDs
            (
                [np.array([0, 1]), np.array([0, 1])],
                [np.array([10, 20]), np.array([10, 20])],
                [
                    np.array([[0.8, 0.0], [0.0, 0.8]]),
                    np.array([[0.8, 0.0], [0.0, 0.8]]),
                ],
                {
                    "IDTP": 4,
                    "IDFN": 0,
                    "IDFP": 0,
                    "IDF1": 1.0,
                    "IDR": 1.0,
                    "IDP": 1.0,
                },
            ),
            # ID switch - GT 0 matched tracker 10 in frame 1, tracker 20 in frame 2
            (
                [np.array([0, 1]), np.array([0, 1])],
                [np.array([10, 20]), np.array([10, 20])],
                [
                    np.array([[0.8, 0.1], [0.1, 0.8]]),  # Normal matching
                    np.array([[0.1, 0.8], [0.8, 0.1]]),  # Swapped!
                ],
                # With ID switch, each GT can only match one tracker globally
                # So 2 IDTP per ID = 4 total, but need to split FN/FP
                {"IDTP_min": 2, "IDF1_min": 0.3},
            ),
            # Low IoU below threshold - no matches
            (
                [np.array([0])],
                [np.array([10])],
                [np.array([[0.3]])],  # Below default 0.5 threshold
                {"IDTP": 0, "IDFN": 1, "IDFP": 1, "IDF1": 0.0},
            ),
            # Multiple objects partial match
            (
                [np.array([0, 1, 2])],
                [np.array([10, 20])],  # Only 2 trackers for 3 GTs
                [np.array([[0.8, 0.0], [0.0, 0.8], [0.0, 0.0]])],
                {"IDTP": 2, "IDFN": 1, "IDFP": 0},
            ),
            # Extra tracker detections
            (
                [np.array([0])],
                [np.array([10, 20, 30])],  # 3 trackers for 1 GT
                [np.array([[0.8, 0.0, 0.0]])],
                {"IDTP": 1, "IDFN": 0, "IDFP": 2},
            ),
        ],
    )
    def test_scenarios(
        self,
        gt_ids: list[np.ndarray],
        tracker_ids: list[np.ndarray],
        similarity_scores: list[np.ndarray],
        expected: dict[str, Any],
    ) -> None:
        """Test Identity metrics computation for various scenarios."""
        result = compute_identity_metrics(gt_ids, tracker_ids, similarity_scores)

        for key, value in expected.items():
            if key.endswith("_min"):
                actual_key = key[:-4]
                assert result[actual_key] >= value, (
                    f"{actual_key} should be >= {value}, got {result[actual_key]}"
                )
            elif key.endswith("_max"):
                actual_key = key[:-4]
                assert result[actual_key] <= value, (
                    f"{actual_key} should be <= {value}, got {result[actual_key]}"
                )
            else:
                if isinstance(value, float):
                    assert result[key] == pytest.approx(value, abs=1e-6), (
                        f"{key} mismatch: expected {value}, got {result[key]}"
                    )
                else:
                    assert result[key] == value, (
                        f"{key} mismatch: expected {value}, got {result[key]}"
                    )

    @pytest.mark.parametrize(
        ("threshold", "expected_idtp"),
        [
            (0.5, 0),  # default threshold — no match
            (0.3, 1),  # lower threshold — should match
        ],
        ids=["threshold_0.5_no_match", "threshold_0.3_matches"],
    )
    def test_threshold_gate(
        self,
        threshold: float,
        expected_idtp: int,
    ) -> None:
        """IoU threshold controls whether a 0.4-similarity pair matches."""
        gt_ids = [np.array([0])]
        tracker_ids = [np.array([10])]
        similarity_scores = [np.array([[0.4]])]
        result = compute_identity_metrics(
            gt_ids, tracker_ids, similarity_scores, threshold=threshold
        )
        assert result["IDTP"] == expected_idtp

    def test_multi_frame_consistency(self) -> None:
        """Consistent tracking across frames yields perfect IDTP/IDFN/IDFP."""
        gt_ids = [np.array([0]), np.array([0]), np.array([0])]
        tracker_ids = [np.array([10]), np.array([10]), np.array([10])]
        similarity_scores = [
            np.array([[0.9]]),
            np.array([[0.85]]),
            np.array([[0.8]]),
        ]

        result = compute_identity_metrics(gt_ids, tracker_ids, similarity_scores)

        assert result["IDTP"] == 3
        assert result["IDFN"] == 0
        assert result["IDFP"] == 0
        assert result["IDF1"] == pytest.approx(1.0)


class TestAggregateIdentityMetrics:
    """Multi-sequence aggregation of Identity metrics."""

    def test_empty(self) -> None:
        """Empty sequence list returns all-zero Identity metrics."""
        result = aggregate_identity_metrics([])

        assert result["IDTP"] == 0
        assert result["IDFN"] == 0
        assert result["IDFP"] == 0
        assert result["IDF1"] == 0.0

    def test_single_sequence(self) -> None:
        """Single-sequence aggregation preserves raw counts; recomputes IDF1."""
        seq_result = {
            "IDTP": 100,
            "IDFN": 10,
            "IDFP": 5,
            "IDF1": 0.93,
            "IDR": 0.91,
            "IDP": 0.95,
        }

        agg = aggregate_identity_metrics([seq_result])

        assert agg["IDTP"] == 100
        assert agg["IDFN"] == 10
        assert agg["IDFP"] == 5
        expected_idf1 = 100 / (100 + 0.5 * 5 + 0.5 * 10)
        assert agg["IDF1"] == pytest.approx(expected_idf1)

    def test_multiple_sequences(self) -> None:
        """Multi-sequence aggregation sums raw counts and recomputes IDF1."""
        seq1 = compute_identity_metrics(
            gt_ids=[np.array([0, 1])],
            tracker_ids=[np.array([10, 20])],
            similarity_scores=[np.array([[0.9, 0.0], [0.0, 0.9]])],
        )
        seq2 = compute_identity_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.9]])],
        )

        agg = aggregate_identity_metrics([seq1, seq2])

        expected_idtp = seq1["IDTP"] + seq2["IDTP"]
        expected_idfn = seq1["IDFN"] + seq2["IDFN"]
        expected_idfp = seq1["IDFP"] + seq2["IDFP"]

        assert agg["IDTP"] == expected_idtp
        assert agg["IDFN"] == expected_idfn
        assert agg["IDFP"] == expected_idfp

        expected_idf1 = expected_idtp / max(
            1.0, expected_idtp + 0.5 * expected_idfp + 0.5 * expected_idfn
        )
        assert agg["IDF1"] == pytest.approx(expected_idf1)
