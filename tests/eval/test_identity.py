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
            {"IDTP": 4, "IDFN": 0, "IDFP": 0, "IDF1": 1.0, "IDR": 1.0, "IDP": 1.0},
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
def test_compute_identity_metrics(
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


def test_compute_identity_metrics_custom_threshold() -> None:
    """Test Identity metrics with custom IoU threshold."""
    gt_ids = [np.array([0])]
    tracker_ids = [np.array([10])]
    similarity_scores = [np.array([[0.4]])]

    # Default threshold 0.5 - no match
    result_high = compute_identity_metrics(
        gt_ids, tracker_ids, similarity_scores, threshold=0.5
    )
    assert result_high["IDTP"] == 0

    # Lower threshold 0.3 - should match
    result_low = compute_identity_metrics(
        gt_ids, tracker_ids, similarity_scores, threshold=0.3
    )
    assert result_low["IDTP"] == 1


def test_compute_identity_metrics_multi_frame_consistency() -> None:
    """Test that Identity correctly handles consistent tracking across frames."""
    # GT ID 0 appears in all 3 frames
    # Tracker ID 10 tracks it consistently
    gt_ids = [np.array([0]), np.array([0]), np.array([0])]
    tracker_ids = [np.array([10]), np.array([10]), np.array([10])]
    similarity_scores = [
        np.array([[0.9]]),
        np.array([[0.85]]),
        np.array([[0.8]]),
    ]

    result = compute_identity_metrics(gt_ids, tracker_ids, similarity_scores)

    # Perfect tracking: 3 IDTP, 0 IDFN, 0 IDFP
    assert result["IDTP"] == 3
    assert result["IDFN"] == 0
    assert result["IDFP"] == 0
    assert result["IDF1"] == pytest.approx(1.0)


def test_aggregate_identity_metrics_empty() -> None:
    """Test aggregation with empty input."""
    result = aggregate_identity_metrics([])

    assert result["IDTP"] == 0
    assert result["IDFN"] == 0
    assert result["IDFP"] == 0
    assert result["IDF1"] == 0.0


def test_aggregate_identity_metrics_single_sequence() -> None:
    """Test aggregation with single sequence returns same values."""
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
    # IDF1 is recomputed from sums
    expected_idf1 = 100 / (100 + 0.5 * 5 + 0.5 * 10)
    assert agg["IDF1"] == pytest.approx(expected_idf1)


def test_aggregate_identity_metrics_multiple_sequences() -> None:
    """Test aggregation sums counts and recomputes ratios."""
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

    # IDTP/IDFN/IDFP should be summed
    expected_idtp = seq1["IDTP"] + seq2["IDTP"]
    expected_idfn = seq1["IDFN"] + seq2["IDFN"]
    expected_idfp = seq1["IDFP"] + seq2["IDFP"]

    assert agg["IDTP"] == expected_idtp
    assert agg["IDFN"] == expected_idfn
    assert agg["IDFP"] == expected_idfp

    # IDF1 should be recomputed from totals
    expected_idf1 = expected_idtp / max(
        1.0, expected_idtp + 0.5 * expected_idfp + 0.5 * expected_idfn
    )
    assert agg["IDF1"] == pytest.approx(expected_idf1)
