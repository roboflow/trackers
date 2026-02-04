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
            {"HOTA": 0.0, "DetA": 0.0, "HOTA_FN": 4 * 19, "HOTA_FP": 0, "HOTA_TP": 0},
        ),
        # No GT detections - all FP
        (
            [np.array([]), np.array([])],
            [np.array([10, 20]), np.array([10, 20])],
            [np.zeros((0, 2)), np.zeros((0, 2))],
            {"HOTA": 0.0, "DetA": 0.0, "HOTA_FP": 4 * 19, "HOTA_FN": 0, "HOTA_TP": 0},
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
def test_compute_hota_metrics(
    gt_ids: list[np.ndarray[Any, np.dtype[Any]]],
    tracker_ids: list[np.ndarray[Any, np.dtype[Any]]],
    similarity_scores: list[np.ndarray[Any, np.dtype[Any]]],
    expected: dict[str, Any],
) -> None:
    result = compute_hota_metrics(gt_ids, tracker_ids, similarity_scores)

    # Verify all expected fields are present
    for field in ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA"]:
        assert field in result
        assert isinstance(result[field], float)

    for field in ["HOTA_TP", "HOTA_FN", "HOTA_FP"]:
        assert field in result
        assert isinstance(result[field], int)

    # Check expected values
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
        elif key.endswith("_approx"):
            actual_key = key[:-7]
            assert result[actual_key] == pytest.approx(value, rel=0.01), (
                f"{actual_key} should be ~{value}, got {result[actual_key]}"
            )
        elif isinstance(value, float):
            assert result[key] == pytest.approx(value), (
                f"Mismatch for {key}: {result[key]} != {value}"
            )
        else:
            assert result[key] == value, f"Mismatch for {key}: {result[key]} != {value}"


def test_compute_hota_metrics_result_structure() -> None:
    """Verify all expected fields are present in result with correct types."""
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


@pytest.mark.parametrize(
    (
        "sequence_metrics",
        "expected",
    ),
    [
        # Empty list
        (
            [],
            {"HOTA": 0.0, "HOTA_TP": 0, "HOTA_FN": 0, "HOTA_FP": 0},
        ),
    ],
)
def test_aggregate_hota_metrics(
    sequence_metrics: list[dict[str, Any]],
    expected: dict[str, Any],
) -> None:
    result = aggregate_hota_metrics(sequence_metrics)

    for key, value in expected.items():
        if isinstance(value, float):
            assert result[key] == pytest.approx(value), (
                f"Mismatch for {key}: {result[key]} != {value}"
            )
        else:
            assert result[key] == value, f"Mismatch for {key}: {result[key]} != {value}"


def test_aggregate_hota_metrics_single_sequence() -> None:
    """Single sequence aggregation should match original."""
    seq_result = compute_hota_metrics(
        gt_ids=[np.array([0, 1])],
        tracker_ids=[np.array([10, 20])],
        similarity_scores=[np.array([[0.8, 0.0], [0.0, 0.8]])],
    )

    agg_result = aggregate_hota_metrics([seq_result])

    assert agg_result["HOTA"] == pytest.approx(seq_result["HOTA"], rel=1e-4)
    assert agg_result["DetA"] == pytest.approx(seq_result["DetA"], rel=1e-4)
    assert agg_result["AssA"] == pytest.approx(seq_result["AssA"], rel=1e-4)


def test_aggregate_hota_metrics_multiple_sequences() -> None:
    """Multiple sequences should aggregate correctly."""
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


def test_aggregate_hota_metrics_weighted_by_tp() -> None:
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
