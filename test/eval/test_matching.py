# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from trackers.eval.matching import EPS, match_detections


@pytest.mark.parametrize(
    (
        "similarity_matrix",
        "threshold",
        "expected_gt",
        "expected_tr",
        "expected_unmatched_gt",
        "expected_unmatched_tr",
    ),
    [
        (
            np.array([[1.0]]),
            0.0,
            np.array([0]),
            np.array([0]),
            np.array([]),
            np.array([]),
        ),  # single perfect match
        (
            np.array(
                [
                    [0.9, 0.1, 0.0],
                    [0.2, 0.8, 0.1],
                    [0.0, 0.1, 0.7],
                ]
            ),
            0.5,
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
            np.array([]),
            np.array([]),
        ),  # 3x3 diagonal matches
        (
            np.array(
                [
                    [0.9, 0.1],
                    [0.2, 0.3],
                ]
            ),
            0.5,
            np.array([0]),
            np.array([0]),
            np.array([1]),
            np.array([1]),
        ),  # partial match, one filtered by threshold
        (
            np.array(
                [
                    [0.9, 0.85],
                    [0.8, 0.95],
                ]
            ),
            0.5,
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([]),
            np.array([]),
        ),  # optimal assignment not greedy
        (
            np.empty((0, 3)),
            0.5,
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([0, 1, 2]),
        ),  # no ground truth
        (
            np.empty((3, 0)),
            0.5,
            np.array([]),
            np.array([]),
            np.array([0, 1, 2]),
            np.array([]),
        ),  # no predictions
        (
            np.empty((0, 0)),
            0.5,
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        ),  # both empty
        (
            np.array(
                [
                    [0.3, 0.2],
                    [0.1, 0.4],
                ]
            ),
            0.5,
            np.array([]),
            np.array([]),
            np.array([0, 1]),
            np.array([0, 1]),
        ),  # all filtered by threshold
        (
            np.array(
                [
                    [0.9, 0.1, 0.0],
                    [0.2, 0.8, 0.1],
                ]
            ),
            0.5,
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([]),
            np.array([2]),
        ),  # more predictions than GT
        (
            np.array(
                [
                    [0.9, 0.1],
                    [0.2, 0.8],
                    [0.1, 0.2],
                ]
            ),
            0.5,
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([2]),
            np.array([]),
        ),  # more GT than predictions
    ],
)
def test_match_detections(
    similarity_matrix: np.ndarray[Any, np.dtype[Any]],
    threshold: float,
    expected_gt: np.ndarray[Any, np.dtype[Any]],
    expected_tr: np.ndarray[Any, np.dtype[Any]],
    expected_unmatched_gt: np.ndarray[Any, np.dtype[Any]],
    expected_unmatched_tr: np.ndarray[Any, np.dtype[Any]],
) -> None:
    gt_idx, tr_idx, unmatched_gt, unmatched_tr = match_detections(
        similarity_matrix, threshold=threshold
    )

    np.testing.assert_array_equal(np.sort(gt_idx), np.sort(expected_gt))
    np.testing.assert_array_equal(np.sort(tr_idx), np.sort(expected_tr))
    np.testing.assert_array_equal(np.sort(unmatched_gt), np.sort(expected_unmatched_gt))
    np.testing.assert_array_equal(np.sort(unmatched_tr), np.sort(expected_unmatched_tr))


@pytest.mark.parametrize(
    ("num_gt", "num_tracker", "threshold"),
    [
        (5, 5, 0.3),
        (10, 5, 0.5),
        (5, 10, 0.5),
        (20, 20, 0.4),
    ],
)
def test_match_detections_consistency_with_scipy(
    num_gt: int, num_tracker: int, threshold: float
) -> None:
    rng = np.random.default_rng(42)
    similarity_matrix = rng.random((num_gt, num_tracker))

    # Our implementation
    gt_idx, tr_idx, _unmatched_gt, _unmatched_tr = match_detections(
        similarity_matrix, threshold=threshold
    )

    # Reference implementation (TrackEval pattern)
    match_rows, match_cols = linear_sum_assignment(-similarity_matrix)
    actually_matched_mask = similarity_matrix[match_rows, match_cols] > threshold + EPS
    ref_gt_idx = match_rows[actually_matched_mask]
    ref_tr_idx = match_cols[actually_matched_mask]

    np.testing.assert_array_equal(np.sort(gt_idx), np.sort(ref_gt_idx))
    np.testing.assert_array_equal(np.sort(tr_idx), np.sort(ref_tr_idx))


def test_match_detections_returns_correct_dtypes() -> None:
    similarity_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    gt_idx, tr_idx, unmatched_gt, unmatched_tr = match_detections(
        similarity_matrix, threshold=0.5
    )

    assert gt_idx.dtype == np.intp
    assert tr_idx.dtype == np.intp
    assert unmatched_gt.dtype == np.intp
    assert unmatched_tr.dtype == np.intp


def test_match_detections_threshold_boundary() -> None:
    # Test exact threshold boundary behavior
    similarity_matrix = np.array([[0.5]])

    # At threshold, should NOT match (must be > threshold)
    gt_idx, tr_idx, _, _ = match_detections(similarity_matrix, threshold=0.5)
    assert len(gt_idx) == 0

    # Just above threshold, should match
    similarity_matrix = np.array([[0.5 + 2 * EPS]])
    gt_idx, _tr_idx, _, _ = match_detections(similarity_matrix, threshold=0.5)
    assert len(gt_idx) == 1


def test_epsilon_consistency() -> None:
    assert EPS == np.finfo("float").eps
