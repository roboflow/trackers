# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from trackers.eval.clear import compute_clear_metrics


@pytest.mark.parametrize(
    (
        "gt_ids",
        "tracker_ids",
        "similarity_scores",
        "threshold",
        "expected",
    ),
    [
        # Perfect tracking - 2 objects, 3 frames
        (
            [np.array([0, 1]), np.array([0, 1]), np.array([0, 1])],
            [np.array([0, 1]), np.array([0, 1]), np.array([0, 1])],
            [
                np.array([[0.9, 0.1], [0.1, 0.9]]),
                np.array([[0.8, 0.2], [0.2, 0.85]]),
                np.array([[0.85, 0.15], [0.1, 0.9]]),
            ],
            0.5,
            {"CLR_TP": 6, "CLR_FN": 0, "CLR_FP": 0, "IDSW": 0, "MT": 2},
        ),
        # Complete miss - tracker has no detections
        (
            [np.array([0, 1]), np.array([0, 1])],
            [np.array([]), np.array([])],
            [np.empty((2, 0)), np.empty((2, 0))],
            0.5,
            {"CLR_TP": 0, "CLR_FN": 4, "CLR_FP": 0, "IDSW": 0, "ML": 2, "MOTA": 0.0},
        ),
        # All false positives - no GT
        (
            [np.array([]), np.array([])],
            [np.array([0, 1]), np.array([0, 1])],
            [np.empty((0, 2)), np.empty((0, 2))],
            0.5,
            {"CLR_TP": 0, "CLR_FN": 0, "CLR_FP": 4, "IDSW": 0, "ML": 0},
        ),
        # Single ID switch
        (
            [np.array([0]), np.array([0]), np.array([0])],
            [np.array([10]), np.array([20]), np.array([20])],
            [np.array([[0.9]]), np.array([[0.8]]), np.array([[0.85]])],
            0.5,
            {"CLR_TP": 3, "CLR_FN": 0, "CLR_FP": 0, "IDSW": 1, "MT": 1},
        ),
        # Multiple ID switches
        (
            [np.array([0]), np.array([0]), np.array([0]), np.array([0])],
            [np.array([1]), np.array([2]), np.array([3]), np.array([4])],
            [
                np.array([[0.9]]),
                np.array([[0.85]]),
                np.array([[0.8]]),
                np.array([[0.75]]),
            ],
            0.5,
            {"CLR_TP": 4, "CLR_FN": 0, "CLR_FP": 0, "IDSW": 3, "MT": 1},
        ),
        # Partial tracking - below threshold matches
        (
            [np.array([0, 1]), np.array([0, 1])],
            [np.array([0, 1]), np.array([0, 1])],
            [
                np.array([[0.8, 0.1], [0.1, 0.8]]),
                np.array([[0.3, 0.1], [0.1, 0.3]]),
            ],
            0.5,
            {"CLR_TP": 2, "CLR_FN": 2, "CLR_FP": 2, "IDSW": 0},
        ),
        # More trackers than GT (extra false positives)
        (
            [np.array([0]), np.array([0])],
            [np.array([0, 1, 2]), np.array([0, 1, 2])],
            [np.array([[0.9, 0.1, 0.1]]), np.array([[0.85, 0.1, 0.1]])],
            0.5,
            {"CLR_TP": 2, "CLR_FN": 0, "CLR_FP": 4, "IDSW": 0, "MT": 1},
        ),
        # More GT than trackers (false negatives)
        (
            [np.array([0, 1, 2]), np.array([0, 1, 2])],
            [np.array([0]), np.array([0])],
            [np.array([[0.9], [0.1], [0.1]]), np.array([[0.85], [0.1], [0.1]])],
            0.5,
            {"CLR_TP": 2, "CLR_FN": 4, "CLR_FP": 0, "IDSW": 0},
        ),
        # Empty sequence
        (
            [],
            [],
            [],
            0.5,
            {"CLR_TP": 0, "CLR_FN": 0, "CLR_FP": 0, "IDSW": 0, "ML": 0},
        ),
        # Non-diagonal optimal matching
        (
            [np.array([0, 1])],
            [np.array([10, 20])],
            [np.array([[0.3, 0.9], [0.8, 0.2]])],
            0.5,
            {"CLR_TP": 2, "CLR_FN": 0, "CLR_FP": 0, "IDSW": 0},
        ),
        # Fragmentation - track interrupted and resumed
        (
            [np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0])],
            [np.array([1]), np.array([]), np.array([1]), np.array([]), np.array([1])],
            [
                np.array([[0.9]]),
                np.empty((1, 0)),
                np.array([[0.85]]),
                np.empty((1, 0)),
                np.array([[0.8]]),
            ],
            0.5,
            {"CLR_TP": 3, "CLR_FN": 2, "CLR_FP": 0, "IDSW": 0, "Frag": 0},
        ),
        # Threshold boundary - exactly at threshold DOES match
        (
            [np.array([0])],
            [np.array([0])],
            [np.array([[0.5]])],
            0.5,
            {"CLR_TP": 1, "CLR_FN": 0, "CLR_FP": 0, "IDSW": 0},
        ),
        # Threshold boundary - just above threshold
        (
            [np.array([0])],
            [np.array([0])],
            [np.array([[0.5001]])],
            0.5,
            {"CLR_TP": 1, "CLR_FN": 0, "CLR_FP": 0, "IDSW": 0},
        ),
        # MT/PT/ML edge case - exactly 80% should be PT not MT
        (
            [
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
            ],
            [
                np.array([1]),
                np.array([1]),
                np.array([1]),
                np.array([1]),
                np.array([1]),
                np.array([1]),
                np.array([1]),
                np.array([1]),
                np.array([]),
                np.array([]),
            ],
            [
                np.array([[0.9]]),
                np.array([[0.9]]),
                np.array([[0.9]]),
                np.array([[0.9]]),
                np.array([[0.9]]),
                np.array([[0.9]]),
                np.array([[0.9]]),
                np.array([[0.9]]),
                np.empty((1, 0)),
                np.empty((1, 0)),
            ],
            0.5,
            {"MT": 0, "PT": 1, "ML": 0},
        ),
        # MT/PT/ML edge case - exactly 20% should be PT not ML
        (
            [
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
                np.array([0]),
            ],
            [
                np.array([1]),
                np.array([1]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
            ],
            [
                np.array([[0.9]]),
                np.array([[0.9]]),
                np.empty((1, 0)),
                np.empty((1, 0)),
                np.empty((1, 0)),
                np.empty((1, 0)),
                np.empty((1, 0)),
                np.empty((1, 0)),
                np.empty((1, 0)),
                np.empty((1, 0)),
            ],
            0.5,
            {"MT": 0, "PT": 1, "ML": 0},
        ),
        # Complex multi-object scenario
        (
            [
                np.array([0, 1, 2]),
                np.array([0, 1, 2]),
                np.array([0, 1]),
                np.array([0, 1, 3]),
            ],
            [
                np.array([10, 11, 12]),
                np.array([10, 11, 12]),
                np.array([10, 11]),
                np.array([10, 11, 13]),
            ],
            [
                np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]),
                np.array([[0.85, 0.1, 0.1], [0.1, 0.85, 0.1], [0.1, 0.1, 0.85]]),
                np.array([[0.9, 0.1], [0.1, 0.9]]),
                np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]),
            ],
            0.5,
            {"CLR_TP": 11, "CLR_FN": 0, "CLR_FP": 0, "IDSW": 0},
        ),
        # ID switch with IDSW minimization preference
        (
            [np.array([0, 1]), np.array([0, 1])],
            [np.array([10, 20]), np.array([10, 20])],
            [
                np.array([[0.9, 0.1], [0.1, 0.85]]),
                np.array([[0.7, 0.75], [0.75, 0.7]]),
            ],
            0.5,
            {"CLR_TP": 4, "IDSW": 0},
        ),
        # MOTA formula verification: (TP - FP - IDSW) / (TP + FN) = 0.25
        (
            [np.array([0, 1]), np.array([0, 1])],
            [np.array([10, 11, 12]), np.array([10, 20, 12])],
            [
                np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]]),
                np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]]),
            ],
            0.5,
            {"CLR_TP": 4, "CLR_FN": 0, "CLR_FP": 2, "IDSW": 1, "MOTA": 0.25},
        ),
        # MOTP formula verification: sum(similarity) / TP = 0.8
        (
            [np.array([0]), np.array([0]), np.array([0])],
            [np.array([0]), np.array([0]), np.array([0])],
            [np.array([[0.9]]), np.array([[0.8]]), np.array([[0.7]])],
            0.5,
            {"CLR_TP": 3, "MOTP": 0.8},
        ),
        # Non-sequential GT IDs (100, 200 instead of 0, 1)
        (
            [np.array([100, 200]), np.array([100, 200])],
            [np.array([1, 2]), np.array([1, 2])],
            [
                np.array([[0.9, 0.1], [0.1, 0.9]]),
                np.array([[0.85, 0.15], [0.15, 0.85]]),
            ],
            0.5,
            {"CLR_TP": 4, "CLR_FN": 0, "IDSW": 0, "MT": 2},
        ),
    ],
)
def test_compute_clear_metrics(
    gt_ids: list[np.ndarray[Any, np.dtype[Any]]],
    tracker_ids: list[np.ndarray[Any, np.dtype[Any]]],
    similarity_scores: list[np.ndarray[Any, np.dtype[Any]]],
    threshold: float,
    expected: dict[str, Any],
) -> None:
    result = compute_clear_metrics(
        gt_ids, tracker_ids, similarity_scores, threshold=threshold
    )

    for key, value in expected.items():
        if isinstance(value, float):
            assert result[key] == pytest.approx(value), (
                f"Mismatch for {key}: {result[key]} != {value}"
            )
        else:
            assert result[key] == value, f"Mismatch for {key}: {result[key]} != {value}"
