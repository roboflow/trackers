# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.core.mcbyte.mask_association import (
    _get_mask_metrics,
    condition_similarity_with_masks,
)
from trackers.core.mcbyte.masks.base import MaskOutput


def _mask_output(
    masks: np.ndarray,
    tracklet_mask_dict: dict[int, int],
    confidences: dict[int, float],
) -> MaskOutput:
    return MaskOutput(
        masks=masks,
        tracklet_mask_dict=tracklet_mask_dict,
        mask_avg_prob_dict=confidences,
    )


def _full_mask(
    height: int = 10,
    width: int = 10,
) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def test_mask_metrics_compute_coverage_and_fill_ratio() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 2:6] = True

    metrics = _get_mask_metrics(
        mask=mask,
        detection_xyxy=np.array([2, 2, 6, 6], dtype=np.float32),
    )

    assert metrics is not None
    mask_coverage, mask_fill_ratio = metrics
    assert np.isclose(mask_coverage, 1.0)
    assert np.isclose(mask_fill_ratio, 1.0)


def test_clear_match_is_locked_and_removed_from_remaining_problem() -> None:
    similarity = np.array(
        [
            [0.8, 0.1],
            [0.1, 0.7],
        ],
        dtype=np.float32,
    )

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10, 20],
        detection_boxes=np.array(
            [
                [0, 0, 5, 5],
                [5, 5, 10, 10],
            ],
            dtype=np.float32,
        ),
        mask_output=None,
        minimum_similarity=0.5,
    )

    assert result.locked_matches == [(0, 0), (1, 1)]
    assert result.remaining_track_indices == []
    assert result.remaining_detection_indices == []
    assert result.conditioned_similarity.shape == (0, 0)


def test_ambiguous_row_receives_mask_fill_bonus() -> None:
    similarity = np.array(
        [
            [0.7, 0.6],
        ],
        dtype=np.float32,
    )
    masks = np.zeros((1, 10, 10), dtype=bool)
    masks[0, 0:5, 0:5] = True

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array(
            [
                [0, 0, 5, 5],
                [5, 5, 10, 10],
            ],
            dtype=np.float32,
        ),
        mask_output=_mask_output(
            masks=masks,
            tracklet_mask_dict={10: 0},
            confidences={10: 0.9},
        ),
        minimum_similarity=0.5,
    )

    np.testing.assert_allclose(
        result.conditioned_similarity,
        np.array([[1.7, 0.6]], dtype=np.float32),
    )


def test_ambiguous_column_pair_receives_mask_fill_bonus() -> None:
    similarity = np.array(
        [
            [0.7],
            [0.6],
        ],
        dtype=np.float32,
    )

    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 0:5, 0:5] = True
    masks[1, 5:10, 5:10] = True

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10, 20],
        detection_boxes=np.array(
            [
                [0, 0, 5, 5],
            ],
            dtype=np.float32,
        ),
        mask_output=_mask_output(
            masks=masks,
            tracklet_mask_dict={
                10: 0,
                20: 1,
            },
            confidences={
                10: 0.9,
                20: 0.9,
            },
        ),
        minimum_similarity=0.5,
    )

    np.testing.assert_allclose(
        result.conditioned_similarity,
        np.array(
            [
                [1.7],
                [0.6],
            ],
            dtype=np.float32,
        ),
    )


def test_ambiguity_is_computed_from_original_similarity_matrix() -> None:
    # Row 0 is ambiguous, column 1 is ambiguous, (1, 1) is also considered ambiguous
    # because its column has two eligible tracklets.
    similarity = np.array(
        [
            [0.7, 0.6],
            [0.1, 0.8],
        ],
        dtype=np.float32,
    )

    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 0:5, 0:5] = True
    masks[1, 5:10, 5:10] = True

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10, 20],
        detection_boxes=np.array(
            [
                [0, 0, 5, 5],
                [5, 5, 10, 10],
            ],
            dtype=np.float32,
        ),
        mask_output=_mask_output(
            masks=masks,
            tracklet_mask_dict={
                10: 0,
                20: 1,
            },
            confidences={
                10: 0.9,
                20: 0.9,
            },
        ),
        minimum_similarity=0.5,
    )

    assert result.locked_matches == []

    # Ensure that both mask bonuses are decided from the untouched original matrix,
    # rather than progressively recalculating ambiguity after one score is modified
    np.testing.assert_allclose(
        result.conditioned_similarity,
        np.array(
            [
                [1.7, 0.6],
                [0.1, 1.8],
            ],
            dtype=np.float32,
        ),
    )


def test_locked_matches_and_ambiguous_subproblem_preserve_original_indices() -> None:
    similarity = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.7, 0.6],
            [0.0, 0.6, 0.7],
        ],
        dtype=np.float32,
    )

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10, 20, 30],
        detection_boxes=np.array(
            [
                [0, 0, 5, 5],
                [5, 5, 10, 10],
                [10, 10, 15, 15],
            ],
            dtype=np.float32,
        ),
        mask_output=None,
        minimum_similarity=0.5,
    )

    assert result.locked_matches == [(0, 0)]
    assert result.remaining_track_indices == [1, 2]
    assert result.remaining_detection_indices == [1, 2]

    np.testing.assert_array_equal(
        result.conditioned_similarity,
        np.array(
            [
                [0.7, 0.6],
                [0.6, 0.7],
            ],
            dtype=np.float32,
        ),
    )


def test_multiple_ambiguous_pairs_receive_independent_mask_bonuses() -> None:
    similarity = np.array(
        [
            [0.7, 0.6],
            [0.6, 0.7],
        ],
        dtype=np.float32,
    )

    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 0:5, 0:5] = True
    masks[1, 5:10, 5:10] = True

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10, 20],
        detection_boxes=np.array(
            [
                [0, 0, 5, 5],
                [5, 5, 10, 10],
            ],
            dtype=np.float32,
        ),
        mask_output=_mask_output(
            masks=masks,
            tracklet_mask_dict={
                10: 0,
                20: 1,
            },
            confidences={
                10: 0.9,
                20: 0.9,
            },
        ),
        minimum_similarity=0.5,
    )

    np.testing.assert_allclose(
        result.conditioned_similarity,
        np.array(
            [
                [1.7, 0.6],
                [0.6, 1.7],
            ],
            dtype=np.float32,
        ),
    )


def test_mask_bonus_is_not_clamped_to_one() -> None:
    similarity = np.array([[0.8, 0.7]], dtype=np.float32)

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 5, 5],
            ],
            dtype=np.float32,
        ),
        mask_output=_mask_output(
            masks=np.stack([_full_mask()]),
            tracklet_mask_dict={10: 0},
            confidences={10: 0.9},
        ),
        minimum_similarity=0.5,
    )

    assert np.isclose(result.conditioned_similarity[0, 0], 1.8)


def test_missing_mask_output_keeps_ambiguous_scores_unchanged() -> None:
    similarity = np.array([[0.7, 0.6]], dtype=np.float32)

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array(
            [
                [0, 0, 5, 5],
                [5, 5, 10, 10],
            ],
            dtype=np.float32,
        ),
        mask_output=None,
        minimum_similarity=0.5,
    )

    np.testing.assert_array_equal(
        result.conditioned_similarity,
        similarity,
    )


def test_missing_tracklet_mask_keeps_scores_unchanged() -> None:
    similarity = np.array([[0.7, 0.6]], dtype=np.float32)

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array(
            [
                [0, 0, 5, 5],
                [5, 5, 10, 10],
            ],
            dtype=np.float32,
        ),
        mask_output=_mask_output(
            masks=np.stack([_full_mask()]),
            tracklet_mask_dict={99: 0},
            confidences={99: 0.9},
        ),
        minimum_similarity=0.5,
    )

    np.testing.assert_array_equal(
        result.conditioned_similarity,
        similarity,
    )


def test_invalid_mask_index_keeps_scores_unchanged() -> None:
    similarity = np.array([[0.7, 0.6]], dtype=np.float32)

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array(
            [
                [0, 0, 5, 5],
                [5, 5, 10, 10],
            ],
            dtype=np.float32,
        ),
        mask_output=_mask_output(
            masks=np.stack([_full_mask()]),
            # valid mask index in this case would be 0
            tracklet_mask_dict={
                10: 5,
            },
            confidences={
                10: 0.9,
            },
        ),
        minimum_similarity=0.5,
    )

    np.testing.assert_array_equal(
        result.conditioned_similarity,
        similarity,
    )


def test_missing_mask_confidence_keeps_scores_unchanged() -> None:
    similarity = np.array([[0.7, 0.6]], dtype=np.float32)

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 5, 5],
            ],
            dtype=np.float32,
        ),
        mask_output=_mask_output(
            masks=np.stack([_full_mask()]),
            tracklet_mask_dict={
                10: 0,
            },
            confidences={},
        ),
        minimum_similarity=0.5,
    )

    np.testing.assert_array_equal(
        result.conditioned_similarity,
        similarity,
    )


def test_low_mask_confidence_keeps_scores_unchanged() -> None:
    similarity = np.array([[0.7, 0.6]], dtype=np.float32)

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 5, 5],
            ],
            dtype=np.float32,
        ),
        mask_output=_mask_output(
            masks=np.stack([_full_mask()]),
            tracklet_mask_dict={10: 0},
            confidences={10: 0.5},
        ),
        minimum_similarity=0.5,
        minimum_mask_average_confidence=0.6,
    )

    np.testing.assert_array_equal(
        result.conditioned_similarity,
        similarity,
    )


def test_low_mask_coverage_keeps_entry_unchanged() -> None:
    similarity = np.array([[0.7, 0.6]], dtype=np.float32)

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array(
            [
                [0, 0, 5, 5],
                [5, 5, 10, 10],
            ],
            dtype=np.float32,
        ),
        mask_output=_mask_output(
            masks=np.stack([_full_mask()]),
            tracklet_mask_dict={10: 0},
            confidences={10: 0.9},
        ),
        minimum_similarity=0.5,
        minimum_mask_coverage=0.9,
    )

    np.testing.assert_array_equal(
        result.conditioned_similarity,
        similarity,
    )


def test_low_mask_fill_ratio_keeps_entry_unchanged() -> None:
    similarity = np.array([[0.7, 0.6]], dtype=np.float32)
    mask = np.zeros((20, 20), dtype=bool)
    mask[0:2, 0:2] = True

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array(
            [
                [0, 0, 20, 20],
                [0, 0, 10, 10],
            ],
            dtype=np.float32,
        ),
        mask_output=_mask_output(
            masks=np.stack([mask]),
            tracklet_mask_dict={10: 0},
            confidences={10: 0.9},
        ),
        minimum_similarity=0.5,
        minimum_mask_coverage=0.9,
        minimum_mask_fill_ratio=0.05,
    )

    np.testing.assert_array_equal(
        result.conditioned_similarity,
        similarity,
    )


def test_empty_mask_keeps_scores_unchanged() -> None:
    similarity = np.array([[0.7, 0.6]], dtype=np.float32)

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array(
            [
                [0, 0, 5, 5],
                [5, 5, 10, 10],
            ],
            dtype=np.float32,
        ),
        mask_output=_mask_output(
            masks=np.zeros((1, 10, 10), dtype=bool),
            tracklet_mask_dict={10: 0},
            confidences={10: 0.9},
        ),
        minimum_similarity=0.5,
    )

    np.testing.assert_array_equal(
        result.conditioned_similarity,
        similarity,
    )


def test_isolated_below_threshold_pair_is_unchanged_when_disabled() -> None:
    similarity = np.array([[0.2]], dtype=np.float32)

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
        mask_output=_mask_output(
            masks=np.stack([_full_mask()]),
            tracklet_mask_dict={10: 0},
            confidences={10: 0.9},
        ),
        minimum_similarity=0.5,
        enable_isolated_mask_matching=False,
    )

    np.testing.assert_array_equal(
        result.conditioned_similarity,
        similarity,
    )


def test_isolated_below_threshold_pair_is_boosted_when_enabled() -> None:
    similarity = np.array([[0.2]], dtype=np.float32)

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
        mask_output=_mask_output(
            masks=np.stack([_full_mask()]),
            tracklet_mask_dict={10: 0},
            confidences={10: 0.9},
        ),
        minimum_similarity=0.5,
        enable_isolated_mask_matching=True,
    )

    assert np.isclose(result.conditioned_similarity[0, 0], 1.2)


def test_below_threshold_pair_is_not_rescued_when_not_isolated() -> None:
    similarity = np.array(
        [
            [0.2, 0.1],
        ],
        dtype=np.float32,
    )

    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 10, 10],
            ],
            dtype=np.float32,
        ),
        mask_output=_mask_output(
            masks=np.stack([_full_mask()]),
            tracklet_mask_dict={10: 0},
            confidences={10: 0.9},
        ),
        minimum_similarity=0.5,
        enable_isolated_mask_matching=True,
    )

    np.testing.assert_array_equal(
        result.conditioned_similarity,
        similarity,
    )


def test_empty_association_problem_is_supported() -> None:
    result = condition_similarity_with_masks(
        similarity=np.empty((0, 0), dtype=np.float32),
        raw_iou_similarity=np.empty((0, 0), dtype=np.float32),
        tracklet_ids=[],
        detection_boxes=np.empty((0, 4), dtype=np.float32),
        mask_output=None,
        minimum_similarity=0.5,
    )

    assert result.locked_matches == []
    assert result.remaining_track_indices == []
    assert result.remaining_detection_indices == []
    assert result.conditioned_similarity.shape == (0, 0)


def test_empty_tracklet_dimension_is_supported() -> None:
    result = condition_similarity_with_masks(
        similarity=np.empty((0, 3), dtype=np.float32),
        raw_iou_similarity=np.empty((0, 3), dtype=np.float32),
        tracklet_ids=[],
        detection_boxes=np.array(
            [
                [0, 0, 5, 5],
                [5, 5, 10, 10],
                [10, 10, 15, 15],
            ],
            dtype=np.float32,
        ),
        mask_output=None,
        minimum_similarity=0.5,
    )

    assert result.locked_matches == []
    assert result.remaining_track_indices == []
    assert result.remaining_detection_indices == [0, 1, 2]
    assert result.conditioned_similarity.shape == (0, 3)


def test_empty_detection_dimension_is_supported() -> None:
    result = condition_similarity_with_masks(
        similarity=np.empty((2, 0), dtype=np.float32),
        raw_iou_similarity=np.empty((2, 0), dtype=np.float32),
        tracklet_ids=[10, 20],
        detection_boxes=np.empty((0, 4), dtype=np.float32),
        mask_output=None,
        minimum_similarity=0.5,
    )

    assert result.locked_matches == []
    assert result.remaining_track_indices == [0, 1]
    assert result.remaining_detection_indices == []
    assert result.conditioned_similarity.shape == (2, 0)
