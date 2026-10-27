# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Explicit geometry, validation and confidence-free mask-association contracts."""

import numpy as np
import pytest

from trackers.core.masks.base import MaskOutput
from trackers.core.mcbyteplusplus.mask_association import _get_mask_metrics, condition_similarity_with_masks


@pytest.mark.parametrize("confidence", [None, {}, {10: 0.0}])
def test_ambiguous_matching_does_not_require_mask_confidence(
    confidence: dict[int, float] | None,
) -> None:
    """EdgeTAM mask geometry is usable without a confidence field."""
    similarity = np.array([[0.7, 0.6]], dtype=np.float32)
    masks = np.zeros((1, 10, 10), dtype=bool)
    masks[0, :5, :5] = True
    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array([[0, 0, 5, 5], [5, 5, 10, 10]], dtype=np.float32),
        mask_output=MaskOutput(masks, {10: 0}, confidence),
        minimum_similarity=0.5,
    )
    np.testing.assert_allclose(result.conditioned_similarity, [[1.7, 0.6]])


@pytest.mark.parametrize("gate", ["area", "coverage", "fill", "geometry", "clear"])
def test_confidence_free_matching_retains_other_gates(gate: str) -> None:
    """Disabling confidence must not admit missing geometry or poor mask evidence."""
    similarity = np.array([[0.7, 0.6]], dtype=np.float32)
    masks = np.zeros((1, 10, 10), dtype=bool)
    masks[0, :5, :5] = True
    boxes = np.array([[0, 0, 5, 5], [5, 5, 10, 10]], dtype=np.float32)
    if gate == "area":
        masks[:] = False
    elif gate == "coverage":
        masks[0, 5:, 5:] = True
    elif gate == "fill":
        masks[:] = False
        masks[0, 0, 0] = True
        boxes[:] = [0, 0, 10, 10]
    elif gate == "geometry":
        similarity[:] = 0.0
    else:
        similarity[0, 1] = 0.1
    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=boxes,
        mask_output=MaskOutput(masks, {10: 0}),
        minimum_similarity=0.5,
    )
    if gate == "clear":
        assert result.locked_matches == [(0, 0)]
        assert result.conditioned_similarity.shape == (0, 1)
    else:
        np.testing.assert_allclose(result.conditioned_similarity, similarity)


@pytest.mark.parametrize(
    "raw_iou,score,eligible",
    [
        pytest.param(0.0, 0.1, False, id="no-raw-overlap"),
        pytest.param(0.2, 0.1, True, id="low-raw-overlap"),
        pytest.param(0.7, 0.2, False, id="only-fused-score-below-threshold"),
    ],
)
@pytest.mark.parametrize("enable_isolated", [False, True])
def test_isolated_matching_uses_raw_iou_and_requires_opt_in(
    raw_iou: float, score: float, eligible: bool, enable_isolated: bool
) -> None:
    """Raw geometry controls eligibility; a mask bonus is added to the separate assignment score."""
    similarity = np.array([[score]], dtype=np.float32)
    expected = np.array([[score + (1.0 if eligible and enable_isolated else 0.0)]], dtype=np.float32)
    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=np.array([[raw_iou]], dtype=np.float32),
        tracklet_ids=[10],
        detection_boxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
        mask_output=MaskOutput(np.ones((1, 10, 10), dtype=bool), {10: 0}),
        minimum_similarity=0.5,
        enable_isolated_mask_matching=enable_isolated,
    )
    np.testing.assert_allclose(result.conditioned_similarity, expected)


def test_overlapping_channels_each_contribute_mask_evidence() -> None:
    """Shared foreground pixels remain usable independently for both owners."""
    similarity = np.full((2, 2), 0.6)
    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[7, 8],
        detection_boxes=np.tile([0.0, 0.0, 10.0, 10.0], (2, 1)),
        mask_output=MaskOutput(np.ones((2, 10, 10), dtype=bool), {7: 0, 8: 1}),
        minimum_similarity=0.5,
    )
    np.testing.assert_allclose(result.conditioned_similarity, np.full((2, 2), 1.6))


@pytest.mark.parametrize(
    "box,expected",
    [([2, 2, 6, 6], (1.0, 1.0)), ([3, 2, 8, 6], (0.75, 0.6))],
)
def test_coverage_and_fill_use_mask_and_box_area_respectively(box, expected) -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 2:6] = True
    assert _get_mask_metrics(mask, np.array(box, dtype=float)) == pytest.approx(expected)


def test_clear_matches_are_locked_and_remove_all_rows_and_columns() -> None:
    """Even masks pointing at the opposite detections cannot displace clear geometric matches."""
    similarity = np.array([[0.8, 0.1], [0.1, 0.7]])
    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 5:, 5:] = True
    masks[1, :5, :5] = True
    expected_matches = [(0, 0), (1, 1)]
    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10, 20],
        detection_boxes=np.array([[0, 0, 5, 5], [5, 5, 10, 10]]),
        mask_output=MaskOutput(masks, {10: 0, 20: 1}),
        minimum_similarity=0.5,
    )
    assert result.locked_matches == expected_matches
    assert result.remaining_track_indices == result.remaining_detection_indices == []
    assert result.conditioned_similarity.shape == (0, 0)


@pytest.mark.parametrize(
    "mask_output",
    [
        None,
        MaskOutput(None, {}),
        MaskOutput(np.empty((0, 10, 10), dtype=bool), {}),
        MaskOutput(np.ones((1, 10, 10), dtype=bool), {}),
    ],
    ids=["no-output", "no-masks", "empty-masks", "no-owners"],
)
def test_missing_mask_evidence_preserves_reduced_geometry_and_original_indices(mask_output) -> None:
    similarity = np.array([[0.9, 0.1, 0.0], [0.1, 0.7, 0.6], [0.0, 0.6, 0.7]])
    original = similarity.copy()
    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10, 20, 30],
        detection_boxes=np.array([[0, 0, 5, 5], [5, 5, 10, 10], [10, 10, 15, 15]]),
        mask_output=mask_output,
        minimum_similarity=0.5,
        enable_isolated_mask_matching=True,
    )
    assert result.locked_matches == [(0, 0)]
    assert result.remaining_track_indices == result.remaining_detection_indices == [1, 2]
    np.testing.assert_array_equal(result.conditioned_similarity, [[0.7, 0.6], [0.6, 0.7]])
    np.testing.assert_array_equal(similarity, original)


@pytest.mark.parametrize(
    "scores,expected",
    [
        ([[0.7], [0.6]], [[1.7], [0.6]]),
        ([[0.7, 0.6], [0.1, 0.8]], [[1.7, 0.6], [0.1, 1.8]]),
        ([[0.7, 0.6], [0.6, 0.7]], [[1.7, 0.6], [0.6, 1.7]]),
    ],
    ids=["ambiguous-column", "original-matrix-ambiguity", "independent-bonuses"],
)
def test_mask_bonuses_use_original_ambiguity_without_clamping_or_mutating_inputs(scores, expected) -> None:
    """Every qualifying pair receives its own fill bonus, even when scores exceed one."""
    similarity = np.array(scores)
    raw_iou = similarity.copy()
    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, :5, :5] = True
    masks[1, 5:, 5:] = True
    original_masks = masks.copy()
    mapping = {10: 0, 20: 1}
    boxes = np.array([[0, 0, 5, 5], [5, 5, 10, 10]])[: similarity.shape[1]]
    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=raw_iou,
        tracklet_ids=[10, 20],
        detection_boxes=boxes,
        mask_output=MaskOutput(masks, mapping),
        minimum_similarity=0.5,
    )
    assert result.locked_matches == []
    assert result.remaining_track_indices == [0, 1]
    assert result.remaining_detection_indices == list(range(similarity.shape[1]))
    np.testing.assert_allclose(result.conditioned_similarity, expected)
    np.testing.assert_array_equal(similarity, scores)
    np.testing.assert_array_equal(raw_iou, scores)
    np.testing.assert_array_equal(masks, original_masks)
    assert mapping == {10: 0, 20: 1}


@pytest.mark.parametrize("mapping", [{99: 0}, {10: 5}, {10: -1}])
def test_missing_owner_or_invalid_mask_row_cannot_boost_scores(mapping) -> None:
    similarity = np.array([[0.7, 0.6]])
    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=similarity,
        tracklet_ids=[10],
        detection_boxes=np.array([[0, 0, 10, 10], [0, 0, 10, 10]]),
        mask_output=MaskOutput(np.ones((1, 10, 10), dtype=bool), mapping),
        minimum_similarity=0.5,
    )
    np.testing.assert_array_equal(result.conditioned_similarity, similarity)


def test_below_threshold_pair_is_not_rescued_when_not_isolated() -> None:
    """Suppressing one fused score must not hide the second positive geometric candidate."""
    similarity = np.array([[0.1, 0.0]])
    raw_iou = np.array([[0.2, 0.1]])
    expected = similarity.copy()
    result = condition_similarity_with_masks(
        similarity=similarity,
        raw_iou_similarity=raw_iou,
        tracklet_ids=[10],
        detection_boxes=np.array([[0, 0, 10, 10], [0, 0, 10, 10]]),
        mask_output=MaskOutput(np.ones((1, 10, 10), dtype=bool), {10: 0}),
        minimum_similarity=0.5,
        enable_isolated_mask_matching=True,
    )
    np.testing.assert_array_equal(result.conditioned_similarity, expected)


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (2, 0)])
def test_empty_association_dimensions_preserve_remaining_indices(shape) -> None:
    rows, columns = shape
    result = condition_similarity_with_masks(
        similarity=np.empty(shape),
        raw_iou_similarity=np.empty(shape),
        tracklet_ids=list(range(rows)),
        detection_boxes=np.empty((columns, 4)),
        mask_output=None,
        minimum_similarity=0.5,
    )
    assert result.locked_matches == []
    assert result.remaining_track_indices == list(range(rows))
    assert result.remaining_detection_indices == list(range(columns))
    assert result.conditioned_similarity.shape == shape


def _valid_association_kwargs() -> dict:
    return {
        "similarity": np.array([[0.7]]),
        "raw_iou_similarity": np.array([[0.7]]),
        "tracklet_ids": [10],
        "detection_boxes": np.array([[0, 0, 5, 5]]),
        "mask_output": None,
        "minimum_similarity": 0.5,
    }


@pytest.mark.parametrize("name", ["minimum_similarity", "minimum_mask_coverage", "minimum_mask_fill_ratio"])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected_with_its_name(name: str, value: float) -> None:
    kwargs = _valid_association_kwargs()
    kwargs[name] = value
    with pytest.raises(ValueError, match=f"{name} must be between 0 and 1"):
        condition_similarity_with_masks(**kwargs)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"similarity": np.array([[1]], dtype=int)}, "floating-point dtype"),
        ({"similarity": np.array([0.7])}, "two-dimensional matrix"),
        ({"raw_iou_similarity": np.array([[0.7, 0.6]])}, "same shape as similarity"),
        ({"tracklet_ids": [10, 20]}, "Number of tracklet IDs"),
        ({"detection_boxes": np.array([[0, 0, 5]])}, "detection_boxes must have shape"),
    ],
)
def test_invalid_inputs_fail_before_association(overrides, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        condition_similarity_with_masks(**(_valid_association_kwargs() | overrides))
