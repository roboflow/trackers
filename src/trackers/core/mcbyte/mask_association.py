# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackers.core.masks.base import MaskOutput

MINIMUM_MASK_AVERAGE_CONFIDENCE = 0.6
MINIMUM_MASK_COVERAGE = 0.9
MINIMUM_MASK_FILL_RATIO = 0.05


@dataclass(frozen=True)
class MaskConditionedAssociation:
    """Association problem prepared for the remaining Hungarian assignment.

    Attributes:
        conditioned_similarity: Similarity matrix restricted to the remaining
            tracklet and detection indices. Qualifying mask evidence is added to
            selected entries without clamping, so values may exceed ``1.0``.
        locked_matches: Clear threshold-valid matches expressed as
            ``(tracklet_index, detection_index)`` pairs in the original matrix.
        remaining_track_indices: Original tracklet indices represented by the
            rows of ``conditioned_similarity``.
        remaining_detection_indices: Original detection indices represented by
            the columns of ``conditioned_similarity``.
    """

    conditioned_similarity: np.ndarray
    locked_matches: list[tuple[int, int]]
    remaining_track_indices: list[int]
    remaining_detection_indices: list[int]


def _validate_threshold(name: str, value: float) -> None:
    """Validate that a threshold is a fraction in the inclusive range [0, 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _validate_inputs(
    similarity: np.ndarray,
    raw_iou_similarity: np.ndarray,
    tracklet_ids: list[int],
    detection_boxes: np.ndarray,
) -> None:
    """Validate matrix dimensions and their associated metadata."""
    if similarity.ndim != 2:
        raise ValueError(f"similarity must be a two-dimensional matrix. Got shape {similarity.shape}.")

    # Mask evidence is accumulated in place with ``+= mask_fill_ratio``. On an
    # integer matrix numpy truncates the float right-hand side to 0, silently
    # discarding the boost, so a floating-point dtype is required.
    if not np.issubdtype(similarity.dtype, np.floating):
        raise ValueError(
            "similarity must have a floating-point dtype so mask evidence can be "
            f"accumulated without truncation. Got dtype {similarity.dtype}."
        )

    if raw_iou_similarity.shape != similarity.shape:
        raise ValueError(
            "raw_iou_similarity must have the same shape as similarity. "
            f"Got {raw_iou_similarity.shape} and {similarity.shape}."
        )

    num_tracklets, num_detections = similarity.shape

    if len(tracklet_ids) != num_tracklets:
        raise ValueError(
            "Number of tracklet IDs must match the number of similarity rows. "
            f"Got {len(tracklet_ids)} IDs and {num_tracklets} rows."
        )

    if detection_boxes.shape != (num_detections, 4):
        raise ValueError(f"detection_boxes must have shape (num_detections, 4). Got {detection_boxes.shape}.")


def _get_clear_matches(
    similarity: np.ndarray,
    minimum_similarity: float,
) -> list[tuple[int, int]]:
    """Return threshold-valid pairs that are unique in both row and column.

    Eligibility is determined from the untouched similarity matrix. A pair is clear when it is the only threshold-valid
    candidate for both its tracklet row and its detection column.
    """
    eligible = similarity >= minimum_similarity
    row_candidate_counts = eligible.sum(axis=1)
    column_candidate_counts = eligible.sum(axis=0)

    # A matrix entry is True only if:
    # the pair is eligible
    # AND its row has exactly one eligible candidate
    # AND its column has exactly one eligible candidate.
    # None indexing: transform into a vector so that it can be broadcasted and
    # performed logical AND with a matrix.
    clear_rows, clear_columns = np.where(
        eligible & (row_candidate_counts[:, None] == 1) & (column_candidate_counts[None, :] == 1)
    )

    return list(
        zip(
            clear_rows.tolist(),
            clear_columns.tolist(),
        )
    )


def _get_remaining_indices(
    num_tracklets: int,
    num_detections: int,
    locked_matches: list[tuple[int, int]],
) -> tuple[list[int], list[int]]:
    """Return original matrix indices not consumed by locked matches."""
    locked_track_indices = {track_index for track_index, _ in locked_matches}
    locked_detection_indices = {detection_index for _, detection_index in locked_matches}

    remaining_track_indices = [index for index in range(num_tracklets) if index not in locked_track_indices]
    remaining_detection_indices = [index for index in range(num_detections) if index not in locked_detection_indices]

    return remaining_track_indices, remaining_detection_indices


def _get_ambiguous_candidate_matrix(
    similarity: np.ndarray,
    minimum_similarity: float,
) -> np.ndarray:
    """Return eligible pairs belonging to an ambiguous row or column.

    Ambiguity is always computed from the untouched base similarity matrix. A pair is ambiguous when its tracklet has
    multiple eligible detections or its detection has multiple eligible tracklets.
    """
    eligible = similarity >= minimum_similarity
    ambiguous_rows = eligible.sum(axis=1) > 1
    ambiguous_columns = eligible.sum(axis=0) > 1

    # Every eligible cell whose row or column is ambiguous
    return eligible & (ambiguous_rows[:, None] | ambiguous_columns[None, :])


def _get_isolated_candidate_matrix(
    raw_iou_similarity: np.ndarray,
    minimum_similarity: float,
) -> np.ndarray:
    """Return isolated positive-IoU pairs below the normal threshold.

    A pair is isolated when it is the only positive-IoU edge in both its row and its column. Isolation is based
    exclusively on raw IoU geometry, not on score-fused similarity.
    """
    positive_iou = raw_iou_similarity > 0.0
    isolated_rows = positive_iou.sum(axis=1) == 1
    isolated_columns = positive_iou.sum(axis=0) == 1

    below_threshold = raw_iou_similarity < minimum_similarity

    # A pair survives only if:
    # raw IoU is greater than zero
    # AND raw IoU is below the normal threshold
    # AND its row has exactly one positive-IoU edge
    # AND its column has exactly one positive-IoU edge.
    return positive_iou & below_threshold & isolated_rows[:, None] & isolated_columns[None, :]


def _get_mask_metrics(
    mask: np.ndarray,
    detection_xyxy: np.ndarray,
) -> tuple[float, float] | None:
    """Return mask coverage and mask fill ratio for one detection box.

    Mask coverage is the fraction of the full visible mask contained inside the
    detection box. Mask fill ratio is the fraction of detection-box pixels
    occupied by the mask.

    Returns:
        ``(mask_coverage, mask_fill_ratio)`` or ``None`` when the mask is not
        visible or the clipped detection box has no positive area.
    """
    mask_bool = mask.astype(bool, copy=False)
    visible_mask_area = int(mask_bool.sum())

    return _get_mask_metrics_with_visible_area(
        mask_bool=mask_bool,
        detection_xyxy=detection_xyxy,
        visible_mask_area=visible_mask_area,
    )


def _get_mask_metrics_with_visible_area(
    mask_bool: np.ndarray,
    detection_xyxy: np.ndarray,
    visible_mask_area: int,
) -> tuple[float, float] | None:
    """Return mask coverage and fill ratio using a precomputed mask area."""
    if mask_bool.ndim != 2:
        raise ValueError(f"Each mask must have shape (H, W). Got shape {mask_bool.shape}.")

    if visible_mask_area == 0:
        return None

    height, width = mask_bool.shape
    x1, y1, x2, y2 = detection_xyxy

    left = int(np.floor(np.clip(x1, 0, width)))
    top = int(np.floor(np.clip(y1, 0, height)))
    right = int(np.ceil(np.clip(x2, 0, width)))
    bottom = int(np.ceil(np.clip(y2, 0, height)))

    if right <= left or bottom <= top:
        return None

    mask_pixels_inside_box = int(mask_bool[top:bottom, left:right].sum())
    detection_area = (right - left) * (bottom - top)

    mask_coverage = mask_pixels_inside_box / visible_mask_area
    mask_fill_ratio = mask_pixels_inside_box / detection_area

    return mask_coverage, mask_fill_ratio


def _apply_mask_similarity_boosts(
    conditioned_similarity: np.ndarray,
    candidate_matrix: np.ndarray,
    remaining_track_indices: list[int],
    remaining_detection_indices: list[int],
    tracklet_ids: list[int],
    detection_boxes: np.ndarray,
    mask_output: MaskOutput,
    minimum_mask_average_confidence: float,
    minimum_mask_coverage: float,
    minimum_mask_fill_ratio: float,
) -> None:
    """Apply mask evidence to qualifying association-score entries.

    The function examines only pairs marked as ``True`` in ``candidate_matrix``.
    These pairs have already been selected by the caller as either ambiguous
    threshold-valid associations or optional isolated low-IoU associations.

    ``conditioned_similarity`` and ``candidate_matrix`` use local indices from
    the reduced association problem after clear matches have been removed.
    ``remaining_track_indices`` and ``remaining_detection_indices`` map these
    local row and column indices back to the corresponding indices in the
    original full association problem.

    For every candidate pair, the function:

        1. resolves the stable tracklet ID associated with the original row;
        2. finds the corresponding local mask index in ``mask_output``;
        3. verifies that the mask index and average mask confidence are valid;
        4. computes mask coverage and mask fill ratio for the original
           detection box;
        5. adds the mask fill ratio to the local association score when all
           configured mask thresholds are satisfied.

    The score matrix is modified in place. Mask bonuses are intentionally not
    clamped, so conditioned association scores may exceed ``1.0``.

    Args:
        conditioned_similarity: Writable reduced association-score matrix with
            shape ``(R, C)``, where ``R`` and ``C`` are the numbers of remaining
            tracklets and detections after clear matches have been removed.
            Qualifying mask fill ratios are added directly to this array.
        candidate_matrix: Boolean matrix with the same shape as
            ``conditioned_similarity``. A ``True`` entry identifies a reduced
            tracklet-detection pair that is eligible for mask-based
            conditioning.
        remaining_track_indices: Mapping from each local row of the reduced
            matrices to its original tracklet-row index in ``tracklet_ids`` and
            the full association matrix.
        remaining_detection_indices: Mapping from each local column of the
            reduced matrices to its original detection-column index in
            ``detection_boxes`` and the full association matrix.
        tracklet_ids: Stable tracker IDs ordered according to the rows of the
            original full association matrix. A tracklet without a corresponding
            mask is skipped.
        detection_boxes: Detection boxes in ``xyxy`` format, ordered according
            to the columns of the original full association matrix. Expected
            shape is ``(num_detections, 4)``.
        mask_output: Current propagated mask output. Its
            ``tracklet_mask_dict`` maps stable tracklet IDs to local mask-array
            indices, while ``mask_avg_prob_dict`` stores average mask confidence
            keyed by stable tracklet ID. Callers must have already established
            that masks and confidences are present.
        minimum_mask_average_confidence: Minimum average propagated-mask
            confidence required before mask evidence may be used.
        minimum_mask_coverage: Minimum fraction of the complete visible mask
            that must lie inside the detection box.
        minimum_mask_fill_ratio: Minimum fraction of the detection-box area that
            must be occupied by the mask.

    Returns:
        None. ``conditioned_similarity`` is updated in place.
    """
    if mask_output.masks is None or mask_output.mask_avg_prob_dict is None:
        return

    # Local indices in the reduced matrix.
    candidate_rows, candidate_columns = np.where(candidate_matrix)

    # A tracklet may have several candidate detections, but its mask area is shared.
    mask_areas: dict[int, int] = {}
    for local_track_index in np.unique(candidate_rows):
        tracklet_id = tracklet_ids[remaining_track_indices[local_track_index]]
        mask_index = mask_output.tracklet_mask_dict.get(tracklet_id)
        if mask_index is None or not 0 <= mask_index < mask_output.masks.shape[0]:
            continue

        if mask_index not in mask_areas:
            mask_areas[mask_index] = int(mask_output.masks[mask_index].astype(bool, copy=False).sum())

    for local_track_index, local_detection_index in zip(
        candidate_rows,
        candidate_columns,
    ):
        original_track_index = remaining_track_indices[local_track_index]
        original_detection_index = remaining_detection_indices[local_detection_index]

        # Resolve stable tracklet ID
        tracklet_id = tracklet_ids[original_track_index]
        mask_index = mask_output.tracklet_mask_dict.get(tracklet_id)
        if mask_index is None:
            continue

        if not 0 <= mask_index < mask_output.masks.shape[0]:
            continue

        average_confidence = mask_output.mask_avg_prob_dict.get(tracklet_id)
        if average_confidence is None or average_confidence < minimum_mask_average_confidence:
            continue

        metrics = _get_mask_metrics_with_visible_area(
            mask_bool=mask_output.masks[mask_index].astype(bool, copy=False),
            detection_xyxy=detection_boxes[original_detection_index],
            visible_mask_area=mask_areas[mask_index],
        )
        if metrics is None:
            continue

        mask_coverage, mask_fill_ratio = metrics

        if mask_fill_ratio < minimum_mask_fill_ratio:
            continue

        if mask_coverage < minimum_mask_coverage:
            continue

        conditioned_similarity[
            local_track_index,
            local_detection_index,
        ] += mask_fill_ratio


def condition_similarity_with_masks(
    *,
    similarity: np.ndarray,
    raw_iou_similarity: np.ndarray,
    tracklet_ids: list[int],
    detection_boxes: np.ndarray,
    mask_output: MaskOutput | None,
    minimum_similarity: float,
    minimum_mask_average_confidence: float = MINIMUM_MASK_AVERAGE_CONFIDENCE,
    minimum_mask_coverage: float = MINIMUM_MASK_COVERAGE,
    minimum_mask_fill_ratio: float = MINIMUM_MASK_FILL_RATIO,
    enable_isolated_mask_matching: bool = False,
) -> MaskConditionedAssociation:
    """Prepare a similarity matrix for mask-conditioned assignment.

    The function preserves clear threshold-valid IoU decisions by locking pairs
    that are unique in both their row and column. It then removes their rows and
    columns from the remaining assignment problem.

    For the remaining pairs, mask evidence may increase association scores when
    a pair is ambiguous in the original similarity matrix. When
    ``enable_isolated_mask_matching`` is enabled, mask evidence may also rescue
    an isolated positive-similarity-IoU pair that lies below the normal association
    threshold.

    Ambiguity and isolation are computed before any mask updates. Qualifying
    mask-fill ratios are added without clamping, so conditioned scores may
    exceed ``1.0``.

    Args:
        similarity: Base stage-specific association similarity matrix with shape
            ``(num_tracklets, num_detections)``. This may be raw IoU or
            score-fused IoU depending on the association stage.
        raw_iou_similarity: Raw IoU matrix with the same shape. It is used only
            to identify optional isolated geometric pairs.
        tracklet_ids: Stable tracker IDs corresponding to similarity rows.
            Tracklets without a usable stable ID may use a negative value; such
            IDs will not resolve to a mask.
        detection_boxes: Detection boxes in ``xyxy`` format with shape
            ``(num_detections, 4)``.
        mask_output: Current propagated masks and tracklet mappings.
        minimum_similarity: Normal minimum similarity required by the current
            association stage.
        minimum_mask_average_confidence: Minimum average propagated-mask
            confidence required to use mask evidence.
        minimum_mask_coverage: Minimum fraction of the visible mask that must
            lie inside the detection box (mc).
        minimum_mask_fill_ratio: Minimum fraction of the detection box that must
            be occupied by the mask (mf).
        enable_isolated_mask_matching: Whether mask evidence may also condition
            isolated positive-IoU pairs below ``minimum_similarity``.

    Returns:
        Prepared reduced association problem, locked clear matches, and the
        original indices represented by the reduced matrix.

    Examples:
        >>> import numpy as np
        >>> from trackers.core.masks.base import MaskOutput
        >>> similarity = np.array([[0.7, 0.6]], dtype=np.float32)
        >>> masks = np.zeros((1, 10, 10), dtype=bool)
        >>> masks[0, 0:5, 0:5] = True
        >>> mask_output = MaskOutput(
        ...     masks=masks,
        ...     tracklet_mask_dict={10: 0},
        ...     mask_avg_prob_dict={10: 0.9},
        ... )
        >>> result = condition_similarity_with_masks(
        ...     similarity=similarity,
        ...     raw_iou_similarity=similarity,
        ...     tracklet_ids=[10],
        ...     detection_boxes=np.array(
        ...         [[0, 0, 5, 5], [5, 5, 10, 10]], dtype=np.float32
        ...     ),
        ...     mask_output=mask_output,
        ...     minimum_similarity=0.5,
        ... )
        >>> result.conditioned_similarity
        array([[1.7, 0.6]], dtype=float32)
    """
    _validate_threshold("minimum_similarity", minimum_similarity)
    _validate_threshold(
        "minimum_mask_average_confidence",
        minimum_mask_average_confidence,
    )
    _validate_threshold("minimum_mask_coverage", minimum_mask_coverage)
    _validate_threshold(
        "minimum_mask_fill_ratio",
        minimum_mask_fill_ratio,
    )
    _validate_inputs(
        similarity=similarity,
        raw_iou_similarity=raw_iou_similarity,
        tracklet_ids=tracklet_ids,
        detection_boxes=detection_boxes,
    )

    # All ambiguity and locking decisions are based on an untouched snapshot of the
    # initial matrix.
    base_similarity = similarity.copy()

    locked_matches = _get_clear_matches(
        similarity=base_similarity,
        minimum_similarity=minimum_similarity,
    )

    remaining_track_indices, remaining_detection_indices = _get_remaining_indices(
        num_tracklets=base_similarity.shape[0],
        num_detections=base_similarity.shape[1],
        locked_matches=locked_matches,
    )

    # copy(): independent working copy that will be modified.
    # Not a meaningful memory concern.
    reduced_similarity = base_similarity[
        np.ix_(
            remaining_track_indices,
            remaining_detection_indices,
        )
    ].copy()

    if (
        mask_output is None
        or mask_output.masks is None
        or mask_output.masks.shape[0] == 0
        or not mask_output.tracklet_mask_dict
        or not mask_output.mask_avg_prob_dict
    ):
        # No tracklet can receive mask evidence, so the reduced problem is final.
        return MaskConditionedAssociation(
            conditioned_similarity=reduced_similarity,
            locked_matches=locked_matches,
            remaining_track_indices=remaining_track_indices,
            remaining_detection_indices=remaining_detection_indices,
        )

    # Ambiguity is a property of the original association situation before any
    # modifications, hence computed from base_similarity.
    ambiguous_candidates = _get_ambiguous_candidate_matrix(
        similarity=base_similarity,
        minimum_similarity=minimum_similarity,
    )

    candidate_matrix = ambiguous_candidates[
        np.ix_(
            remaining_track_indices,
            remaining_detection_indices,
        )
    ]

    if enable_isolated_mask_matching:
        isolated_candidates = _get_isolated_candidate_matrix(
            raw_iou_similarity=raw_iou_similarity,
            minimum_similarity=minimum_similarity,
        )
        candidate_matrix = (
            candidate_matrix
            | isolated_candidates[
                np.ix_(
                    remaining_track_indices,
                    remaining_detection_indices,
                )
            ]
        )

    _apply_mask_similarity_boosts(
        conditioned_similarity=reduced_similarity,
        candidate_matrix=candidate_matrix,
        remaining_track_indices=remaining_track_indices,
        remaining_detection_indices=remaining_detection_indices,
        tracklet_ids=tracklet_ids,
        detection_boxes=detection_boxes,
        mask_output=mask_output,
        minimum_mask_average_confidence=minimum_mask_average_confidence,
        minimum_mask_coverage=minimum_mask_coverage,
        minimum_mask_fill_ratio=minimum_mask_fill_ratio,
    )

    return MaskConditionedAssociation(
        conditioned_similarity=reduced_similarity,
        locked_matches=locked_matches,
        remaining_track_indices=remaining_track_indices,
        remaining_detection_indices=remaining_detection_indices,
    )
