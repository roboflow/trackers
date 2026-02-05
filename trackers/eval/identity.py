# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Adapted from TrackEval (https://github.com/JonathonLuiten/TrackEval)
# Copyright (c) Jonathon Luiten. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Reference: trackeval/metrics/identity.py:32-89 (eval_sequence method)
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_identity_metrics(
    gt_ids: list[np.ndarray],
    tracker_ids: list[np.ndarray],
    similarity_scores: list[np.ndarray],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute Identity metrics for multi-object tracking evaluation.

    Identity metrics measure global ID consistency by finding the optimal
    one-to-one assignment between ground truth IDs and tracker IDs that
    maximizes the number of correctly identified detections (IDTP).

    Args:
        gt_ids: List of ground truth ID arrays, one per frame. Each array has
            shape `(num_gt_t,)` containing integer IDs for GTs in that frame.
        tracker_ids: List of tracker ID arrays, one per frame. Each array has
            shape `(num_tracker_t,)` containing integer IDs for detections.
        similarity_scores: List of similarity matrices, one per frame. Each
            matrix has shape `(num_gt_t, num_tracker_t)` with IoU or similar
            similarity scores.
        threshold: Similarity threshold for a valid match. Defaults to 0.5.

    Returns:
        Dictionary containing Identity metrics:
            - IDF1: ID F1 score (harmonic mean of IDR and IDP)
            - IDR: ID Recall (IDTP / (IDTP + IDFN))
            - IDP: ID Precision (IDTP / (IDTP + IDFP))
            - IDTP: ID True Positives
            - IDFN: ID False Negatives
            - IDFP: ID False Positives

    Examples:
        ```python
        >>> import numpy as np
        >>> from trackers.eval.identity import compute_identity_metrics

        >>> gt_ids = [
        ...     np.array([0, 1]),
        ...     np.array([0, 1]),
        ...     np.array([0, 1]),
        ... ]
        >>> tracker_ids = [
        ...     np.array([10, 20]),
        ...     np.array([10, 30]),
        ...     np.array([10, 30]),
        ... ]
        >>> similarity_scores = [
        ...     np.array([[0.9, 0.1], [0.1, 0.8]]),
        ...     np.array([[0.85, 0.1], [0.1, 0.75]]),
        ...     np.array([[0.8, 0.1], [0.1, 0.7]]),
        ... ]
        >>> result = compute_identity_metrics(gt_ids, tracker_ids, similarity_scores)
        >>> result["IDF1"]  # doctest: +ELLIPSIS
        0.833...
        >>> result["IDTP"]
        5
        >>>
        ```
    """
    # Count total detections
    num_gt_dets = sum(len(ids) for ids in gt_ids)
    num_tracker_dets = sum(len(ids) for ids in tracker_ids)

    # Handle empty sequences (ref: identity.py:40-45)
    if num_tracker_dets == 0:
        return {
            "IDF1": 0.0,
            "IDR": 0.0,
            "IDP": 0.0,
            "IDTP": 0,
            "IDFN": num_gt_dets,
            "IDFP": 0,
        }
    if num_gt_dets == 0:
        return {
            "IDF1": 0.0,
            "IDR": 0.0,
            "IDP": 0.0,
            "IDTP": 0,
            "IDFN": 0,
            "IDFP": num_tracker_dets,
        }

    # Get unique IDs
    all_gt_ids = np.concatenate(gt_ids) if gt_ids else np.array([])
    all_tracker_ids = np.concatenate(tracker_ids) if tracker_ids else np.array([])
    unique_gt_ids = np.unique(all_gt_ids)
    unique_tracker_ids = np.unique(all_tracker_ids)
    num_gt_ids = len(unique_gt_ids)
    num_tracker_ids = len(unique_tracker_ids)

    # Create ID mappings for array indexing
    gt_id_to_idx = {int(id_): idx for idx, id_ in enumerate(unique_gt_ids)}
    tracker_id_to_idx = {int(id_): idx for idx, id_ in enumerate(unique_tracker_ids)}

    # Variables for global association (ref: identity.py:48-50)
    potential_matches_count = np.zeros((num_gt_ids, num_tracker_ids))
    gt_id_count = np.zeros(num_gt_ids)
    tracker_id_count = np.zeros(num_tracker_ids)

    # First pass: accumulate global track information (ref: identity.py:53-61)
    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(gt_ids, tracker_ids)):
        if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
            # Still count IDs even if no matches possible
            if len(gt_ids_t) > 0:
                gt_indices = np.array([gt_id_to_idx[int(id_)] for id_ in gt_ids_t])
                gt_id_count[gt_indices] += 1
            if len(tracker_ids_t) > 0:
                tr_indices = np.array(
                    [tracker_id_to_idx[int(id_)] for id_ in tracker_ids_t]
                )
                tracker_id_count[tr_indices] += 1
            continue

        gt_indices = np.array([gt_id_to_idx[int(id_)] for id_ in gt_ids_t])
        tr_indices = np.array([tracker_id_to_idx[int(id_)] for id_ in tracker_ids_t])

        similarity = similarity_scores[t]

        # Count potential matches (similarity >= threshold) (ref: identity.py:55-57)
        matches_mask = similarity >= threshold
        match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
        potential_matches_count[
            gt_indices[match_idx_gt], tr_indices[match_idx_tracker]
        ] += 1

        # Count detections per ID (ref: identity.py:60-61)
        gt_id_count[gt_indices] += 1
        tracker_id_count[tr_indices] += 1

    # Build cost matrix for Hungarian algorithm (ref: identity.py:64-77)
    # Matrix is (num_gt_ids + num_tracker_ids) x (num_gt_ids + num_tracker_ids)
    # to allow for unmatched IDs on both sides
    matrix_size = num_gt_ids + num_tracker_ids
    fp_mat = np.zeros((matrix_size, matrix_size))
    fn_mat = np.zeros((matrix_size, matrix_size))

    # Set high cost for invalid assignments (ref: identity.py:68-69)
    fp_mat[num_gt_ids:, :num_tracker_ids] = 1e10
    fn_mat[:num_gt_ids, num_tracker_ids:] = 1e10

    # Fill in costs for GT IDs (ref: identity.py:70-72)
    for gt_idx in range(num_gt_ids):
        fn_mat[gt_idx, :num_tracker_ids] = gt_id_count[gt_idx]
        fn_mat[gt_idx, num_tracker_ids + gt_idx] = gt_id_count[gt_idx]

    # Fill in costs for tracker IDs (ref: identity.py:73-75)
    for tr_idx in range(num_tracker_ids):
        fp_mat[:num_gt_ids, tr_idx] = tracker_id_count[tr_idx]
        fp_mat[tr_idx + num_gt_ids, tr_idx] = tracker_id_count[tr_idx]

    # Subtract potential matches (ref: identity.py:76-77)
    fn_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count
    fp_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count

    # Hungarian algorithm for optimal assignment (ref: identity.py:80)
    match_rows, match_cols = linear_sum_assignment(fn_mat + fp_mat)

    # Compute IDTP, IDFN, IDFP (ref: identity.py:83-85)
    idfn = int(fn_mat[match_rows, match_cols].sum())
    idfp = int(fp_mat[match_rows, match_cols].sum())
    idtp = int(gt_id_count.sum()) - idfn

    # Compute final scores (ref: identity.py:132-134)
    idr = idtp / max(1.0, idtp + idfn)
    idp = idtp / max(1.0, idtp + idfp)
    idf1 = idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn)

    return {
        "IDF1": float(idf1),
        "IDR": float(idr),
        "IDP": float(idp),
        "IDTP": idtp,
        "IDFN": idfn,
        "IDFP": idfp,
    }


def aggregate_identity_metrics(
    sequence_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate Identity metrics across multiple sequences.

    Sums IDTP, IDFN, IDFP across sequences, then recomputes IDF1, IDR, IDP
    from the totals.

    Args:
        sequence_metrics: List of per-sequence Identity metric dictionaries.

    Returns:
        Aggregated Identity metrics dictionary.
    """
    if not sequence_metrics:
        return {
            "IDF1": 0.0,
            "IDR": 0.0,
            "IDP": 0.0,
            "IDTP": 0,
            "IDFN": 0,
            "IDFP": 0,
        }

    # Sum integer fields (ref: identity.py:122-123)
    idtp = sum(m["IDTP"] for m in sequence_metrics)
    idfn = sum(m["IDFN"] for m in sequence_metrics)
    idfp = sum(m["IDFP"] for m in sequence_metrics)

    # Recompute final scores (ref: identity.py:124, 132-134)
    idr = idtp / max(1.0, idtp + idfn)
    idp = idtp / max(1.0, idtp + idfp)
    idf1 = idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn)

    return {
        "IDF1": float(idf1),
        "IDR": float(idr),
        "IDP": float(idp),
        "IDTP": idtp,
        "IDFN": idfn,
        "IDFP": idfp,
    }
