# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Adapted from TrackEval (https://github.com/JonathonLuiten/TrackEval)
# Copyright (c) Jonathon Luiten. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Reference: trackeval/metrics/hota.py:25-117 (eval_sequence method)
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

# Epsilon for floating point comparisons - must match TrackEval exactly
EPS = np.finfo("float").eps

# Alpha thresholds for HOTA evaluation (IoU thresholds)
# TrackEval uses np.arange(0.05, 0.99, 0.05) which gives 19 values
ALPHA_THRESHOLDS = np.arange(0.05, 0.99, 0.05)


def compute_hota_metrics(
    gt_ids: list[np.ndarray],
    tracker_ids: list[np.ndarray],
    similarity_scores: list[np.ndarray],
) -> dict[str, Any]:
    """Compute HOTA metrics for multi-object tracking evaluation.

    HOTA (Higher Order Tracking Accuracy) evaluates both detection and
    association quality across multiple IoU thresholds. The final HOTA
    score is the geometric mean of detection accuracy (DetA) and
    association accuracy (AssA), averaged over all thresholds.

    Args:
        gt_ids: List of ground truth ID arrays, one per frame. Each array has
            shape `(num_gt_t,)` containing integer IDs for GTs in that frame.
        tracker_ids: List of tracker ID arrays, one per frame. Each array has
            shape `(num_tracker_t,)` containing integer IDs for detections.
        similarity_scores: List of similarity matrices, one per frame. Each
            matrix has shape `(num_gt_t, num_tracker_t)` with IoU or similar
            similarity scores.

    Returns:
        Dictionary containing HOTA metrics:
        - `HOTA`: Higher Order Tracking Accuracy as `float` in range `[0, 1]`.
            Geometric mean of DetA and AssA, averaged over alpha thresholds.
        - `DetA`: Detection Accuracy as `float` in range `[0, 1]`.
        - `AssA`: Association Accuracy as `float` in range `[0, 1]`.
        - `DetRe`: Detection Recall as `float` in range `[0, 1]`.
        - `DetPr`: Detection Precision as `float` in range `[0, 1]`.
        - `AssRe`: Association Recall as `float` in range `[0, 1]`.
        - `AssPr`: Association Precision as `float` in range `[0, 1]`.
        - `LocA`: Localization Accuracy as `float` in range `[0, 1]`.
        - `OWTA`: Open World Tracking Accuracy as `float` in range `[0, 1]`.
        - `HOTA_TP`: True positives summed over all alphas as `int`.
        - `HOTA_FN`: False negatives summed over all alphas as `int`.
        - `HOTA_FP`: False positives summed over all alphas as `int`.
        - Arrays for per-alpha values (for aggregation):
            `HOTA_TP_array`, `HOTA_FN_array`, `HOTA_FP_array`,
            `AssA_array`, `AssRe_array`, `AssPr_array`, `LocA_array`.

    Examples:
        ```python
        import numpy as np
        from trackers.eval import compute_hota_metrics

        gt_ids = [np.array([0, 1]), np.array([0, 1])]
        tracker_ids = [np.array([10, 20]), np.array([10, 20])]
        similarity_scores = [
            np.array([[0.9, 0.1], [0.1, 0.8]]),
            np.array([[0.85, 0.1], [0.1, 0.75]]),
        ]
        result = compute_hota_metrics(gt_ids, tracker_ids, similarity_scores)
        result["HOTA"]
        # ~0.85
        ```
    """
    num_alphas = len(ALPHA_THRESHOLDS)

    # Count total detections
    num_gt_dets = sum(len(ids) for ids in gt_ids)
    num_tracker_dets = sum(len(ids) for ids in tracker_ids)

    # Initialize result arrays
    hota_tp: np.ndarray = np.zeros(num_alphas, dtype=np.float64)
    hota_fn: np.ndarray = np.zeros(num_alphas, dtype=np.float64)
    hota_fp: np.ndarray = np.zeros(num_alphas, dtype=np.float64)
    loc_a: np.ndarray = np.zeros(num_alphas, dtype=np.float64)
    ass_a: np.ndarray = np.zeros(num_alphas, dtype=np.float64)
    ass_re: np.ndarray = np.zeros(num_alphas, dtype=np.float64)
    ass_pr: np.ndarray = np.zeros(num_alphas, dtype=np.float64)

    # Handle edge case: no tracker detections
    if num_tracker_dets == 0:
        hota_fn[:] = num_gt_dets
        loc_a[:] = 1.0
        return _build_result(
            hota_tp, hota_fn, hota_fp, ass_a, ass_re, ass_pr, loc_a
        )

    # Handle edge case: no GT detections
    if num_gt_dets == 0:
        hota_fp[:] = num_tracker_dets
        loc_a[:] = 1.0
        return _build_result(
            hota_tp, hota_fn, hota_fp, ass_a, ass_re, ass_pr, loc_a
        )

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

    # Variables for global association (ref: hota.py:48-50)
    potential_matches_count: np.ndarray = np.zeros(
        (num_gt_ids, num_tracker_ids), dtype=np.float64
    )
    gt_id_count: np.ndarray = np.zeros((num_gt_ids, 1), dtype=np.float64)
    tracker_id_count: np.ndarray = np.zeros((1, num_tracker_ids), dtype=np.float64)

    # First pass: accumulate global track information (ref: hota.py:53-65)
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
                tracker_id_count[0, tr_indices] += 1
            continue

        gt_indices = np.array([gt_id_to_idx[int(id_)] for id_ in gt_ids_t])
        tr_indices = np.array([tracker_id_to_idx[int(id_)] for id_ in tracker_ids_t])

        similarity = similarity_scores[t]

        # Compute similarity IoU for potential matches (ref: hota.py:57-60)
        sim_iou_denom = (
            similarity.sum(0)[np.newaxis, :]
            + similarity.sum(1)[:, np.newaxis]
            - similarity
        )
        sim_iou = np.zeros_like(similarity)
        sim_iou_mask = sim_iou_denom > 0 + EPS
        sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]

        # Accumulate potential matches (ref: hota.py:61)
        potential_matches_count[
            gt_indices[:, np.newaxis], tr_indices[np.newaxis, :]
        ] += sim_iou

        # Count detections per ID (ref: hota.py:64-65)
        gt_id_count[gt_indices] += 1
        tracker_id_count[0, tr_indices] += 1

    # Calculate global alignment score (ref: hota.py:68)
    global_alignment_score = potential_matches_count / (
        gt_id_count + tracker_id_count - potential_matches_count
    )

    # Per-alpha match counts for association metrics
    matches_counts: list[np.ndarray] = [
        np.zeros((num_gt_ids, num_tracker_ids), dtype=np.float64)
        for _ in range(num_alphas)
    ]

    # Second pass: calculate scores for each timestep (ref: hota.py:72-101)
    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(gt_ids, tracker_ids)):
        # Handle empty frames (ref: hota.py:74-81)
        if len(gt_ids_t) == 0:
            hota_fp += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            hota_fn += len(gt_ids_t)
            continue

        gt_indices = np.array([gt_id_to_idx[int(id_)] for id_ in gt_ids_t])
        tr_indices = np.array([tracker_id_to_idx[int(id_)] for id_ in tracker_ids_t])

        similarity = similarity_scores[t]

        # Build score matrix for Hungarian matching (ref: hota.py:84-85)
        score_mat = (
            global_alignment_score[gt_indices[:, np.newaxis], tr_indices[np.newaxis, :]]
            * similarity
        )

        # Hungarian algorithm for optimal assignment (ref: hota.py:88)
        match_rows, match_cols = linear_sum_assignment(-score_mat)

        # Calculate statistics for each alpha threshold (ref: hota.py:91-101)
        for a, alpha in enumerate(ALPHA_THRESHOLDS):
            actually_matched_mask = similarity[match_rows, match_cols] >= alpha - EPS
            alpha_match_rows = match_rows[actually_matched_mask]
            alpha_match_cols = match_cols[actually_matched_mask]
            num_matches = len(alpha_match_rows)

            hota_tp[a] += num_matches
            hota_fn[a] += len(gt_ids_t) - num_matches
            hota_fp[a] += len(tracker_ids_t) - num_matches

            if num_matches > 0:
                loc_a[a] += np.sum(similarity[alpha_match_rows, alpha_match_cols])
                matches_counts[a][
                    gt_indices[alpha_match_rows], tr_indices[alpha_match_cols]
                ] += 1

    # Calculate association scores for each alpha (ref: hota.py:105-112)
    for a in range(num_alphas):
        matches_count = matches_counts[a]

        # AssA: association accuracy (ref: hota.py:107-108)
        ass_a_mat = matches_count / np.maximum(
            1, gt_id_count + tracker_id_count - matches_count
        )
        ass_a[a] = np.sum(matches_count * ass_a_mat) / np.maximum(1, hota_tp[a])

        # AssRe: association recall (ref: hota.py:109-110)
        ass_re_mat = matches_count / np.maximum(1, gt_id_count)
        ass_re[a] = np.sum(matches_count * ass_re_mat) / np.maximum(1, hota_tp[a])

        # AssPr: association precision (ref: hota.py:111-112)
        ass_pr_mat = matches_count / np.maximum(1, tracker_id_count)
        ass_pr[a] = np.sum(matches_count * ass_pr_mat) / np.maximum(1, hota_tp[a])

    # Finalize LocA (ref: hota.py:115)
    loc_a = np.maximum(1e-10, loc_a) / np.maximum(1e-10, hota_tp)

    return _build_result(hota_tp, hota_fn, hota_fp, ass_a, ass_re, ass_pr, loc_a)


def _build_result(
    hota_tp: np.ndarray,
    hota_fn: np.ndarray,
    hota_fp: np.ndarray,
    ass_a: np.ndarray,
    ass_re: np.ndarray,
    ass_pr: np.ndarray,
    loc_a: np.ndarray,
) -> dict[str, Any]:
    """Build result dictionary from per-alpha arrays.

    Computes final metrics (DetA, DetRe, DetPr, HOTA, OWTA) and averages
    over alpha thresholds for summary values.

    Args:
        hota_tp: True positives per alpha.
        hota_fn: False negatives per alpha.
        hota_fp: False positives per alpha.
        ass_a: Association accuracy per alpha.
        ass_re: Association recall per alpha.
        ass_pr: Association precision per alpha.
        loc_a: Localization accuracy per alpha.

    Returns:
        Dictionary with all HOTA metrics.
    """
    # Compute detection metrics (ref: hota.py:170-172)
    det_re = hota_tp / np.maximum(1, hota_tp + hota_fn)
    det_pr = hota_tp / np.maximum(1, hota_tp + hota_fp)
    det_a = hota_tp / np.maximum(1, hota_tp + hota_fn + hota_fp)

    # Compute HOTA and OWTA (ref: hota.py:173-174)
    hota = np.sqrt(det_a * ass_a)
    owta = np.sqrt(det_re * ass_a)

    # Average over alpha thresholds for summary values
    return {
        # Summary metrics (averaged over alphas)
        "HOTA": float(np.mean(hota)),
        "DetA": float(np.mean(det_a)),
        "AssA": float(np.mean(ass_a)),
        "DetRe": float(np.mean(det_re)),
        "DetPr": float(np.mean(det_pr)),
        "AssRe": float(np.mean(ass_re)),
        "AssPr": float(np.mean(ass_pr)),
        "LocA": float(np.mean(loc_a)),
        "OWTA": float(np.mean(owta)),
        # Integer totals (summed over alphas, for display)
        "HOTA_TP": int(np.sum(hota_tp)),
        "HOTA_FN": int(np.sum(hota_fn)),
        "HOTA_FP": int(np.sum(hota_fp)),
        # Per-alpha arrays (for aggregation)
        "HOTA_TP_array": hota_tp.copy(),
        "HOTA_FN_array": hota_fn.copy(),
        "HOTA_FP_array": hota_fp.copy(),
        "AssA_array": ass_a.copy(),
        "AssRe_array": ass_re.copy(),
        "AssPr_array": ass_pr.copy(),
        "LocA_array": loc_a.copy(),
    }


def aggregate_hota_metrics(
    sequence_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate HOTA metrics across multiple sequences.

    Uses weighted averaging by HOTA_TP for association metrics,
    then recomputes detection and HOTA metrics from totals.

    Args:
        sequence_metrics: List of HOTA metric dictionaries from individual
            sequences.

    Returns:
        Aggregated HOTA metrics dictionary.
    """
    if not sequence_metrics:
        return _build_result(
            np.zeros(len(ALPHA_THRESHOLDS)),
            np.zeros(len(ALPHA_THRESHOLDS)),
            np.zeros(len(ALPHA_THRESHOLDS)),
            np.zeros(len(ALPHA_THRESHOLDS)),
            np.zeros(len(ALPHA_THRESHOLDS)),
            np.zeros(len(ALPHA_THRESHOLDS)),
            np.ones(len(ALPHA_THRESHOLDS)),
        )

    # Sum integer arrays (ref: hota.py:122-123)
    hota_tp = np.sum([m["HOTA_TP_array"] for m in sequence_metrics], axis=0)
    hota_fn = np.sum([m["HOTA_FN_array"] for m in sequence_metrics], axis=0)
    hota_fp = np.sum([m["HOTA_FP_array"] for m in sequence_metrics], axis=0)

    # Weighted average for association metrics (ref: hota.py:124-125)
    def weighted_avg(field: str) -> np.ndarray:
        weighted_sum = np.sum(
            [m[field] * m["HOTA_TP_array"] for m in sequence_metrics], axis=0
        )
        return weighted_sum / np.maximum(1e-10, hota_tp)

    ass_a = weighted_avg("AssA_array")
    ass_re = weighted_avg("AssRe_array")
    ass_pr = weighted_avg("AssPr_array")

    # Weighted average for LocA (ref: hota.py:126-127)
    loc_a_weighted = np.sum(
        [m["LocA_array"] * m["HOTA_TP_array"] for m in sequence_metrics], axis=0
    )
    loc_a = np.maximum(1e-10, loc_a_weighted) / np.maximum(1e-10, hota_tp)

    return _build_result(hota_tp, hota_fn, hota_fp, ass_a, ass_re, ass_pr, loc_a)
