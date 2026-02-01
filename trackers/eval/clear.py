# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Adapted from TrackEval (https://github.com/JonathonLuiten/TrackEval)
# Copyright (c) Jonathon Luiten. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Reference: trackeval/metrics/clear.py:38-129 (eval_sequence method)
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

# Epsilon for floating point comparisons - must match TrackEval exactly
EPS = np.finfo("float").eps


def compute_clear_metrics(
    gt_ids: list[np.ndarray],
    tracker_ids: list[np.ndarray],
    similarity_scores: list[np.ndarray],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute CLEAR metrics for multi-object tracking evaluation.
    Calculate standard CLEAR metrics including MOTA, MOTP, ID switches, and
    track quality metrics (MT/PT/ML) from per-frame ground truth and tracker
    associations.

    Args:
        gt_ids: List of ground truth ID arrays, one per frame. Each array has
            shape `(num_gt_t,)` containing integer IDs for GTs in that frame.
        tracker_ids: List of tracker ID arrays, one per frame. Each array has
            shape `(num_tracker_t,)` containing integer IDs for detections.
        similarity_scores: List of similarity matrices, one per frame. Each
            matrix has shape `(num_gt_t, num_tracker_t)` with IoU or similar
            similarity scores.
        threshold: Minimum similarity score for valid match. Defaults to 0.5.

    Returns:
        Dictionary containing CLEAR metrics:
        - `CLR_TP`: True positives (correct matches) as `int`.
        - `CLR_FN`: False negatives (missed GTs) as `int`.
        - `CLR_FP`: False positives (extra detections) as `int`.
        - `IDSW`: ID switches as `int`.
        - `MOTA`: Multiple Object Tracking Accuracy as `float` in range
            `(-inf, 1]`. Computed as `(TP - FP - IDSW) / (TP + FN)`.
        - `MOTP`: Multiple Object Tracking Precision as `float` in range
            `[0, 1]`. Average similarity of matched pairs.
        - `MT`: Mostly Tracked count (>80% tracked) as `int`.
        - `PT`: Partially Tracked count (20-80% tracked) as `int`.
        - `ML`: Mostly Lost count (<20% tracked) as `int`.
        - `MTR`: Mostly Tracked ratio as `float`.
        - `PTR`: Partially Tracked ratio as `float`.
        - `MLR`: Mostly Lost ratio as `float`.
        - `Frag`: Fragmentations (track interruptions) as `int`.

    Examples:
        ```python
        import numpy as np
        from trackers.eval import compute_clear_metrics

        gt_ids = [
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0, 1]),
        ]
        tracker_ids = [
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0, 1]),
        ]
        similarity_scores = [
            np.array([[0.9, 0.1], [0.1, 0.9]]),
            np.array([[0.8, 0.2], [0.2, 0.85]]),
            np.array([[0.85, 0.15], [0.1, 0.9]]),
        ]
        result = compute_clear_metrics(
            gt_ids, tracker_ids, similarity_scores, threshold=0.5
        )
        result["MOTA"]
        # 1.0
        result["MT"]
        # 2
        ```

        ```python
        import numpy as np
        from trackers.eval import compute_clear_metrics

        gt_ids = [
            np.array([0, 1, 2]),
            np.array([0, 1, 2]),
        ]
        tracker_ids = [
            np.array([10, 11]),
            np.array([10, 12]),
        ]
        similarity_scores = [
            np.array([[0.8, 0.1], [0.1, 0.7], [0.0, 0.0]]),
            np.array([[0.75, 0.1], [0.1, 0.0], [0.0, 0.65]]),
        ]
        result = compute_clear_metrics(
            gt_ids, tracker_ids, similarity_scores, threshold=0.5
        )
        result["CLR_FN"]
        # 2
        result["IDSW"]
        # 1
        ```
    """
    # Count total detections
    num_gt_dets = sum(len(ids) for ids in gt_ids)
    num_tracker_dets = sum(len(ids) for ids in tracker_ids)

    # Get unique GT IDs across all frames (sorted for searchsorted)
    all_gt_ids = np.concatenate(gt_ids) if gt_ids and num_gt_dets > 0 else np.array([])
    unique_gt_ids = np.unique(all_gt_ids)
    num_gt_ids = len(unique_gt_ids)

    # Handle edge case: no tracker detections
    if num_tracker_dets == 0:
        num_gt_ids_total = num_gt_ids
        return {
            "CLR_TP": 0,
            "CLR_FN": num_gt_dets,
            "CLR_FP": 0,
            "IDSW": 0,
            "MOTA": 0.0 if num_gt_dets == 0 else -num_gt_dets / max(1.0, num_gt_dets),
            "MOTP": 0.0,
            "MT": 0,
            "PT": 0,
            "ML": num_gt_ids_total,
            "MTR": 0.0,
            "PTR": 0.0,
            "MLR": 1.0 if num_gt_ids_total > 0 else 0.0,
            "Frag": 0,
        }

    # Handle edge case: no GT detections
    if num_gt_dets == 0:
        return {
            "CLR_TP": 0,
            "CLR_FN": 0,
            "CLR_FP": num_tracker_dets,
            "IDSW": 0,
            "MOTA": 0.0,
            "MOTP": 0.0,
            "MT": 0,
            "PT": 0,
            "ML": 0,
            "MTR": 0.0,
            "PTR": 0.0,
            "MLR": 0.0,
            "Frag": 0,
        }

    # Initialize counters
    clr_tp = 0
    clr_fn = 0
    clr_fp = 0
    idsw = 0
    motp_sum = 0.0

    # Per-GT tracking arrays
    gt_id_count = np.zeros(num_gt_ids)
    gt_matched_count = np.zeros(num_gt_ids)
    gt_frag_count = np.zeros(num_gt_ids)

    # For IDSW tracking: prev_tracker_id tracks last-ever match per GT,
    # prev_timestep_tracker_id tracks match from previous frame only
    prev_tracker_id = np.full(num_gt_ids, np.nan)
    prev_timestep_tracker_id = np.full(num_gt_ids, np.nan)

    # Process each timestep
    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(gt_ids, tracker_ids)):
        # Map GT IDs to indices using searchsorted (vectorized)
        gt_indices_t = np.atleast_1d(np.searchsorted(unique_gt_ids, gt_ids_t))

        # Handle empty frames
        if len(gt_ids_t) == 0:
            clr_fp += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            clr_fn += len(gt_ids_t)
            gt_id_count[gt_indices_t] += 1
            continue

        # Get similarity matrix for this timestep
        similarity = similarity_scores[t]

        # Build score matrix with IDSW prioritization (ref: clear.py:78-82)
        # Add 1000 bonus for continuing previous association to minimize ID switches
        score_mat = (
            tracker_ids_t[np.newaxis, :]
            == prev_timestep_tracker_id[gt_indices_t[:, np.newaxis]]
        )
        score_mat = 1000 * score_mat + similarity
        score_mat[similarity < threshold - EPS] = 0

        # Hungarian algorithm for optimal assignment
        match_rows, match_cols = linear_sum_assignment(-score_mat)
        match_rows = np.asarray(match_rows)
        match_cols = np.asarray(match_cols)

        # Filter matches that are actually valid (score > 0)
        actually_matched_mask = score_mat[match_rows, match_cols] > 0 + EPS
        match_rows = match_rows[actually_matched_mask]
        match_cols = match_cols[actually_matched_mask]

        matched_gt_indices = gt_indices_t[match_rows]
        matched_tracker_ids_t = tracker_ids_t[match_cols]

        # Compute ID switches (ref: clear.py:94-97)
        # IDSW occurs when GT was previously matched to a different tracker
        prev_matched_tracker_ids = prev_tracker_id[matched_gt_indices]
        is_idsw = ~np.isnan(prev_matched_tracker_ids) & np.not_equal(
            matched_tracker_ids_t, prev_matched_tracker_ids
        )
        idsw += int(np.sum(is_idsw))

        # Update per-GT counters
        gt_id_count[gt_indices_t] += 1
        gt_matched_count[matched_gt_indices] += 1

        # Track fragmentations (ref: clear.py:102-107)
        not_previously_tracked = np.isnan(prev_timestep_tracker_id)
        prev_tracker_id[matched_gt_indices] = matched_tracker_ids_t
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt_indices] = matched_tracker_ids_t
        currently_tracked = ~np.isnan(prev_timestep_tracker_id)
        gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

        # Accumulate basic statistics
        num_matches = len(matched_gt_indices)
        clr_tp += num_matches
        clr_fn += len(gt_ids_t) - num_matches
        clr_fp += len(tracker_ids_t) - num_matches

        if num_matches > 0:
            motp_sum += float(np.sum(similarity[match_rows, match_cols]))

    # Compute MT/PT/ML (ref: clear.py:118-121)
    valid_mask = gt_id_count > 0
    tracked_ratio = gt_matched_count[valid_mask] / gt_id_count[valid_mask]
    mt = int(np.sum(np.greater(tracked_ratio, 0.8)))
    pt = int(np.sum(np.greater_equal(tracked_ratio, 0.2))) - mt
    ml = num_gt_ids - mt - pt

    # Compute fragmentations
    frag = int(np.sum(np.subtract(gt_frag_count[gt_frag_count > 0], 1)))

    # Compute final metrics (ref: clear.py:167-186)
    num_gt_ids_total = mt + pt + ml
    mtr = mt / max(1.0, num_gt_ids_total)
    ptr = pt / max(1.0, num_gt_ids_total)
    mlr = ml / max(1.0, num_gt_ids_total)
    mota = (clr_tp - clr_fp - idsw) / max(1.0, clr_tp + clr_fn)
    motp = motp_sum / max(1.0, clr_tp)

    return {
        "CLR_TP": clr_tp,
        "CLR_FN": clr_fn,
        "CLR_FP": clr_fp,
        "IDSW": idsw,
        "MOTA": float(mota),
        "MOTP": float(motp),
        "MT": mt,
        "PT": pt,
        "ML": ml,
        "MTR": float(mtr),
        "PTR": float(ptr),
        "MLR": float(mlr),
        "Frag": frag,
    }
