# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Adapted from TrackEval (https://github.com/JonathonLuiten/TrackEval)
# Copyright (c) Jonathon Luiten. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# Epsilon for floating point comparisons - must match TrackEval exactly
EPS = np.finfo("float").eps


def match_detections(
    similarity_matrix: np.ndarray,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find optimal assignment between ground truth and tracker detections.
    Use the Hungarian algorithm (Jonker-Volgenant) to compute the optimal
    one-to-one assignment that maximizes total similarity. Matches with
    similarity below the threshold are filtered out.

    Args:
        similarity_matrix: Similarity scores with shape `(N, M)` where N is the
            number of ground truth detections and M is the number of tracker
            detections. Higher values indicate better matches.
        threshold: Minimum similarity score to consider a valid match. Matches
            with similarity at or below this value are filtered out.

    Returns:
        Tuple of four `numpy.ndarray`:
        - `matched_gt_indices`: Indices of matched ground truth detections.
        - `matched_tracker_indices`: Indices of matched tracker detections.
        - `unmatched_gt_indices`: Indices of unmatched ground truth detections.
        - `unmatched_tracker_indices`: Indices of unmatched tracker detections.

    Examples:
        ```python
        import numpy as np
        from trackers.eval import match_detections

        similarity_matrix = np.array([
            [0.9, 0.1, 0.0],
            [0.2, 0.8, 0.1],
            [0.0, 0.1, 0.7],
        ])
        gt_idx, tr_idx, unmatched_gt, unmatched_tr = match_detections(
            similarity_matrix, threshold=0.5
        )
        print(gt_idx, tr_idx)
        # [0 1 2] [0 1 2]
        ```

        ```python
        import numpy as np
        from trackers.eval import match_detections

        similarity_matrix = np.array([
            [0.9, 0.1],
            [0.2, 0.3],
        ])
        gt_idx, tr_idx, unmatched_gt, unmatched_tr = match_detections(
            similarity_matrix, threshold=0.5
        )
        print(gt_idx, tr_idx, unmatched_gt, unmatched_tr)
        # [0] [0] [1] [1]
        ```
    """
    num_gt = similarity_matrix.shape[0] if len(similarity_matrix.shape) >= 1 else 0
    num_tracker = similarity_matrix.shape[1] if len(similarity_matrix.shape) >= 2 else 0

    # Handle empty inputs
    if num_gt == 0:
        return (
            np.array([], dtype=np.intp),
            np.array([], dtype=np.intp),
            np.array([], dtype=np.intp),
            np.arange(num_tracker, dtype=np.intp),
        )
    if num_tracker == 0:
        return (
            np.array([], dtype=np.intp),
            np.array([], dtype=np.intp),
            np.arange(num_gt, dtype=np.intp),
            np.array([], dtype=np.intp),
        )

    # Hungarian algorithm to find optimal assignment (maximize similarity)
    match_rows, match_cols = linear_sum_assignment(-similarity_matrix)

    # Filter matches below threshold
    actually_matched_mask = similarity_matrix[match_rows, match_cols] > threshold + EPS
    matched_gt_indices = match_rows[actually_matched_mask]
    matched_tracker_indices = match_cols[actually_matched_mask]

    # Compute unmatched indices
    all_gt_indices = np.arange(num_gt, dtype=np.intp)
    all_tracker_indices = np.arange(num_tracker, dtype=np.intp)
    unmatched_gt_indices = np.setdiff1d(all_gt_indices, matched_gt_indices)
    unmatched_tracker_indices = np.setdiff1d(
        all_tracker_indices, matched_tracker_indices
    )

    return (
        matched_gt_indices.astype(np.intp),
        matched_tracker_indices.astype(np.intp),
        unmatched_gt_indices.astype(np.intp),
        unmatched_tracker_indices.astype(np.intp),
    )
