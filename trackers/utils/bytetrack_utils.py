from typing import List, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = detections.confidence
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def linear_assignment_with_cost_limit_scipy(
    cost_matrix: np.ndarray,
    cost_limit: float,
) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
    """
    SciPy linear_sum_assignment with added cost limit to remove from matching the one that have higer cost.
    In order to do so, we need to extend the cost matrix to be square and add dummy rows/columns,
    that if matched indicate unmatch for the original rows/columns. The cost of matching to a dummy row/column is set to cost_limit / 2,
    which ensures that any match with cost higher than cost_limit will not be selected because matching to a dummy will be cheaper.

    Returns:
        matches: list of (track_idx, det_idx)
        matched_tracks: set of track indices
        matched_dets: set of detection indices
    """  # noqa: E501
    n_tracks, n_dets = cost_matrix.shape

    # Handle empty cases
    if n_tracks == 0 or n_dets == 0:
        return [], set(), set()

    matched_indices: List[Tuple[int, int]] = []
    matched_tracks: Set[int] = set()
    matched_dets: Set[int] = set()

    if not np.isfinite(cost_limit):
        # Standard assignment
        r, c = linear_sum_assignment(cost_matrix)
        matches = list(zip(r.tolist(), c.tolist()))
        matched_tracks = set(r.tolist())
        matched_dets = set(c.tolist())
        return matches, matched_tracks, matched_dets

    # If not square, extend: N = n_tracks + n_dets
    N = n_tracks + n_dets
    Cext = np.full((N, N), cost_limit / 2.0, dtype=float)
    Cext[:n_tracks, :n_dets] = cost_matrix  # real costs
    Cext[n_tracks:, n_dets:] = 0.0  # dummy-to-dummy block

    # Solve extended assignment with scipy
    r, c = linear_sum_assignment(Cext)

    for rr, cc in zip(r, c):
        # rr < n_tracks and cc >= n_dets  => track rr is assigned to a dummy column (unmatched track) # noqa: E501
        # rr >= n_tracks and cc < n_dets  => detection cc is assigned to a dummy row (unmatched detection) # noqa: E501
        # rr >= n_tracks and cc >= n_dets => dummy-dummy, ignore

        # Real track -> real detection assignment
        if rr < n_tracks and cc < n_dets:
            # we enforce it explicitly anyway,
            # even though the dummy assignment cost should do it
            if cost_matrix[rr, cc] <= cost_limit:
                matched_indices.append((int(rr), int(cc)))
                matched_tracks.add(int(rr))
                matched_dets.add(int(cc))
    return matched_indices, matched_tracks, matched_dets
