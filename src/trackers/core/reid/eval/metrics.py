# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Standard person re-ID retrieval metrics.

Implements the query/gallery evaluation protocol used by Market-1501, MSMT17,
and most other re-ID benchmarks:

- **CMC Rank-k** — probability that at least one correct match appears in the
  top-k retrieved gallery items.
- **mAP** — mean Average Precision across all queries; rewards retrieving *all*
  correct instances highly, not just one.
- **mINP** — mean Inverse Negative Penalty; penalises how far down the *hardest*
  correct match falls.

Junk rule (standard):
  When evaluating a query ``(pid, camid)``, gallery items that share *both* the
  same person ID **and** the same camera ID are excluded (trivially easy
  same-camera matches). Items with ``pid == -1`` (Market-1501 distractors) are
  also excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReidMetrics:
    """Re-ID retrieval metrics for a single evaluation run.

    All values are percentages in ``[0, 100]``.

    Attributes:
        map: Mean Average Precision.
        rank1: CMC Rank-1 accuracy.
        rank5: CMC Rank-5 accuracy.
        rank10: CMC Rank-10 accuracy.
        minp: Mean Inverse Negative Penalty.
        num_queries: Number of valid queries used in the computation.
    """

    map: float
    rank1: float
    rank5: float
    rank10: float
    minp: float
    num_queries: int

    def __str__(self) -> str:
        return (
            f"mAP: {self.map:.1f}%  "
            f"Rank-1: {self.rank1:.1f}%  "
            f"Rank-5: {self.rank5:.1f}%  "
            f"Rank-10: {self.rank10:.1f}%  "
            f"mINP: {self.minp:.1f}%  "
            f"(n={self.num_queries})"
        )


def compute_reid_metrics(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 10,
) -> ReidMetrics:
    """Compute CMC, mAP, and mINP from a pre-computed distance matrix.

    Applies the standard junk rule: for each query ``(pid, camid)`` gallery
    items with the same ``pid`` **and** the same ``camid`` are excluded, as
    are distractor items (``pid == -1``).

    Args:
        distmat: Distance matrix of shape ``(num_queries, num_gallery)``.
            Lower values mean more similar. Typically 1 − cosine similarity
            for L2-normalised embeddings.
        q_pids: Person IDs for each query, shape ``(num_queries,)``.
        g_pids: Person IDs for each gallery item, shape ``(num_gallery,)``.
        q_camids: Camera IDs for each query, shape ``(num_queries,)``.
        g_camids: Camera IDs for each gallery item, shape ``(num_gallery,)``.
        max_rank: Highest CMC rank to compute (must be ≤ ``num_gallery``).

    Returns:
        :class:`ReidMetrics` with all scores as percentages.

    Raises:
        ValueError: If ``distmat`` shape is inconsistent with the pid/camid
            arrays, or if ``max_rank`` exceeds the gallery size.

    Examples:
        >>> import numpy as np
        >>> distmat = np.array([[0.1, 0.9], [0.8, 0.2]])
        >>> q_pids   = np.array([1, 2])
        >>> g_pids   = np.array([1, 2])
        >>> q_camids = np.array([0, 0])
        >>> g_camids = np.array([1, 1])
        >>> m = compute_reid_metrics(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=2)
        >>> m.rank1
        100.0
    """
    num_q, num_g = distmat.shape

    if len(q_pids) != num_q or len(q_camids) != num_q:
        raise ValueError("q_pids / q_camids length must match distmat rows.")
    if len(g_pids) != num_g or len(g_camids) != num_g:
        raise ValueError("g_pids / g_camids length must match distmat columns.")
    if max_rank > num_g:
        raise ValueError(f"max_rank ({max_rank}) exceeds gallery size ({num_g}).")

    cmc_accumulator = np.zeros(max_rank, dtype=np.float64)
    ap_list: list[float] = []
    inp_list: list[float] = []

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # Sort gallery indices by ascending distance (closest first).
        order = np.argsort(distmat[q_idx])
        sorted_g_pids = g_pids[order]
        sorted_g_camids = g_camids[order]

        # Junk mask: trivial same-camera matches + Market-1501 distractors.
        junk = ((sorted_g_pids == q_pid) & (sorted_g_camids == q_camid)) | (sorted_g_pids == -1)
        valid = ~junk

        sorted_g_pids_valid = sorted_g_pids[valid]
        matches = (sorted_g_pids_valid == q_pid).astype(np.float32)

        num_rel = int(matches.sum())
        if num_rel == 0:
            # No valid positives for this query — skip (happens on corrupted splits).
            continue

        # --- CMC ---
        # cmc_q[k] = 1 if any correct match in top (k+1).
        cmc_q = np.minimum(matches.cumsum(), 1.0)
        cmc_accumulator += cmc_q[:max_rank]

        # --- mAP ---
        num_valid = len(matches)
        precision_at_k = matches.cumsum() / (np.arange(num_valid) + 1)
        ap = float((precision_at_k * matches).sum() / num_rel)
        ap_list.append(ap)

        # --- mINP ---
        last_true = int(np.where(matches)[0][-1]) + 1  # 1-indexed position
        inp_list.append(num_rel / last_true)

    n = len(ap_list)
    if n == 0:
        return ReidMetrics(map=0.0, rank1=0.0, rank5=0.0, rank10=0.0, minp=0.0, num_queries=0)

    cmc = (cmc_accumulator / n) * 100.0

    return ReidMetrics(
        map=float(np.mean(ap_list)) * 100.0,
        rank1=float(cmc[0]),
        rank5=float(cmc[min(4, max_rank - 1)]),
        rank10=float(cmc[min(9, max_rank - 1)]),
        minp=float(np.mean(inp_list)) * 100.0,
        num_queries=n,
    )
