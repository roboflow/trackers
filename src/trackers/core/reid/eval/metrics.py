# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Re-ID retrieval metrics (CMC, mAP, mINP)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReidMetrics:
    """CMC / mAP / mINP scores (percentages) for one evaluation run."""

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
    """Compute CMC, mAP, and mINP from a query×gallery distance matrix.

    Excludes same-(pid, camid) gallery matches and ``pid == -1`` junk items.

    Args:
        distmat: Distance matrix ``(num_queries, num_gallery)`` (lower = closer).
        q_pids: Query person IDs.
        g_pids: Gallery person IDs.
        q_camids: Query camera IDs.
        g_camids: Gallery camera IDs.
        max_rank: Highest CMC rank to compute.

    Returns:
        :class:`ReidMetrics` with scores as percentages.
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
