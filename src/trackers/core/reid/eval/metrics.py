# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID retrieval metrics (CMC, mAP, mINP)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReIDMetrics:
    """CMC / mAP / mINP scores (percentages) for one evaluation run."""

    mean_average_precision: float
    rank1: float
    rank5: float
    rank10: float
    minp: float
    num_queries: int

    def __str__(self) -> str:
        return (
            f"mAP: {self.mean_average_precision:.1f}%  "
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
    *,
    gallery_junk_pids: frozenset[int] = frozenset({-1}),
) -> ReIDMetrics:
    """Compute CMC, mAP, and mINP from a query x gallery distance matrix.

    Excludes same-(pid, camid) gallery matches and gallery junk person IDs
    (``pid in gallery_junk_pids``). Market-1501 gallery distractors use
    ``gallery_junk_pids=frozenset({-1, 0})``; datasets where ``pid=0`` is
    valid should keep the default ``{-1}`` only.

    Args:
        distmat: Distance matrix ``(num_queries, num_gallery)`` (lower = closer).
        q_pids: Query person IDs.
        g_pids: Gallery person IDs.
        q_camids: Query camera IDs.
        g_camids: Gallery camera IDs.
        max_rank: Highest CMC rank to compute (must be ``>= 10``).
        gallery_junk_pids: Gallery person IDs treated as junk during ranking.

    Returns:
        ``ReIDMetrics`` with scores as percentages.

    Raises:
        ValueError: If ID lengths do not match ``distmat``, or ``max_rank < 10``.
    """
    num_q, num_g = distmat.shape

    if len(q_pids) != num_q or len(q_camids) != num_q:
        raise ValueError("q_pids / q_camids length must match distmat rows.")
    if len(g_pids) != num_g or len(g_camids) != num_g:
        raise ValueError("g_pids / g_camids length must match distmat columns.")
    if max_rank < 10:
        raise ValueError(f"max_rank must be >= 10, got {max_rank}")

    cmc_accumulator = np.zeros(max_rank, dtype=np.float64)
    ap_list: list[float] = []
    inp_list: list[float] = []

    junk_pid_array = np.array(sorted(gallery_junk_pids), dtype=g_pids.dtype) if gallery_junk_pids else np.empty(0)

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = np.argsort(distmat[q_idx])
        sorted_g_pids = g_pids[order]
        sorted_g_camids = g_camids[order]

        junk = (sorted_g_pids == q_pid) & (sorted_g_camids == q_camid)
        if junk_pid_array.size > 0:
            junk |= np.isin(sorted_g_pids, junk_pid_array)
        valid = ~junk

        sorted_g_pids_valid = sorted_g_pids[valid]
        matches = (sorted_g_pids_valid == q_pid).astype(np.float32)

        num_rel = int(matches.sum())
        if num_rel == 0:
            continue

        cmc_q = np.minimum(matches.cumsum(), 1.0)
        if len(cmc_q) < max_rank:
            if len(cmc_q) == 0:
                continue
            cmc_q = np.pad(cmc_q, (0, max_rank - len(cmc_q)), constant_values=cmc_q[-1])
        else:
            cmc_q = cmc_q[:max_rank]
        cmc_accumulator += cmc_q

        num_valid = len(matches)
        precision_at_k = matches.cumsum() / (np.arange(num_valid) + 1)
        ap = float((precision_at_k * matches).sum() / num_rel)
        ap_list.append(ap)

        last_true = int(np.where(matches)[0][-1]) + 1
        inp_list.append(num_rel / last_true)

    n = len(ap_list)
    if n == 0:
        return ReIDMetrics(
            mean_average_precision=0.0,
            rank1=0.0,
            rank5=0.0,
            rank10=0.0,
            minp=0.0,
            num_queries=0,
        )

    cmc = (cmc_accumulator / n) * 100.0

    return ReIDMetrics(
        mean_average_precision=float(np.mean(ap_list)) * 100.0,
        rank1=float(cmc[0]),
        rank5=float(cmc[min(4, max_rank - 1)]),
        rank10=float(cmc[min(9, max_rank - 1)]),
        minp=float(np.mean(inp_list)) * 100.0,
        num_queries=n,
    )
