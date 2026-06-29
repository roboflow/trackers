# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""End-to-end re-ID evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackers.core.reid.eval.datasets import ReidSplit
from trackers.core.reid.eval.metrics import ReidMetrics, compute_reid_metrics
from trackers.core.reid.model import ReIDModel


@dataclass
class ReidResult:
    """Evaluation output: metrics, raw embeddings, and optional distance matrix."""

    metrics: ReidMetrics
    query_embeddings: np.ndarray
    gallery_embeddings: np.ndarray
    distmat: np.ndarray


def _distance_matrix(q_embs: np.ndarray, g_embs: np.ndarray, metric: str) -> np.ndarray:
    """Build a query×gallery distance matrix (``cosine`` or ``euclidean``)."""
    # All ops below are done in place on a single (Nq, Ng) array to keep peak
    # memory at one distance matrix — important for large galleries (e.g.
    # MSMT17's 11.7k × 82.2k matrix is ~3.8 GB on its own).
    if metric == "cosine":
        qn = (q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-12)).astype(
            np.float32, copy=False
        )
        gn = (g_embs / (np.linalg.norm(g_embs, axis=1, keepdims=True) + 1e-12)).astype(
            np.float32, copy=False
        )
        distmat = qn @ gn.T  # cosine similarity
        distmat *= -1.0
        distmat += 1.0  # → 1 − cosine similarity
        return distmat
    if metric == "euclidean":
        distmat = (q_embs @ g_embs.T).astype(np.float32, copy=False)
        distmat *= -2.0
        distmat += (q_embs**2).sum(axis=1, keepdims=True)
        distmat += (g_embs**2).sum(axis=1, keepdims=True).T
        np.maximum(distmat, 0.0, out=distmat)
        np.sqrt(distmat, out=distmat)
        return distmat
    raise ValueError(f"Unknown distance metric: {metric!r}. Use 'cosine' or 'euclidean'.")


class ReidEvaluator:
    """Run embedding extraction and retrieval metrics for a :class:`ReIDModel`.

    Args:
        model: Appearance model to evaluate.
        batch_size: Images per forward pass.
    """

    def __init__(self, model: ReIDModel, batch_size: int = 64) -> None:
        self._model = model
        self._batch_size = batch_size

    def evaluate(
        self,
        query: ReidSplit,
        gallery: ReidSplit,
        max_rank: int = 10,
        verbose: bool = True,
        distance: str = "cosine",
        query_embeddings: np.ndarray | None = None,
        gallery_embeddings: np.ndarray | None = None,
        return_distmat: bool = True,
    ) -> ReidResult:
        """Extract embeddings (unless provided) and compute retrieval metrics.

        Args:
            query: Query split.
            gallery: Gallery split.
            max_rank: Highest CMC rank to report.
            verbose: Print progress to stdout.
            distance: ``"cosine"`` or ``"euclidean"``.
            query_embeddings: Optional pre-extracted raw query embeddings.
            gallery_embeddings: Optional pre-extracted raw gallery embeddings.
            return_distmat: Return the distance matrix (set ``False`` to save memory).

        Returns:
            :class:`ReidResult`.
        """
        if query_embeddings is None or gallery_embeddings is None:
            if verbose:
                print(f"Extracting query embeddings  ({len(query)} images)…")
            q_embs = self._model.extract_features_from_paths(
                query.image_paths, batch_size=self._batch_size, normalize=False
            )

            if verbose:
                print(f"Extracting gallery embeddings ({len(gallery)} images)…")
            g_embs = self._model.extract_features_from_paths(
                gallery.image_paths, batch_size=self._batch_size, normalize=False
            )
        else:
            q_embs, g_embs = query_embeddings, gallery_embeddings

        if verbose:
            print(f"Computing distance matrix ({distance})…")
        distmat = _distance_matrix(q_embs, g_embs, distance)

        if verbose:
            print("Computing metrics…")
        metrics = compute_reid_metrics(
            distmat=distmat,
            q_pids=query.pids,
            g_pids=gallery.pids,
            q_camids=query.camids,
            g_camids=gallery.camids,
            max_rank=max_rank,
        )

        if verbose:
            print(f"\nResults ({distance})\n{'-' * 50}\n{metrics}\n{'-' * 50}")

        if not return_distmat:
            del distmat
            distmat = np.empty((0, 0), dtype=np.float32)

        return ReidResult(
            metrics=metrics,
            query_embeddings=q_embs,
            gallery_embeddings=g_embs,
            distmat=distmat,
        )
