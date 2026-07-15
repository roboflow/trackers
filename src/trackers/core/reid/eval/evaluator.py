# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""End-to-end ReID evaluation pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from trackers.core.reid.encoder import ReIDPathEncoder
from trackers.core.reid.eval.datasets import ReIDSplit
from trackers.core.reid.eval.metrics import ReIDMetrics, compute_reid_metrics

logger = logging.getLogger(__name__)


@dataclass
class ReIDResult:
    """Metrics, embeddings, and optional distance matrix from one evaluation run."""

    metrics: ReIDMetrics
    query_embeddings: np.ndarray
    gallery_embeddings: np.ndarray
    distmat: np.ndarray


def _distance_matrix(q_embs: np.ndarray, g_embs: np.ndarray, metric: str) -> np.ndarray:
    """Build a query x gallery distance matrix (``cosine`` or ``euclidean``)."""
    if metric == "cosine":
        qn = (q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-12)).astype(np.float32, copy=False)
        gn = (g_embs / (np.linalg.norm(g_embs, axis=1, keepdims=True) + 1e-12)).astype(np.float32, copy=False)
        distmat = qn @ gn.T
        distmat *= -1.0
        distmat += 1.0
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


class ReIDEvaluator:
    """Run embedding extraction and retrieval metrics for a ReID encoder.

    Args:
        model: Encoder implementing ``ReIDPathEncoder`` (for example ``ReIDModel``).
        batch_size: Images per forward pass.
    """

    def __init__(self, model: ReIDPathEncoder, batch_size: int = 64) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._model = model
        self._batch_size = batch_size

    def evaluate(
        self,
        query: ReIDSplit,
        gallery: ReIDSplit,
        max_rank: int = 10,
        verbose: bool = True,
        distance: str = "cosine",
        query_embeddings: np.ndarray | None = None,
        gallery_embeddings: np.ndarray | None = None,
        return_distmat: bool = True,
    ) -> ReIDResult:
        """Extract embeddings (unless provided) and compute retrieval metrics.

        Args:
            query: Query split.
            gallery: Gallery split.
            max_rank: Highest CMC rank to report.
            verbose: Log progress when ``True``.
            distance: ``"cosine"`` or ``"euclidean"``.
            query_embeddings: Optional pre-extracted query embeddings.
                When omitted, query embeddings are extracted from ``query``.
            gallery_embeddings: Optional pre-extracted gallery embeddings.
                When omitted, gallery embeddings are extracted from ``gallery``.
            return_distmat: Return the distance matrix (set ``False`` to save memory).

        Returns:
            ``ReIDResult`` with metrics, embeddings, and distance matrix
            (empty distmat when ``return_distmat`` is ``False``).
        """
        if query_embeddings is None:
            if verbose:
                logger.info("Extracting query embeddings (%s images)…", len(query))
            q_embs = self._model.extract_features_from_paths(
                query.image_paths, batch_size=self._batch_size, normalize=False
            )
        else:
            if query_embeddings.shape[0] != len(query):
                raise ValueError(
                    f"query_embeddings rows ({query_embeddings.shape[0]}) must match query length ({len(query)})"
                )
            q_embs = query_embeddings

        if gallery_embeddings is None:
            if verbose:
                logger.info("Extracting gallery embeddings (%s images)…", len(gallery))
            g_embs = self._model.extract_features_from_paths(
                gallery.image_paths, batch_size=self._batch_size, normalize=False
            )
        else:
            if gallery_embeddings.shape[0] != len(gallery):
                raise ValueError(
                    f"gallery_embeddings rows ({gallery_embeddings.shape[0]}) must "
                    f"match gallery length ({len(gallery)})"
                )
            g_embs = gallery_embeddings

        if verbose:
            logger.info("Computing distance matrix (%s)…", distance)
        distmat = _distance_matrix(q_embs, g_embs, distance)

        if verbose:
            logger.info("Computing metrics…")
        metrics = compute_reid_metrics(
            distmat=distmat,
            q_pids=query.pids,
            g_pids=gallery.pids,
            q_camids=query.camids,
            g_camids=gallery.camids,
            max_rank=max_rank,
            gallery_junk_pids=gallery.gallery_junk_pids,
        )

        if verbose:
            logger.info("Results (%s)\n%s\n%s\n%s", distance, "-" * 50, metrics, "-" * 50)

        if not return_distmat:
            del distmat
            distmat = np.empty((0, 0), dtype=np.float32)

        return ReIDResult(
            metrics=metrics,
            query_embeddings=q_embs,
            gallery_embeddings=g_embs,
            distmat=distmat,
        )
