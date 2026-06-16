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
    """Full result of a re-ID evaluation run.

    Attributes:
        metrics: CMC / mAP / mINP scores.
        query_embeddings: Raw (un-normalised) query embeddings, shape ``(Nq, D)``.
            Can be passed back into :meth:`ReidEvaluator.evaluate` to re-score
            under a different distance metric without re-extracting features.
        gallery_embeddings: Raw (un-normalised) gallery embeddings, shape ``(Ng, D)``.
        distmat: Distance matrix used for scoring, shape ``(Nq, Ng)``.
    """

    metrics: ReidMetrics
    query_embeddings: np.ndarray
    gallery_embeddings: np.ndarray
    distmat: np.ndarray


def _distance_matrix(q_embs: np.ndarray, g_embs: np.ndarray, metric: str) -> np.ndarray:
    """Build a query×gallery distance matrix under the requested metric.

    Args:
        q_embs: Raw query embeddings, shape ``(Nq, D)``.
        g_embs: Raw gallery embeddings, shape ``(Ng, D)``.
        metric: ``"cosine"`` (1 − cosine similarity on L2-normalised vectors)
            or ``"euclidean"`` (L2 distance on the raw vectors).

    Returns:
        Float32 distance matrix of shape ``(Nq, Ng)`` (lower = more similar).

    Raises:
        ValueError: If *metric* is not ``"cosine"`` or ``"euclidean"``.
    """
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
    """Evaluates a :class:`~trackers.core.reid.model.ReIDModel` on a re-ID dataset.

    Encapsulates the full pipeline: batched embedding extraction →
    cosine distance matrix → standard query/gallery metrics.

    Args:
        model: The appearance model to evaluate.
        batch_size: Images per forward pass. Reduce if GPU memory is tight
            (default 64 works on a T4 / A100 Colab instance for OSNet).

    Examples:
        >>> evaluator = ReidEvaluator(None, batch_size=32)  # doctest: +SKIP
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
        """Run end-to-end evaluation on query and gallery splits.

        Steps:

        1. Batch-extract raw embeddings for query and gallery (skipped when
           *query_embeddings* / *gallery_embeddings* are provided).
        2. Build the distance matrix under *distance*.
        3. Apply the junk rule and compute CMC / mAP / mINP.

        Args:
            query: Query split (anchors).
            gallery: Gallery split (search pool).
            max_rank: Highest CMC rank to report (default 10).
            verbose: Print progress messages to stdout.
            distance: ``"cosine"`` (1 − cosine similarity on L2-normalised
                embeddings, the default) or ``"euclidean"`` (L2 distance on the
                raw embeddings, matching the torchreid model-zoo protocol).
            query_embeddings: Optional pre-extracted **raw** query embeddings.
                When given (together with *gallery_embeddings*), feature
                extraction is skipped — useful for re-scoring the same
                embeddings under a different *distance*.
            gallery_embeddings: Optional pre-extracted **raw** gallery embeddings.
            return_distmat: If ``True`` (default) the full distance matrix is
                returned on the result. Set ``False`` to free it immediately
                after scoring — important for large galleries (MSMT17's matrix
                is ~3.8 GB), especially when re-scoring under multiple metrics.

        Returns:
            :class:`ReidResult` containing metrics and raw embeddings. The
            ``distmat`` field is the scoring matrix, or an empty array when
            *return_distmat* is ``False``.
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
