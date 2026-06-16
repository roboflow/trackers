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
        query_embeddings: L2-normalised query embeddings, shape ``(Nq, D)``.
        gallery_embeddings: L2-normalised gallery embeddings, shape ``(Ng, D)``.
        distmat: Cosine distance matrix, shape ``(Nq, Ng)``.
    """

    metrics: ReidMetrics
    query_embeddings: np.ndarray
    gallery_embeddings: np.ndarray
    distmat: np.ndarray


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
    ) -> ReidResult:
        """Run end-to-end evaluation on query and gallery splits.

        Steps:

        1. Batch-extract L2-normalised embeddings for query and gallery.
        2. Compute the cosine distance matrix (1 − dot product for normalised
           vectors).
        3. Apply the junk rule and compute CMC / mAP / mINP.

        Args:
            query: Query split (anchors).
            gallery: Gallery split (search pool).
            max_rank: Highest CMC rank to report (default 10).
            verbose: Print progress messages to stdout.

        Returns:
            :class:`ReidResult` containing metrics, embeddings, and the
            distance matrix.
        """
        if verbose:
            print(f"Extracting query embeddings  ({len(query)} images)…")
        q_embs = self._model.extract_features_from_paths(
            query.image_paths, batch_size=self._batch_size
        )

        if verbose:
            print(f"Extracting gallery embeddings ({len(gallery)} images)…")
        g_embs = self._model.extract_features_from_paths(
            gallery.image_paths, batch_size=self._batch_size
        )

        if verbose:
            print("Computing distance matrix…")
        # Cosine distance = 1 − cosine similarity. For L2-normalised vectors,
        # cosine similarity = dot product, so distance = 1 − (q @ g.T).
        distmat = 1.0 - (q_embs @ g_embs.T).astype(np.float32)

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
            print(f"\nResults\n{'-' * 50}\n{metrics}\n{'-' * 50}")

        return ReidResult(
            metrics=metrics,
            query_embeddings=q_embs,
            gallery_embeddings=g_embs,
            distmat=distmat,
        )
