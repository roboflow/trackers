# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from trackers.core.reid.eval.datasets import MARKET1501_GALLERY_JUNK_PIDS, ReIDSplit
from trackers.core.reid.eval.evaluator import ReIDEvaluator
from trackers.core.reid.eval.metrics import compute_reid_metrics


def _split(
    q_pid: int,
    g_pids: list[int],
    *,
    gallery_junk_pids: frozenset[int] = frozenset({-1}),
) -> tuple[ReIDSplit, ReIDSplit]:
    query = ReIDSplit(
        image_paths=["q.jpg"],
        pids=np.array([q_pid], dtype=np.int32),
        camids=np.array([0], dtype=np.int32),
        gallery_junk_pids=frozenset({-1}),
    )
    gallery = ReIDSplit(
        image_paths=[f"g{i}.jpg" for i in range(len(g_pids))],
        pids=np.array(g_pids, dtype=np.int32),
        camids=np.zeros(len(g_pids), dtype=np.int32),
        gallery_junk_pids=gallery_junk_pids,
    )
    return query, gallery


class TestComputeReidMetrics:
    def test_market1501_pid_zero_is_junk_in_gallery(self) -> None:
        distmat = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        metrics = compute_reid_metrics(
            distmat,
            q_pids=np.array([1]),
            g_pids=np.array([0, 1, 2]),
            q_camids=np.array([0]),
            g_camids=np.array([0, 1, 0]),
            max_rank=3,
            gallery_junk_pids=MARKET1501_GALLERY_JUNK_PIDS,
        )
        assert metrics.rank1 == pytest.approx(100.0)

    def test_pid_zero_valid_when_not_marked_junk(self) -> None:
        distmat = np.array([[0.2, 0.1]], dtype=np.float32)
        metrics = compute_reid_metrics(
            distmat,
            q_pids=np.array([0]),
            g_pids=np.array([1, 0]),
            q_camids=np.array([0]),
            g_camids=np.array([0, 1]),
            max_rank=2,
            gallery_junk_pids=frozenset({-1}),
        )
        assert metrics.rank1 == pytest.approx(100.0)

    def test_cmc_pads_when_valid_gallery_shorter_than_max_rank(self) -> None:
        distmat = np.array([[0.2, 0.1]], dtype=np.float32)
        metrics = compute_reid_metrics(
            distmat,
            q_pids=np.array([3]),
            g_pids=np.array([3, 8]),
            q_camids=np.array([0]),
            g_camids=np.array([1, 0]),
            max_rank=5,
        )
        assert metrics.rank1 == pytest.approx(0.0)
        assert metrics.rank5 == pytest.approx(100.0)
        assert metrics.rank10 == pytest.approx(100.0)

    def test_evaluator_passes_gallery_junk_pids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        def _fake_metrics(**kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            from trackers.core.reid.eval.metrics import ReIDMetrics

            return ReIDMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 1)

        monkeypatch.setattr("trackers.core.reid.eval.evaluator.compute_reid_metrics", _fake_metrics)

        query, gallery = _split(1, [0, 1], gallery_junk_pids=MARKET1501_GALLERY_JUNK_PIDS)
        q_embs = np.ones((1, 4), dtype=np.float32)
        g_embs = np.ones((2, 4), dtype=np.float32)

        class _StubModel:
            def extract_features_from_paths(self, *args, **kwargs):  # noqa: ANN002, ANN003
                raise AssertionError("should not be called")

        ReIDEvaluator(_StubModel()).evaluate(
            query,
            gallery,
            query_embeddings=q_embs,
            gallery_embeddings=g_embs,
            verbose=False,
        )
        assert captured["gallery_junk_pids"] == MARKET1501_GALLERY_JUNK_PIDS

    def test_reid_metrics_map_alias(self) -> None:
        from trackers.core.reid.eval.metrics import ReIDMetrics

        metrics = ReIDMetrics(mean_average_precision=42.0, rank1=1.0, rank5=2.0, rank10=3.0, minp=4.0, num_queries=1)
        assert metrics.map == pytest.approx(42.0)

    def test_evaluator_rejects_invalid_batch_size(self) -> None:
        class _StubModel:
            pass

        with pytest.raises(ValueError, match="batch_size"):
            ReIDEvaluator(_StubModel(), batch_size=0)
