# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""NumPy-only ReID gallery evaluation tests.

These do not require ``trackers[reid]`` and run in every CI job.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

# ---------------------------------------------------------------------------
# Eval metrics + dataset loaders
# ---------------------------------------------------------------------------


class TestComputeReidMetrics:
    def test_market1501_pid_zero_is_junk_in_gallery(self) -> None:
        from trackers.core.reid.eval.datasets import MARKET1501_GALLERY_JUNK_PIDS
        from trackers.core.reid.eval.metrics import compute_reid_metrics

        metrics = compute_reid_metrics(
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            q_pids=np.array([1]),
            g_pids=np.array([0, 1, 2]),
            q_camids=np.array([0]),
            g_camids=np.array([0, 1, 0]),
            max_rank=10,
            gallery_junk_pids=MARKET1501_GALLERY_JUNK_PIDS,
        )
        assert metrics.rank1 == pytest.approx(100.0)

    def test_cmc_pads_when_valid_gallery_shorter_than_max_rank(self) -> None:
        from trackers.core.reid.eval.metrics import compute_reid_metrics

        metrics = compute_reid_metrics(
            np.array([[0.2, 0.1]], dtype=np.float32),
            q_pids=np.array([3]),
            g_pids=np.array([3, 8]),
            q_camids=np.array([0]),
            g_camids=np.array([1, 0]),
            max_rank=10,
        )
        assert metrics.rank1 == pytest.approx(0.0)
        assert metrics.rank5 == pytest.approx(100.0)

    def test_max_rank_below_ten_raises(self) -> None:
        from trackers.core.reid.eval.metrics import compute_reid_metrics

        with pytest.raises(ValueError, match="max_rank"):
            compute_reid_metrics(
                np.array([[0.1, 0.2]], dtype=np.float32),
                q_pids=np.array([1]),
                g_pids=np.array([1, 2]),
                q_camids=np.array([0]),
                g_camids=np.array([1, 0]),
                max_rank=5,
            )

    def test_computes_map_cmc_and_minp_on_controlled_ranking(self) -> None:
        """One query, two true gallery matches after junk removal.

        Distance order of gallery ids ``[2, 1, 1, 1]`` with cams ``[1, 1, 0, 2]``.
        Same-(pid, camid) as the query (third gallery entry) is junk, so the
        valid ranking is distractor → match → match.

        Hand-computed: Rank-1 = 0%, AP = 7/12, mINP = 2/3.
        """
        from trackers.core.reid.eval.metrics import compute_reid_metrics

        metrics = compute_reid_metrics(
            np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            q_pids=np.array([1]),
            g_pids=np.array([2, 1, 1, 1]),
            q_camids=np.array([0]),
            g_camids=np.array([1, 1, 0, 2]),
            max_rank=10,
            gallery_junk_pids=frozenset(),
        )
        assert metrics.num_queries == 1
        assert metrics.rank1 == pytest.approx(0.0)
        assert metrics.rank5 == pytest.approx(100.0)
        assert metrics.rank10 == pytest.approx(100.0)
        assert metrics.mean_average_precision == pytest.approx((7.0 / 12.0) * 100.0)
        assert metrics.minp == pytest.approx((2.0 / 3.0) * 100.0)


class TestReIDEvaluator:
    def test_reuses_provided_query_embeddings_only(self, tmp_path: Path) -> None:
        from trackers.core.reid.eval.datasets import ReIDSplit
        from trackers.core.reid.eval.evaluator import ReIDEvaluator

        query_img = tmp_path / "q.jpg"
        gallery_img = tmp_path / "g.jpg"
        query_img.write_bytes(b"jpeg")
        gallery_img.write_bytes(b"jpeg")
        query = ReIDSplit(
            image_paths=[str(query_img)],
            pids=np.array([1]),
            camids=np.array([0]),
        )
        gallery = ReIDSplit(
            image_paths=[str(gallery_img)],
            pids=np.array([1]),
            camids=np.array([1]),
        )

        class _Encoder:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
                raise NotImplementedError

            def extract_features_from_paths(
                self,
                image_paths: Sequence[str],
                *,
                batch_size: int = 64,
                normalize: bool = False,
            ) -> np.ndarray:
                self.calls.append(list(image_paths))
                return np.ones((len(image_paths), 2), dtype=np.float32)

        encoder = _Encoder()
        provided_query = np.array([[0.0, 1.0]], dtype=np.float32)
        result = ReIDEvaluator(encoder).evaluate(
            query,
            gallery,
            query_embeddings=provided_query,
            verbose=False,
        )
        assert encoder.calls == [[str(gallery_img)]]
        np.testing.assert_array_equal(result.query_embeddings, provided_query)

    def test_wrong_length_provided_embeddings_raise(self, tmp_path: Path) -> None:
        from trackers.core.reid.eval.datasets import ReIDSplit
        from trackers.core.reid.eval.evaluator import ReIDEvaluator

        query_img = tmp_path / "q.jpg"
        gallery_img = tmp_path / "g.jpg"
        query_img.write_bytes(b"jpeg")
        gallery_img.write_bytes(b"jpeg")
        query = ReIDSplit(
            image_paths=[str(query_img)],
            pids=np.array([1]),
            camids=np.array([0]),
        )
        gallery = ReIDSplit(
            image_paths=[str(gallery_img)],
            pids=np.array([1]),
            camids=np.array([1]),
        )

        class _Encoder:
            def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
                raise NotImplementedError

            def extract_features_from_paths(
                self,
                image_paths: Sequence[str],
                *,
                batch_size: int = 64,
                normalize: bool = False,
            ) -> np.ndarray:
                return np.ones((len(image_paths), 2), dtype=np.float32)

        evaluator = ReIDEvaluator(_Encoder())
        with pytest.raises(ValueError, match="query_embeddings"):
            evaluator.evaluate(
                query,
                gallery,
                query_embeddings=np.ones((2, 2), dtype=np.float32),
                verbose=False,
            )
        with pytest.raises(ValueError, match="gallery_embeddings"):
            evaluator.evaluate(
                query,
                gallery,
                gallery_embeddings=np.ones((3, 2), dtype=np.float32),
                verbose=False,
            )


class TestMarket1501Loader:
    def test_load_market1501_from_temp_tree(self, tmp_path) -> None:
        from trackers.core.reid.eval.datasets import (
            MARKET1501_GALLERY_JUNK_PIDS,
            load_market1501,
        )

        query_dir = tmp_path / "query"
        gallery_dir = tmp_path / "bounding_box_test"
        query_dir.mkdir()
        gallery_dir.mkdir()
        (query_dir / "0001_c1s1_001051_00.jpg").write_bytes(b"jpeg")
        (gallery_dir / "0000_c1s1_000151_01.jpg").write_bytes(b"jpeg")
        (gallery_dir / "0002_c2s1_000851_01.jpg").write_bytes(b"jpeg")

        query, gallery = load_market1501(tmp_path)
        assert len(query) == 1
        assert query.pids.tolist() == [1]
        assert gallery.pids.tolist() == [0, 2]
        assert gallery.gallery_junk_pids == MARKET1501_GALLERY_JUNK_PIDS


class TestMSMT17Loader:
    def test_load_msmt17_from_temp_lists(self, tmp_path) -> None:
        from trackers.core.reid.eval.datasets import load_msmt17

        root = Path(tmp_path)
        test_root = root / "test"
        test_root.mkdir()
        rel = "0001/0001_019_07_0303morning_0020_1.jpg"
        image_path = test_root / rel
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(b"jpeg")
        (root / "list_query.txt").write_text(f"{rel} 42\n")
        (root / "list_gallery.txt").write_text(f"{rel} 42 6\n")

        query, gallery = load_msmt17(root)
        assert query.pids.tolist() == [42]
        assert query.camids.tolist() == [6]
        assert gallery.pids.tolist() == [42]
        assert gallery.gallery_junk_pids == frozenset({-1})
