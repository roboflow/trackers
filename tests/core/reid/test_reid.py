# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""NumPy-only ReID tests (FeatureBank, appearance distance, gallery eval).

These do not require ``trackers[reid]`` and run in every CI job.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Feature bank + appearance distance
# ---------------------------------------------------------------------------


class TestFeatureBank:
    def test_first_update_stores_raw_embedding(self) -> None:
        from trackers.core.reid.feature_bank import FeatureBank

        bank = FeatureBank(alpha=0.9)
        bank.update(np.array([3.0, 4.0], dtype=np.float32))
        feature = bank.feature
        assert feature is not None
        np.testing.assert_allclose(feature, [3.0, 4.0], atol=1e-6)

    def test_second_update_blends_without_renormalizing(self) -> None:
        from trackers.core.reid.feature_bank import FeatureBank

        alpha = 0.75
        bank = FeatureBank(alpha=alpha)
        bank.update(np.array([1.0, 0.0], dtype=np.float32))
        bank.update(np.array([0.0, 1.0], dtype=np.float32))
        feature = bank.feature
        assert feature is not None
        expected = np.array([alpha, 1.0 - alpha], dtype=np.float32)
        np.testing.assert_allclose(feature, expected, atol=1e-6)

    def test_zero_embedding_is_accepted(self) -> None:
        from trackers.core.reid.feature_bank import FeatureBank

        bank = FeatureBank()
        bank.update(np.zeros(8, dtype=np.float32))
        feature = bank.feature
        assert feature is not None
        np.testing.assert_allclose(feature, 0.0)

    def test_non_finite_embedding_raises(self) -> None:
        from trackers.core.reid.feature_bank import FeatureBank

        bank = FeatureBank()
        with pytest.raises(ValueError, match="finite"):
            bank.update(np.array([1.0, np.nan], dtype=np.float32))
        assert not bank.is_initialized

    def test_shape_change_raises(self) -> None:
        from trackers.core.reid.feature_bank import FeatureBank

        bank = FeatureBank()
        bank.update(np.array([1.0, 0.0], dtype=np.float32))
        before = bank.feature
        assert before is not None
        with pytest.raises(ValueError, match="shape"):
            bank.update(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        after = bank.feature
        assert after is not None
        np.testing.assert_allclose(before, after)


class TestAppearanceSimilarity:
    def test_identical_unit_vectors_are_one(self) -> None:
        from trackers.core.reid.appearance import appearance_similarity

        track = np.array([1.0, 0.0], dtype=np.float32)
        dets = np.array([[1.0, 0.0]], dtype=np.float32)
        sim = appearance_similarity([track], dets)
        np.testing.assert_allclose(sim, [[1.0]], atol=1e-6)

    def test_orthogonal_unit_vectors_are_zero(self) -> None:
        from trackers.core.reid.appearance import appearance_similarity

        track = np.array([1.0, 0.0], dtype=np.float32)
        dets = np.array([[0.0, 1.0]], dtype=np.float32)
        sim = appearance_similarity([track], dets)
        np.testing.assert_allclose(sim, [[0.0]], atol=1e-6)

    def test_unnormalized_parallel_vectors_are_one(self) -> None:
        from trackers.core.reid.appearance import appearance_similarity

        track = np.array([3.0, 4.0], dtype=np.float32)
        dets = np.array([[6.0, 8.0]], dtype=np.float32)
        sim = appearance_similarity([track], dets)
        np.testing.assert_allclose(sim, [[1.0]], atol=1e-6)

    def test_none_track_yields_zero_row(self) -> None:
        from trackers.core.reid.appearance import appearance_similarity

        dets = np.array([[1.0, 0.0]], dtype=np.float32)
        sim = appearance_similarity([None, np.array([1.0, 0.0], dtype=np.float32)], dets)
        np.testing.assert_allclose(sim[0], [0.0], atol=1e-6)
        np.testing.assert_allclose(sim[1], [1.0], atol=1e-6)

    def test_empty_inputs_return_empty_matrix(self) -> None:
        from trackers.core.reid.appearance import appearance_similarity

        sim = appearance_similarity([], np.empty((0, 4), dtype=np.float32))
        assert sim.shape == (0, 0)

        sim = appearance_similarity(
            [np.array([1.0, 0.0], dtype=np.float32)],
            np.empty((0, 2), dtype=np.float32),
        )
        assert sim.shape == (1, 0)

    def test_non_finite_detection_rows_raise(self) -> None:
        from trackers.core.reid.appearance import appearance_similarity

        with pytest.raises(ValueError, match="finite"):
            appearance_similarity(
                [np.array([1.0, 0.0], dtype=np.float32)],
                np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32),
            )

    def test_incompatible_track_dimensions_raise(self) -> None:
        from trackers.core.reid.appearance import appearance_similarity

        with pytest.raises(ValueError, match="dim"):
            appearance_similarity(
                [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
                np.array([[1.0, 0.0]], dtype=np.float32),
            )


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

            def extract_features_from_paths(
                self,
                image_paths: list[str],
                *,
                batch_size: int = 64,
                normalize: bool = True,
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
            def extract_features_from_paths(
                self,
                image_paths: list[str],
                *,
                batch_size: int = 64,
                normalize: bool = True,
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
