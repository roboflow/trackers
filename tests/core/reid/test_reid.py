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
    def test_first_update_normalizes(self) -> None:
        from trackers.core.reid.feature_bank import FeatureBank

        bank = FeatureBank(alpha=0.9)
        bank.update(np.array([3.0, 4.0], dtype=np.float32))
        feature = bank.feature
        assert feature is not None
        np.testing.assert_allclose(np.linalg.norm(feature), 1.0, atol=1e-6)

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
            max_rank=3,
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
            max_rank=5,
        )
        assert metrics.rank1 == pytest.approx(0.0)
        assert metrics.rank5 == pytest.approx(100.0)

    def test_reid_metrics_map_alias(self) -> None:
        from trackers.core.reid.eval.metrics import ReIDMetrics

        metrics = ReIDMetrics(
            mean_average_precision=42.0,
            rank1=1.0,
            rank5=2.0,
            rank10=3.0,
            minp=4.0,
            num_queries=1,
        )
        assert metrics.mean_average_precision == pytest.approx(42.0)


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
