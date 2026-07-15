# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from trackers.core.reid.distance import appearance_similarity, sanitize_embedding_matrix
from trackers.core.reid.feature_bank import FeatureBank


class TestFeatureBank:
    def test_first_update_normalizes(self) -> None:
        bank = FeatureBank(alpha=0.9)
        assert bank.update(np.array([3.0, 4.0], dtype=np.float32))
        feature = bank.feature
        assert feature is not None
        np.testing.assert_allclose(np.linalg.norm(feature), 1.0, atol=1e-6)

    def test_zero_norm_embedding_is_skipped(self) -> None:
        bank = FeatureBank()
        assert bank.update(np.zeros(8, dtype=np.float32)) is False
        assert not bank.is_initialized

    def test_non_finite_embedding_is_skipped(self) -> None:
        bank = FeatureBank()
        assert bank.update(np.array([1.0, np.nan], dtype=np.float32)) is False
        assert not bank.is_initialized

    def test_shape_change_is_rejected_and_state_preserved(self) -> None:
        bank = FeatureBank()
        bank.update(np.array([1.0, 0.0], dtype=np.float32))
        before = bank.feature
        assert before is not None
        assert bank.update(np.array([1.0, 0.0, 0.0], dtype=np.float32)) is False
        after = bank.feature
        assert after is not None
        assert after.shape == before.shape
        np.testing.assert_allclose(before, after)

    def test_reset_clears_state(self) -> None:
        bank = FeatureBank()
        bank.update(np.array([1.0, 0.0], dtype=np.float32))
        bank.reset()
        assert not bank.is_initialized


class TestAppearanceSimilarity:
    def test_non_finite_detection_rows_are_sanitized(self) -> None:
        track_feats = [np.array([1.0, 0.0], dtype=np.float32)]
        det_embeddings = np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32)
        sim = appearance_similarity(track_feats, det_embeddings)
        assert np.isfinite(sim).all()
        assert sim[0, 0] == pytest.approx(1.0)
        assert sim[0, 1] == pytest.approx(0.0)

    def test_rejects_non_2d_detection_matrix(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            sanitize_embedding_matrix(np.array([1.0, 0.0], dtype=np.float32))

    def test_skips_incompatible_track_dimensions(self) -> None:
        track_feats = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
        det_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        sim = appearance_similarity(track_feats, det_embeddings)
        assert sim.shape == (1, 1)
        assert sim[0, 0] == pytest.approx(0.0)
