# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Per-track appearance feature bank tests."""

from __future__ import annotations

import numpy as np
import pytest

from trackers.core.reid.feature_bank import FeatureBank


class TestFeatureBank:
    """Unit tests for ``FeatureBank`` L2 + EMA behavior."""

    def test_first_update_normalizes_embedding(self) -> None:
        # BoT-SORT normalizes the embedding before storing it.
        bank = FeatureBank(alpha=0.9)
        bank.update(np.array([3.0, 4.0], dtype=np.float32))
        feature = bank.feature
        assert feature is not None
        np.testing.assert_allclose(feature, [0.6, 0.8], atol=1e-6)

    def test_blends_on_unit_sphere(self) -> None:
        # BoT-SORT: EMA on unit vectors, then L2-normalize the blend again.
        bank = FeatureBank(alpha=0.75)
        bank.update(np.array([1.0, 0.0], dtype=np.float32))
        bank.update(np.array([0.0, 1.0], dtype=np.float32))
        feature = bank.feature
        assert feature is not None
        # 0.75*[1,0] + 0.25*[0,1] = [0.75, 0.25], then / ||.||
        expected = np.array([0.75, 0.25], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(feature, expected, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(feature), 1.0, atol=1e-6)

    def test_zero_embedding_is_accepted(self) -> None:
        bank = FeatureBank()
        bank.update(np.zeros(8, dtype=np.float32))
        feature = bank.feature
        assert feature is not None
        np.testing.assert_allclose(feature, 0.0)

    def test_non_finite_embedding_raises(self) -> None:
        bank = FeatureBank()
        with pytest.raises(ValueError, match="finite"):
            bank.update(np.array([1.0, np.nan], dtype=np.float32))
        assert bank.feature is None

    def test_shape_change_raises(self) -> None:
        bank = FeatureBank()
        bank.update(np.array([1.0, 0.0], dtype=np.float32))
        before = bank.feature
        assert before is not None
        with pytest.raises(ValueError, match="shape"):
            bank.update(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        after = bank.feature
        assert after is not None
        np.testing.assert_allclose(after, before)
