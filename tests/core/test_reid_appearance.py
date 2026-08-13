# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Appearance similarity and embedding extraction tests."""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from trackers.core.reid.appearance import appearance_similarity, extract_detection_embeddings


def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)


class TestAppearanceSimilarity:
    """Unit tests for cosine ``appearance_similarity`` and embedding extraction."""

    def test_identical_vectors_are_one(self) -> None:
        similarity = appearance_similarity(
            [np.array([1.0, 0.0], dtype=np.float32)],
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(similarity, [[1.0]], atol=1e-6)

    def test_orthogonal_vectors_are_zero(self) -> None:
        similarity = appearance_similarity(
            [np.array([1.0, 0.0], dtype=np.float32)],
            np.array([[0.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(similarity, [[0.0]], atol=1e-6)

    def test_normalizes_both_sides(self) -> None:
        similarity = appearance_similarity(
            [np.array([3.0, 4.0], dtype=np.float32)],
            np.array([[6.0, 8.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(similarity, [[1.0]], atol=1e-6)

    def test_none_track_yields_zero_row(self) -> None:
        similarity = appearance_similarity(
            [None, np.array([1.0, 0.0], dtype=np.float32)],
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(similarity, [[0.0], [1.0]], atol=1e-6)

    def test_empty_inputs_return_empty_matrix(self) -> None:
        assert appearance_similarity([], np.empty((0, 4), dtype=np.float32)).shape == (0, 0)
        assert appearance_similarity(
            [np.array([1.0, 0.0], dtype=np.float32)],
            np.empty((0, 2), dtype=np.float32),
        ).shape == (1, 0)

    def test_non_finite_detection_rows_raise(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            appearance_similarity(
                [np.array([1.0, 0.0], dtype=np.float32)],
                np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32),
            )

    def test_incompatible_track_dimensions_raise(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            appearance_similarity(
                [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
                np.array([[1.0, 0.0]], dtype=np.float32),
            )

    def test_extract_detection_embeddings_requires_one_row_per_box(self) -> None:
        # Encoder must return embeddings.shape[0] == len(boxes).
        class _WrongLengthEncoder:
            def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
                return np.empty((0, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="rows"):
            extract_detection_embeddings(
                _WrongLengthEncoder(),
                _frame(),
                np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            )
