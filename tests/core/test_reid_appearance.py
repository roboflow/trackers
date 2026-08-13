# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Appearance similarity and embedding extraction tests."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv

from trackers.core.reid.appearance import (
    appearance_similarity,
    extract_detection_embeddings,
    extract_ground_truth_embeddings,
)


def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)


class _MeanIntensityEncoder:
    """Encoder returning one row per box, tagged with the frame's mean intensity."""

    def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        return np.full((len(detections), 2), float(frame.mean()), dtype=np.float32)


def _write_sequence(root: Path, name: str, rows: list[tuple[int, int, int, int]], frames: int = 3) -> None:
    """Write a MOT sequence where each row is ``(frame, track_id, confidence, class)``."""
    (root / name / "gt").mkdir(parents=True)
    (root / name / "img1").mkdir(parents=True)
    lines = [
        f"{frame},{track_id},0,0,10,10,{confidence},{class_id},1" for frame, track_id, confidence, class_id in rows
    ]
    (root / name / "gt" / "gt.txt").write_text("\n".join(lines))
    for frame_id in range(1, frames + 1):
        cv2.imwrite(str(root / name / "img1" / f"{frame_id:06d}.jpg"), _frame(frame_id))


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


class TestExtractGroundTruthEmbeddings:
    """Unit tests for reading a MOT-format dataset into labeled crop embeddings."""

    def test_identities_are_renumbered_across_sequences(self, tmp_path: Path) -> None:
        # Track 1 in two sequences is two people, so the ids must not collapse.
        rows = [(1, 1, 1, 1), (1, 2, 1, 1), (2, 1, 1, 1), (2, 2, 1, 1)]
        _write_sequence(tmp_path, "seq_a", rows)
        _write_sequence(tmp_path, "seq_b", rows)

        embeddings, ids, frame_ids, sequence_ids = extract_ground_truth_embeddings(_MeanIntensityEncoder(), tmp_path)

        assert embeddings.shape == (8, 2)
        assert set(zip(sequence_ids.tolist(), ids.tolist())) == {(0, 0), (0, 1), (1, 2), (1, 3)}
        np.testing.assert_array_equal(np.unique(frame_ids), [1, 2])

    def test_ignored_rows_and_unwanted_classes_are_dropped(self, tmp_path: Path) -> None:
        _write_sequence(tmp_path, "seq_a", [(1, 1, 1, 1), (1, 2, 0, 1), (1, 3, 1, 7)])

        _, every_class, _, _ = extract_ground_truth_embeddings(_MeanIntensityEncoder(), tmp_path)
        _, pedestrians, _, _ = extract_ground_truth_embeddings(_MeanIntensityEncoder(), tmp_path, keep_classes=(1,))

        assert len(every_class) == 2
        assert len(pedestrians) == 1

    def test_frame_stride_subsamples_frames(self, tmp_path: Path) -> None:
        _write_sequence(tmp_path, "seq_a", [(frame, 1, 1, 1) for frame in (1, 2, 3)])

        _, _, frame_ids, _ = extract_ground_truth_embeddings(_MeanIntensityEncoder(), tmp_path, frame_stride=2)

        np.testing.assert_array_equal(frame_ids, [1, 3])

    def test_frames_without_images_are_skipped(self, tmp_path: Path) -> None:
        _write_sequence(tmp_path, "seq_a", [(frame, 1, 1, 1) for frame in (1, 2, 3)], frames=2)

        _, _, frame_ids, _ = extract_ground_truth_embeddings(_MeanIntensityEncoder(), tmp_path)

        np.testing.assert_array_equal(frame_ids, [1, 2])

    def test_dataset_without_annotations_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=re.escape("gt/gt.txt")):
            extract_ground_truth_embeddings(_MeanIntensityEncoder(), tmp_path)
