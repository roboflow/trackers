# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for AI City 2024 multicamera file preparation semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trackers.io import load_multicamera_file, load_scene_camera_map
from trackers.io.multicamera import _euclidean_similarity, _prepare_multicamera_files

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "multicamera"


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


class TestLoadMulticameraFile:
    """Parser contract for AI City 2024 9-column text files."""

    def test_parses_nine_columns_and_filters_cameras(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "rows.txt",
            "1 10 0 11 12 13 14 1.5 2.5\n2 20 1 21 22 23 24 3.5 4.5\n",
        )

        data = load_multicamera_file(path, camera_ids=[2])

        np.testing.assert_allclose(data, [[2, 20, 1, 21, 22, 23, 24, 3.5, 4.5]])

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            ("", "empty"),
            ("   \n  \n", "empty"),
            ("1 1 0 0 0 1 1 0\n", "Expected 9 columns"),
            ("1 1 0 0 0 1 1 abc 0\n", "Invalid AI City"),
            ("1 1 0 0 0 1 1 nan 0\n", "non-finite"),
            ("1 1 1.5 0 0 1 1 0 0\n", "non-negative integers"),
        ],
    )
    def test_malformed_input(self, tmp_path: Path, content: str, match: str) -> None:
        path = _write(tmp_path / "bad.txt", content)
        with pytest.raises(ValueError, match=match):
            load_multicamera_file(path)


class TestPrepareSemantics:
    """AI City 2024 preparation rules that carry NVIDIA parity."""

    def test_camera_filter_precedes_keep_first_dedup(self, tmp_path: Path) -> None:
        gt = _write(
            tmp_path / "gt.txt",
            "99 10 0 0 0 1 1 9.0 9.0\n1 10 0 0 0 1 1 0.0 0.0\n",
        )
        pred = _write(tmp_path / "pred.txt", "1 10 0 0 0 1 1 0.0 0.0\n")
        seq = _prepare_multicamera_files(
            gt,
            pred,
            file_format="aicity-2024",
            camera_ids=[1],
            zero_distance=2.0,
        )

        assert len(seq.gt_ids[0]) == 1
        assert seq.similarity_scores[0][0, 0] == pytest.approx(1.0)

    def test_camera_filter_applies_to_predictions(self, tmp_path: Path) -> None:
        gt = _write(tmp_path / "gt.txt", "1 1 0 0 0 1 1 0.0 0.0\n")
        pred = _write(
            tmp_path / "pred.txt",
            "1 1 0 0 0 1 1 0.0 0.0\n99 2 0 0 0 1 1 5.0 5.0\n",
        )
        seq = _prepare_multicamera_files(
            gt,
            pred,
            file_format="aicity-2024",
            camera_ids=[1],
            zero_distance=2.0,
        )

        assert len(seq.tracker_ids[0]) == 1

    def test_half_to_even_rounding_reaches_similarity(self, tmp_path: Path) -> None:
        # 0.1245 rounds to 0.124 half-to-even but to 0.125 half-up.
        gt = _write(tmp_path / "gt.txt", "1 1 0 0 0 1 1 0.0 0.0\n")
        pred = _write(tmp_path / "pred.txt", "1 1 0 0 0 1 1 0.1245 0.0\n")
        seq = _prepare_multicamera_files(
            gt,
            pred,
            file_format="aicity-2024",
            camera_ids=[1],
            zero_distance=2.0,
        )

        assert seq.similarity_scores[0][0, 0] == pytest.approx(1.0 - 0.124 / 2.0)

    def test_frame_offset_preserves_gap_entries(self, tmp_path: Path) -> None:
        data = _write(
            tmp_path / "data.txt",
            "1 1 0 0 0 1 1 0 0\n1 1 2 0 0 1 1 0 0\n",
        )
        seq = _prepare_multicamera_files(
            data,
            data,
            file_format="aicity-2024",
            camera_ids=[1],
            zero_distance=2.0,
        )

        assert len(seq.gt_ids) == 3
        assert [len(ids) for ids in seq.gt_ids] == [1, 0, 1]
        assert seq.similarity_scores[1].shape == (0, 0)


class TestEuclideanSimilarity:
    """World-plane similarity ``max(0, 1 - dist / zero_distance)``."""

    @pytest.mark.parametrize(
        ("distance", "expected"),
        [
            (0.0, 1.0),
            (1.0, 0.5),
            (2.0, 0.0),
            (3.0, 0.0),
        ],
    )
    def test_distance_thresholds(self, distance: float, expected: float) -> None:
        origin = np.array([[0.0, 0.0]])
        point = np.array([[distance, 0.0]])

        similarity = _euclidean_similarity(origin, point, zero_distance=2.0)

        assert similarity[0, 0] == pytest.approx(expected)

    def test_alpha_boundary_is_exact_at_world_plane_magnitude(self) -> None:
        # A Gram-matrix expansion returns 0.49999999997 here and fails the 0.5 alpha threshold.
        gt = np.array([[373.553, -494.735]])
        pred = np.array([[374.553, -494.735]])

        assert _euclidean_similarity(gt, pred, zero_distance=2.0)[0, 0] == 0.5


class TestSceneCameraMap:
    """NVIDIA ``scene_name_2_cam_id`` JSON loader."""

    def test_load_scene_camera_map(self) -> None:
        mapping = load_scene_camera_map(FIXTURE_DIR / "scene_camera_map.json")
        assert mapping["scene_a"] == [1, 2]
        assert mapping["scene_b"] == [3, 4]
