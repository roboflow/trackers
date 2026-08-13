# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for AI City 2024 multi-camera evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trackers.eval import (
    MulticameraBenchmarkResult,
    aggregate_hota_metrics,
    evaluate_multicamera_scene,
    evaluate_multicamera_scenes,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "multicamera"
SCENE_MEAN_FIELDS = ("HOTA", "DetA", "AssA", "LocA")


@pytest.fixture(scope="module")
def fixture_result() -> MulticameraBenchmarkResult:
    return evaluate_multicamera_scenes(
        gt_dir=FIXTURE_DIR / "gt",
        tracker_dir=FIXTURE_DIR / "pred",
        scene_camera_map=FIXTURE_DIR / "scene_camera_map.json",
    )


class TestEvaluateMulticameraScene:
    """Single-scene world-plane HOTA."""

    def test_perfect_scene_and_identity_swap(self, tmp_path: Path) -> None:
        gt = tmp_path / "gt.txt"
        pred = tmp_path / "pred.txt"
        gt.write_text("1 1 0 0 0 1 1 0 0\n1 2 0 0 0 1 1 1 0\n1 1 1 0 0 1 1 0 0\n1 2 1 0 0 1 1 1 0\n")
        pred.write_text("1 1 0 0 0 1 1 0 0\n1 2 0 0 0 1 1 1 0\n1 2 1 0 0 1 1 0 0\n1 1 1 0 0 1 1 1 0\n")

        perfect = evaluate_multicamera_scene("perfect", gt, gt, camera_ids=[1])
        swapped = evaluate_multicamera_scene("swapped", gt, pred, camera_ids=[1])

        assert perfect.HOTA is not None and swapped.HOTA is not None
        assert perfect.HOTA.HOTA == pytest.approx(1.0)
        assert swapped.HOTA.AssA < perfect.HOTA.AssA

    def test_empty_prediction_file_scores_zero(self, tmp_path: Path) -> None:
        gt = tmp_path / "gt.txt"
        pred = tmp_path / "pred.txt"
        gt.write_text("1 1 0 0 0 1 1 0 0\n")
        pred.write_text("")

        result = evaluate_multicamera_scene("empty", gt, pred, camera_ids=[1])

        assert result.HOTA is not None
        assert result.HOTA.HOTA == 0.0
        assert result.HOTA.HOTA_TP == 0
        assert result.HOTA.HOTA_FN == 19


class TestEvaluateMulticameraScenes:
    """Benchmark-level evaluation and the unweighted scene mean."""

    def test_nvidia_golden_parity(self, fixture_result: MulticameraBenchmarkResult) -> None:
        expected = json.loads((FIXTURE_DIR / "expected_results.json").read_text())

        actual = {
            f"{name}.{field}": getattr(scene.HOTA, field)
            for name, scene in fixture_result.scenes.items()
            if scene.HOTA is not None
            for field in SCENE_MEAN_FIELDS
        }
        frozen = {
            f"{name}.{field}": value for name, values in expected["scenes"].items() for field, value in values.items()
        }
        assert actual == pytest.approx(frozen, rel=1e-4, abs=1e-4)
        assert fixture_result.aggregate.to_dict() == pytest.approx(expected["SCENE_MEAN"], rel=1e-4, abs=1e-4)

    def test_scene_mean_is_unweighted(self, fixture_result: MulticameraBenchmarkResult) -> None:
        scene_hota = [scene.HOTA for scene in fixture_result.scenes.values() if scene.HOTA is not None]
        tp_weighted = aggregate_hota_metrics(
            [metrics.to_dict(include_arrays=True, arrays_as_list=False) for metrics in scene_hota]
        )

        assert fixture_result.aggregate.HOTA == pytest.approx(sum(m.HOTA for m in scene_hota) / len(scene_hota))
        assert fixture_result.aggregate.HOTA != pytest.approx(tp_weighted["HOTA"], rel=1e-4, abs=1e-4)

    def test_missing_prediction_file_raises(self, tmp_path: Path) -> None:
        gt_dir = tmp_path / "gt"
        (gt_dir / "scene_a").mkdir(parents=True)
        (gt_dir / "scene_a" / "ground_truth.txt").write_text("1 1 0 0 0 1 1 0 0\n")
        tracker_dir = tmp_path / "pred"
        tracker_dir.mkdir()

        with pytest.raises(FileNotFoundError, match=r"Multi-camera file not found: .*scene_a\.txt"):
            evaluate_multicamera_scenes(
                gt_dir=gt_dir,
                tracker_dir=tracker_dir,
                scene_camera_map={"scene_a": [1]},
            )


class TestMulticameraBenchmarkResult:
    """Serialization and rendering of the multi-camera result type."""

    def test_round_trip(self, fixture_result: MulticameraBenchmarkResult, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        fixture_result.save(path)
        restored = MulticameraBenchmarkResult.load(path)

        assert restored.to_dict() == fixture_result.to_dict()
        assert set(restored.scenes) == {"scene_a", "scene_b"}

    def test_aggregate_holds_only_official_fields(self, fixture_result: MulticameraBenchmarkResult) -> None:
        payload = json.loads(fixture_result.json())

        assert tuple(payload["aggregate"]) == SCENE_MEAN_FIELDS
        assert "HOTA_TP" in payload["scenes"]["scene_a"]["HOTA"]

    def test_table_lists_scenes_and_scene_mean(self, fixture_result: MulticameraBenchmarkResult) -> None:
        table = fixture_result.table()

        assert "SCENE_MEAN" in table
        assert "COMBINED" not in table
        assert table.splitlines()[0].split() == ["Sequence", *SCENE_MEAN_FIELDS]
