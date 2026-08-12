# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trackers.eval import (
    aggregate_hota_metrics,
    evaluate_mot_sequence,
    evaluate_multicamera_scene,
    evaluate_multicamera_scenes,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "multicamera"


@pytest.fixture
def sample_mot_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create sample GT and tracker MOT files for testing."""
    gt_content = "1,1,100,200,50,60,1,1\n1,2,150,250,40,50,1,1\n2,1,105,205,50,60,1,1\n"
    tracker_content = "1,10,102,202,50,60,0.9,1\n1,20,152,252,40,50,0.8,1\n2,10,107,207,50,60,0.9,1\n"

    gt_path = tmp_path / "gt.txt"
    tracker_path = tmp_path / "tracker.txt"
    gt_path.write_text(gt_content)
    tracker_path.write_text(tracker_content)

    return gt_path, tracker_path


class TestEvaluateMOTSequence:
    """MOT sequence evaluation: single-metric, multi-metric, and output formats."""

    @pytest.mark.parametrize(
        ("metric", "check_field", "other_metrics"),
        [
            ("HOTA", ("HOTA", "HOTA"), ["CLEAR", "Identity"]),
            ("Identity", ("Identity", "IDF1"), ["CLEAR", "HOTA"]),
            ("CLEAR", ("CLEAR", "MOTA"), ["HOTA", "Identity"]),
        ],
        ids=["hota_only", "identity_only", "clear_only"],
    )
    def test_single_metric(
        self,
        sample_mot_files: tuple[Path, Path],
        metric: str,
        check_field: tuple[str, str],
        other_metrics: list[str],
    ) -> None:
        """Single-metric evaluation returns only the requested metric."""
        gt_path, tracker_path = sample_mot_files
        result = evaluate_mot_sequence(gt_path=gt_path, tracker_path=tracker_path, metrics=[metric])
        attr_name, field_name = check_field
        computed = getattr(result, attr_name)
        assert computed is not None
        assert getattr(computed, field_name) is not None
        if metric == "HOTA":
            assert computed.DetA is not None
            assert computed.AssA is not None
        for other in other_metrics:
            assert getattr(result, other) is None

    def test_all_metrics(self, sample_mot_files: tuple[Path, Path]) -> None:
        """All three metric groups are present when all metrics requested."""
        gt_path, tracker_path = sample_mot_files

        result = evaluate_mot_sequence(
            gt_path=gt_path,
            tracker_path=tracker_path,
            metrics=["CLEAR", "HOTA", "Identity"],
        )

        assert result.CLEAR is not None
        assert result.HOTA is not None
        assert result.Identity is not None

    def test_table_hota_only(self, sample_mot_files: tuple[Path, Path]) -> None:
        """Table() shows HOTA and DetA; MOTA absent when only HOTA computed."""
        gt_path, tracker_path = sample_mot_files

        result = evaluate_mot_sequence(
            gt_path=gt_path,
            tracker_path=tracker_path,
            metrics=["HOTA"],
        )

        table_str = result.table()
        assert "HOTA" in table_str
        assert "DetA" in table_str
        assert "MOTA" not in table_str

    def test_json_hota_only(self, sample_mot_files: tuple[Path, Path]) -> None:
        """Json() includes HOTA fields when only HOTA computed."""
        gt_path, tracker_path = sample_mot_files

        result = evaluate_mot_sequence(
            gt_path=gt_path,
            tracker_path=tracker_path,
            metrics=["HOTA"],
        )

        json_str = result.json()
        assert "HOTA" in json_str
        assert "DetA" in json_str


class TestEvaluateMulticamera:
    """AI City 2024 multi-camera scene evaluation."""

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

    def test_two_scene_nvidia_golden_scene_mean(self) -> None:
        expected = json.loads((FIXTURE_DIR / "expected_results.json").read_text())
        result = evaluate_multicamera_scenes(
            gt_dir=FIXTURE_DIR / "gt",
            tracker_dir=FIXTURE_DIR / "pred",
            scene_camera_map=FIXTURE_DIR / "scene_camera_map.json",
        )

        fields = ("HOTA", "DetA", "AssA", "LocA")
        scene_hota = {name: result.sequences[name].HOTA for name in expected["scenes"]}
        actual = {
            f"{name}.{field}": getattr(metrics, field)
            for name, metrics in scene_hota.items()
            if metrics is not None
            for field in fields
        }
        frozen = {
            f"{name}.{field}": value for name, values in expected["scenes"].items() for field, value in values.items()
        }
        assert actual == pytest.approx(frozen, rel=1e-4, abs=1e-4)

        assert result.aggregation == "scene_mean"
        assert result.aggregate.HOTA is not None
        aggregate = result.aggregate.HOTA
        assert {field: getattr(aggregate, field) for field in fields} == pytest.approx(
            expected["SCENE_MEAN"], rel=1e-4, abs=1e-4
        )
        tp_weighted = aggregate_hota_metrics(
            [
                metrics.to_dict(include_arrays=True, arrays_as_list=False)
                for metrics in scene_hota.values()
                if metrics is not None
            ]
        )
        assert aggregate.HOTA == pytest.approx(0.5 * (actual["scene_a.HOTA"] + actual["scene_b.HOTA"]))
        assert aggregate.HOTA != pytest.approx(tp_weighted["HOTA"], rel=1e-4, abs=1e-4)

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
