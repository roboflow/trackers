# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trackers.eval import (
    aggregate_hota_metrics,
    evaluate_mot_sequence,
    evaluate_multicamera_scene,
    evaluate_multicamera_scenes,
)
from trackers.eval import evaluate as evaluate_module
from trackers.eval.hota import ALPHA_THRESHOLDS
from trackers.io import multicamera as multicamera_module
from trackers.io.multicamera import _euclidean_similarity

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "multicamera"
OFFICIAL_SCENES = tuple(f"scene_{index:03d}" for index in range(61, 91))
OFFICIAL_SCENE_MAP_SHA256 = "f1f1c873d40a50e075d85a364554d902968b2c6717f16ebd5e63d43300f50bac"


def _write_multicamera_benchmark_files(
    root: Path,
    scene_names: tuple[str, ...],
) -> tuple[Path, Path]:
    """Create minimal valid GT and prediction files for benchmark-contract tests."""
    gt_dir = root / "gt"
    tracker_dir = root / "pred"
    tracker_dir.mkdir(parents=True)
    row = "1 1 0 0 0 1 1 0 0\n"
    for scene_name in scene_names:
        scene_dir = gt_dir / scene_name
        scene_dir.mkdir(parents=True)
        (scene_dir / "ground_truth.txt").write_text(row)
        (tracker_dir / f"{scene_name}.txt").write_text(row)
    return gt_dir, tracker_dir


def _modified_official_scene_map() -> dict[str, list[int]]:
    """Return a full-name-set map with intentionally noncanonical cameras."""
    return {scene_name: [1] for scene_name in OFFICIAL_SCENES}


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

    def test_tier1_nvidia_goldens(self) -> None:
        expected = json.loads((FIXTURE_DIR / "expected_results.json").read_text())
        result = evaluate_multicamera_scenes(
            gt_dir=FIXTURE_DIR / "gt",
            tracker_dir=FIXTURE_DIR / "pred",
            scene_camera_map=FIXTURE_DIR / "scene_camera_map.json",
            allow_partial=True,
        )
        assert result.aggregation == "scene_mean"
        for scene_name, values in expected["scenes"].items():
            hota = result.sequences[scene_name].HOTA
            assert hota is not None
            for field, value in values.items():
                assert getattr(hota, field) == pytest.approx(value, rel=1e-4, abs=1e-4)
        agg = result.aggregate.HOTA
        assert agg is not None
        for field, value in expected["SCENE_MEAN"].items():
            assert getattr(agg, field) == pytest.approx(value, rel=1e-4, abs=1e-4)

    def test_scene_mean_differs_from_tp_weighted(self) -> None:
        result = evaluate_multicamera_scenes(
            gt_dir=FIXTURE_DIR / "gt",
            tracker_dir=FIXTURE_DIR / "pred",
            scene_camera_map=FIXTURE_DIR / "scene_camera_map.json",
            allow_partial=True,
        )
        per_scene = [
            seq.HOTA.to_dict(include_arrays=True, arrays_as_list=False)
            for seq in result.sequences.values()
            if seq.HOTA is not None
        ]
        tp_weighted = aggregate_hota_metrics(per_scene)
        assert result.aggregate.HOTA is not None
        assert result.sequences["scene_a"].HOTA is not None
        assert result.sequences["scene_b"].HOTA is not None
        assert result.aggregate.HOTA.HOTA == pytest.approx(
            0.5 * (result.sequences["scene_a"].HOTA.HOTA + result.sequences["scene_b"].HOTA.HOTA)
        )
        assert result.aggregate.HOTA.HOTA != pytest.approx(tp_weighted["HOTA"], rel=1e-4, abs=1e-4)

    def test_alpha_thresholds_match_hota_module(self) -> None:
        assert len(ALPHA_THRESHOLDS) == 19
        assert ALPHA_THRESHOLDS == pytest.approx(np.arange(0.05, 0.99, 0.05))

    def test_official_scene_map_digest_is_pinned(self) -> None:
        """Complete benchmark identity pins the canonical NVIDIA map bytes."""
        assert evaluate_module._OFFICIAL_SCENE_CAMERA_MAP_SHA256 == OFFICIAL_SCENE_MAP_SHA256

    def test_euclidean_similarity_contract(self) -> None:
        origin = np.array([[0.0, 0.0]])
        assert _euclidean_similarity(origin, np.array([[0.0, 0.0]]), 2.0)[0, 0] == 1.0
        assert _euclidean_similarity(origin, np.array([[1.0, 0.0]]), 2.0)[0, 0] == pytest.approx(0.5)
        assert _euclidean_similarity(origin, np.array([[2.0, 0.0]]), 2.0)[0, 0] == 0.0
        assert _euclidean_similarity(origin, np.array([[2.5, 0.0]]), 2.0)[0, 0] == 0.0
        with pytest.raises(ValueError, match="zero_distance"):
            _euclidean_similarity(origin, origin, 0.0)

    def test_missing_prediction_file_raises(self, tmp_path: Path) -> None:
        gt_dir = tmp_path / "gt"
        (gt_dir / "scene_a").mkdir(parents=True)
        (gt_dir / "scene_a" / "ground_truth.txt").write_text("1 1 0 0 0 1 1 0 0\n")
        tracker_dir = tmp_path / "pred"
        tracker_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Tracker file not found"):
            evaluate_multicamera_scenes(
                gt_dir=gt_dir,
                tracker_dir=tracker_dir,
                scene_camera_map={"scene_a": [1]},
                allow_partial=True,
            )

    def test_partial_scene_map_rejected_by_default(self, tmp_path: Path) -> None:
        """A partial official split cannot masquerade as a complete benchmark."""
        gt_dir, tracker_dir = _write_multicamera_benchmark_files(tmp_path, ("scene_061",))

        with pytest.raises(ValueError, match=r"partial|complete|canonical"):
            evaluate_multicamera_scenes(
                gt_dir=gt_dir,
                tracker_dir=tracker_dir,
                scene_camera_map={"scene_061": [1]},
            )

    def test_scene_subset_rejected_by_default(self, tmp_path: Path) -> None:
        """Selecting a subset requires explicit partial-evaluation opt-in."""
        gt_dir, tracker_dir = _write_multicamera_benchmark_files(tmp_path, ("scene_a",))

        with pytest.raises(ValueError, match=r"partial|complete|allow_partial"):
            evaluate_multicamera_scenes(
                gt_dir=gt_dir,
                tracker_dir=tracker_dir,
                scene_camera_map={"scene_a": [1], "scene_b": [2]},
                scenes=["scene_a"],
            )

    def test_explicit_partial_evaluation_serializes_coverage(self, tmp_path: Path) -> None:
        """Opted-in subsets carry complete serialized coverage metadata."""
        gt_dir, tracker_dir = _write_multicamera_benchmark_files(tmp_path, ("scene_061",))

        result = evaluate_multicamera_scenes(
            gt_dir=gt_dir,
            tracker_dir=tracker_dir,
            scene_camera_map={"scene_061": [1]},
            allow_partial=True,
        )
        payload = json.loads(result.json())

        assert result.aggregate.sequence == "PARTIAL_SCENE_MEAN"
        assert payload["coverage"] == {
            "benchmark": "AI City Challenge 2024",
            "split": "test",
            "protocol": "MTMC_Tracking_2024",
            "file_format": "aicity-2024",
            "canonical_scene_camera_map_sha256": evaluate_module._OFFICIAL_SCENE_CAMERA_MAP_SEMANTIC_SHA256,
            "scene_camera_map_sha256": evaluate_module._scene_camera_map_sha256({"scene_061": [1]}),
            "expected_scenes": list(OFFICIAL_SCENES),
            "evaluated_scenes": ["scene_061"],
            "missing_scenes": list(OFFICIAL_SCENES[1:]),
            "complete": False,
        }

    def test_canonical_subset_round_trips_partial_coverage(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        camera_map = _modified_official_scene_map()
        monkeypatch.setattr(
            evaluate_module,
            "_OFFICIAL_SCENE_CAMERA_MAP_SEMANTIC_SHA256",
            evaluate_module._scene_camera_map_sha256(camera_map),
        )
        gt_dir, tracker_dir = _write_multicamera_benchmark_files(tmp_path, ("scene_061",))

        result = evaluate_multicamera_scenes(
            gt_dir=gt_dir,
            tracker_dir=tracker_dir,
            scene_camera_map=camera_map,
            scenes=["scene_061"],
            allow_partial=True,
        )
        restored = type(result).from_dict(json.loads(result.json()))

        assert restored.coverage is not None
        assert restored.aggregate.sequence == "PARTIAL_SCENE_MEAN"
        assert restored.coverage.evaluated_scenes == ["scene_061"]
        assert restored.coverage.missing_scenes == list(OFFICIAL_SCENES[1:])
        assert restored.coverage.complete is False

    def test_canonical_map_order_does_not_change_completeness(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        camera_map = _modified_official_scene_map()
        reversed_map = dict(reversed(camera_map.items()))
        monkeypatch.setattr(
            evaluate_module,
            "_OFFICIAL_SCENE_CAMERA_MAP_SEMANTIC_SHA256",
            evaluate_module._scene_camera_map_sha256(camera_map),
        )
        gt_dir, tracker_dir = _write_multicamera_benchmark_files(tmp_path, OFFICIAL_SCENES)

        result = evaluate_multicamera_scenes(
            gt_dir=gt_dir,
            tracker_dir=tracker_dir,
            scene_camera_map=reversed_map,
        )

        assert result.coverage is not None
        assert result.coverage.complete is True
        assert list(result.sequences) == list(OFFICIAL_SCENES)
        assert result.coverage.canonical_scene_camera_map_sha256 == result.coverage.scene_camera_map_sha256

    def test_complete_name_set_requires_canonical_map_identity(self, tmp_path: Path) -> None:
        """Official scene names alone cannot authenticate a modified camera map."""
        gt_dir, tracker_dir = _write_multicamera_benchmark_files(tmp_path, OFFICIAL_SCENES)
        modified_map = _modified_official_scene_map()

        with pytest.raises(ValueError, match=r"canonical|digest|camera map"):
            evaluate_multicamera_scenes(
                gt_dir=gt_dir,
                tracker_dir=tracker_dir,
                scene_camera_map=modified_map,
            )

    def test_scene_path_traversal_rejected_before_file_access(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"scene|path|traversal"):
            evaluate_multicamera_scenes(
                gt_dir=tmp_path / "gt",
                tracker_dir=tmp_path / "pred",
                scene_camera_map={"../outside": [1]},
                allow_partial=True,
            )

    @pytest.mark.parametrize("zero_distance", [0.0, -1.0, float("nan"), float("inf")])
    def test_public_zero_distance_rejected_before_file_access(
        self,
        tmp_path: Path,
        zero_distance: float,
    ) -> None:
        with pytest.raises(ValueError, match="zero_distance"):
            evaluate_multicamera_scene(
                scene="scene_a",
                gt_path=tmp_path / "missing-gt.txt",
                tracker_path=tmp_path / "missing-pred.txt",
                camera_ids=[1],
                zero_distance=zero_distance,
            )

    def test_multiscene_zero_distance_rejected_before_file_access(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="zero_distance"):
            evaluate_multicamera_scenes(
                gt_dir=tmp_path / "missing-gt",
                tracker_dir=tmp_path / "missing-pred",
                scene_camera_map={"scene_a": [1]},
                zero_distance=float("nan"),
                allow_partial=True,
            )

    @pytest.mark.parametrize(
        ("limit_name", "gt_rows", "pred_rows"),
        [
            pytest.param(
                "MAX_FRAME_PAIR_COUNT",
                [(0, 1), (0, 2), (0, 3)],
                [(0, 11), (0, 12)],
                id="frame-pair-limit-plus-one",
            ),
            pytest.param(
                "MAX_IDENTITY_PAIR_COUNT",
                [(0, 1), (1, 2), (2, 3)],
                [(0, 11), (1, 12)],
                id="identity-pair-limit-plus-one",
            ),
        ],
    )
    def test_public_allocation_guards_reject_limit_plus_one(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        limit_name: str,
        gt_rows: list[tuple[int, int]],
        pred_rows: list[tuple[int, int]],
    ) -> None:
        monkeypatch.setattr(multicamera_module, limit_name, 4)
        gt = tmp_path / "gt.txt"
        pred = tmp_path / "pred.txt"
        gt.write_text("".join(f"1 {object_id} {frame} 0 0 1 1 0 0\n" for frame, object_id in gt_rows))
        pred.write_text("".join(f"1 {object_id} {frame} 0 0 1 1 0 0\n" for frame, object_id in pred_rows))

        with pytest.raises(ValueError, match=limit_name):
            evaluate_multicamera_scene("bounded", gt, pred, camera_ids=[1])

    @pytest.mark.parametrize("limit_name", ["MAX_FRAME_PAIR_COUNT", "MAX_IDENTITY_PAIR_COUNT"])
    def test_public_allocation_guards_accept_exact_limit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        limit_name: str,
    ) -> None:
        monkeypatch.setattr(multicamera_module, limit_name, 4)
        rows = "1 1 0 0 0 1 1 0 0\n1 2 0 0 0 1 1 1 0\n"
        gt = tmp_path / "gt.txt"
        pred = tmp_path / "pred.txt"
        gt.write_text(rows)
        pred.write_text(rows)

        result = evaluate_multicamera_scene("bounded", gt, pred, camera_ids=[1])

        assert result.HOTA is not None
        assert result.HOTA.HOTA == pytest.approx(1.0)

    def test_scene_name_not_from_gt_stem(self) -> None:
        result = evaluate_multicamera_scene(
            scene="scene_a",
            gt_path=FIXTURE_DIR / "gt" / "scene_a" / "ground_truth.txt",
            tracker_path=FIXTURE_DIR / "pred" / "scene_a.txt",
            camera_ids=[1, 2],
        )
        assert result.sequence == "scene_a"
        assert result.sequence != "ground_truth"

    def test_perfect_scene_hota_one(self, tmp_path: Path) -> None:
        content = "1 1 0 0 0 1 1 0 0\n1 2 0 0 0 1 1 1 0\n"
        gt = tmp_path / "gt.txt"
        pred = tmp_path / "pred.txt"
        gt.write_text(content)
        pred.write_text(content)
        result = evaluate_multicamera_scene(
            scene="perfect",
            gt_path=gt,
            tracker_path=pred,
            camera_ids=[1],
        )
        assert result.HOTA is not None
        assert result.HOTA.HOTA == pytest.approx(1.0)

    def test_id_swap_degrades_assa(self, tmp_path: Path) -> None:
        gt = tmp_path / "gt.txt"
        pred = tmp_path / "pred.txt"
        gt.write_text("1 1 0 0 0 1 1 0 0\n1 2 0 0 0 1 1 1 0\n1 1 1 0 0 1 1 0 0\n1 2 1 0 0 1 1 1 0\n")
        pred.write_text("1 1 0 0 0 1 1 0 0\n1 2 0 0 0 1 1 1 0\n1 2 1 0 0 1 1 0 0\n1 1 1 0 0 1 1 1 0\n")
        perfect = evaluate_multicamera_scene("p", gt, gt, camera_ids=[1])
        swapped = evaluate_multicamera_scene("s", gt, pred, camera_ids=[1])
        assert perfect.HOTA is not None and swapped.HOTA is not None
        assert swapped.HOTA.AssA < perfect.HOTA.AssA

    def test_duplicate_rows_across_cameras_collapse(self, tmp_path: Path) -> None:
        gt = tmp_path / "gt.txt"
        pred = tmp_path / "pred.txt"
        gt.write_text("1 10 0 0 0 1 1 0 0\n2 10 0 0 0 1 1 5 0\n")
        pred.write_text("1 10 0 0 0 1 1 0 0\n2 10 0 0 0 1 1 5 0\n")
        result = evaluate_multicamera_scene("dup", gt, pred, camera_ids=[1, 2])
        assert result.HOTA is not None
        assert result.HOTA.HOTA_TP == 19  # one det x 19 alphas
