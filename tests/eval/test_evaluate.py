# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from trackers.eval import evaluate_mot_sequence, evaluate_mot_sequences
from trackers.eval.mot_classes import MOTClassConfig, MOTClassPreset


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

    @pytest.mark.parametrize(
        ("gt_relpath", "class_config", "expect_warning"),
        [
            pytest.param("MOT20-01.txt", "mot17", True, id="mot20-flat-mot17-warns"),
            pytest.param("MOT20-01/gt/gt.txt", "mot17", True, id="mot20-mot-layout-mot17-warns"),
            pytest.param("MOT20-01.txt", "mot20", False, id="mot20-flat-mot20-silent"),
            pytest.param("MOT20-01/gt/gt.txt", "mot20", False, id="mot20-mot-layout-mot20-silent"),
            pytest.param("MOT20-01.txt", MOTClassConfig(), False, id="mot20-flat-explicit-config-silent"),
            pytest.param("MOT17-02-DPM.txt", "mot17", False, id="mot17-name-silent"),
            pytest.param("sequence1.txt", "mot17", False, id="generic-name-silent"),
        ],
    )
    def test_mot20_warning_depends_on_sequence_name_and_class_config(
        self,
        tmp_path: Path,
        gt_relpath: str,
        class_config: MOTClassPreset | MOTClassConfig,
        expect_warning: bool,
    ) -> None:
        """A MOT20-named sequence warns only when the MOT17 preset is in effect.

        Users scoring MOT20 sequences with the default preset silently mis-handle class 6 (non_mot_vehicle). The warning
        must fire for both flat and MOT-style (`<seq>/gt/gt.txt`) layouts, and stay silent for an explicit `mot20`
        preset, an explicit `MOTClassConfig`, and sequence names that are not MOT20.
        """
        gt_file = tmp_path / "gt" / gt_relpath
        gt_file.parent.mkdir(parents=True)
        gt_file.write_text("1,1,10,10,20,20,1,1\n")
        tracker_file = tmp_path / "tracker.txt"
        tracker_file.write_text("1,1,10,10,20,20,1,1\n")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            evaluate_mot_sequence(gt_file, tracker_file, class_config=class_config)

        mot20_warnings = [w for w in caught if issubclass(w.category, UserWarning) and "MOT20" in str(w.message)]
        assert bool(mot20_warnings) is expect_warning


class TestEvaluateMOTSequencesClassConfig:
    """`class_config` passed to `evaluate_mot_sequences` must reach the per-sequence scoring."""

    @pytest.mark.parametrize(
        "gt_relpath",
        [
            pytest.param("sequence1.txt", id="flat-layout"),
            pytest.param("sequence1/gt/gt.txt", id="mot-layout"),
        ],
    )
    @pytest.mark.parametrize(
        ("class_config", "expected_mota"),
        [
            pytest.param("mot17", 0.0, id="mot17-class6-prediction-is-false-positive"),
            pytest.param("mot20", 1.0, id="mot20-class6-prediction-is-ignored"),
        ],
    )
    def test_class_config_changes_aggregate_mota(
        self,
        tmp_path: Path,
        gt_relpath: str,
        class_config: MOTClassPreset,
        expected_mota: float,
    ) -> None:
        """A prediction on a class-6 region is a false positive under mot17 but ignored under mot20.

        One scored pedestrian (class 1) is tracked correctly. A second prediction overlaps a class-6 ground-truth row:
        MOT17 scores it as a false positive (MOTA 0.0 with one GT), MOT20 treats class 6 as a distractor and removes it
        (MOTA 1.0). If the kwarg were dropped, both presets would give the same value.
        """
        gt_dir = tmp_path / "gt"
        gt_file = gt_dir / gt_relpath
        gt_file.parent.mkdir(parents=True)
        gt_file.write_text("1,1,10,10,20,20,1,1\n1,2,200,200,20,20,1,6\n")
        tracker_dir = tmp_path / "trackers"
        tracker_dir.mkdir()
        (tracker_dir / "sequence1.txt").write_text("1,1,10,10,20,20,0.9,1\n1,2,200,200,20,20,0.9,1\n")

        result = evaluate_mot_sequences(gt_dir, tracker_dir, metrics=["CLEAR"], class_config=class_config)

        assert result.aggregate.CLEAR is not None
        assert result.aggregate.CLEAR.MOTA == pytest.approx(expected_mota)
