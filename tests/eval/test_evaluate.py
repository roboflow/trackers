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
        ("gt_relpath", "expected_sequence"),
        [
            pytest.param("MOT17-02-FRCNN.txt", "MOT17-02-FRCNN", id="flat-layout"),
            pytest.param("MOT17-02-FRCNN/gt/gt.txt", "MOT17-02-FRCNN", id="mot-layout"),
            pytest.param("gt.txt", "gt", id="bare-gt-file-keeps-stem"),
            pytest.param("MOT17-02-FRCNN/gt.txt", "gt", id="gt-file-outside-gt-dir-keeps-stem"),
            pytest.param("data/sequence-A/gt.txt", "gt", id="unrelated-grandparent-is-ignored"),
        ],
    )
    def test_sequence_name_from_gt_path(self, tmp_path: Path, gt_relpath: str, expected_sequence: str) -> None:
        """A MOT-layout `<seq>/gt/gt.txt` file is named after `<seq>`; every other file keeps its stem."""
        gt_file = tmp_path / gt_relpath
        gt_file.parent.mkdir(parents=True, exist_ok=True)
        gt_file.write_text("1,1,10,10,20,20,1,1\n")
        tracker_file = tmp_path / "tracker.txt"
        tracker_file.write_text("1,1,10,10,20,20,1,1\n")

        result = evaluate_mot_sequence(gt_file, tracker_file)

        assert result.sequence == expected_sequence

    @pytest.mark.parametrize(
        "gt_arg",
        [
            pytest.param("gt/gt.txt", id="relative-inside-sequence-dir"),
            pytest.param("./gt/gt.txt", id="dot-relative"),
            pytest.param("../MOT17-02-FRCNN/gt/gt.txt", id="parent-traversal"),
        ],
    )
    def test_relative_gt_path_is_named_after_its_sequence_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gt_arg: str
    ) -> None:
        """A relative `gt/gt.txt` resolves against the working directory, never to an empty or `..` name."""
        sequence_dir = tmp_path / "MOT17-02-FRCNN"
        (sequence_dir / "gt").mkdir(parents=True)
        (sequence_dir / "gt" / "gt.txt").write_text("1,1,10,10,20,20,1,1\n")
        tracker_file = tmp_path / "tracker.txt"
        tracker_file.write_text("1,1,10,10,20,20,1,1\n")
        monkeypatch.chdir(sequence_dir)

        result = evaluate_mot_sequence(gt_arg, tracker_file)

        assert result.sequence == "MOT17-02-FRCNN"

    def test_mot20_layout_sets_name_and_warns_about_class_config(self, tmp_path: Path) -> None:
        """The same name drives the MOT20 warning and the result, so a `MOT20-01/gt/gt.txt` file does both."""
        gt_file = tmp_path / "MOT20-01" / "gt" / "gt.txt"
        gt_file.parent.mkdir(parents=True)
        gt_file.write_text("1,1,10,10,20,20,1,1\n")
        tracker_file = tmp_path / "tracker.txt"
        tracker_file.write_text("1,1,10,10,20,20,1,1\n")

        with pytest.warns(UserWarning, match="MOT20-01"):
            result = evaluate_mot_sequence(gt_file, tracker_file)

        assert result.sequence == "MOT20-01"

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


class TestEvaluateMOTSequencesNaming:
    """`evaluate_mot_sequences` names each result after the sequence it was discovered under."""

    @pytest.mark.parametrize(
        "gt_template",
        [
            pytest.param("{seq}.txt", id="flat-layout"),
            pytest.param("{seq}/gt/gt.txt", id="mot-layout"),
        ],
    )
    def test_each_result_carries_its_sequence_name(self, tmp_path: Path, gt_template: str) -> None:
        """Every `result.sequences[name].sequence` equals its key, with no per-result rename step left to drift."""
        gt_dir = tmp_path / "gt"
        tracker_dir = tmp_path / "trackers"
        tracker_dir.mkdir()
        for name in ("sequence1", "sequence2"):
            gt_file = gt_dir / gt_template.format(seq=name)
            gt_file.parent.mkdir(parents=True, exist_ok=True)
            gt_file.write_text("1,1,10,10,20,20,1,1\n")
            (tracker_dir / f"{name}.txt").write_text("1,1,10,10,20,20,0.9,1\n")

        result = evaluate_mot_sequences(gt_dir, tracker_dir)

        assert {name: seq_result.sequence for name, seq_result in result.sequences.items()} == {
            "sequence1": "sequence1",
            "sequence2": "sequence2",
        }


class TestEvaluateMOTSequencesDiscovery:
    """Sequence discovery must ignore hidden entries in both supported layouts."""

    @pytest.mark.parametrize(
        ("gt_relpath", "hidden_relpath"),
        [
            pytest.param("sequence1.txt", ".hidden.txt", id="flat-layout"),
            pytest.param("sequence1/gt/gt.txt", ".sequence1-backup/gt/gt.txt", id="mot-layout"),
        ],
    )
    def test_hidden_entries_are_not_treated_as_sequences(
        self,
        tmp_path: Path,
        gt_relpath: str,
        hidden_relpath: str,
    ) -> None:
        """A hidden ground-truth entry is skipped instead of being evaluated as a sequence.

        Backup and sync tools can leave hidden copies next to the sequences (`.sequence1-backup/gt/gt.txt`). Such a copy
        is not a sequence to evaluate, and discovering one aborts the whole benchmark when it has no prediction file.
        """
        gt_dir = tmp_path / "gt"
        for relpath in (gt_relpath, hidden_relpath):
            gt_file = gt_dir / relpath
            gt_file.parent.mkdir(parents=True, exist_ok=True)
            gt_file.write_text("1,1,10,10,20,20,1,1\n")
        tracker_dir = tmp_path / "trackers"
        tracker_dir.mkdir()
        (tracker_dir / "sequence1.txt").write_text("1,1,10,10,20,20,0.9,1\n")

        result = evaluate_mot_sequences(gt_dir, tracker_dir, metrics=["CLEAR"])

        assert list(result.sequences) == ["sequence1"]

    def test_hidden_root_file_does_not_switch_mot_layout_to_flat(self, tmp_path: Path) -> None:
        """A hidden `.txt` file at the root of a MOT-layout `gt_dir` leaves the layout detected as MOT.

        Format detection used to count hidden root files, so `.notes.txt` next to `sequence1/gt/gt.txt` selected the
        flat layout, whose discovery skips hidden files and then found no sequence at all (`ValueError`).
        """
        gt_dir = tmp_path / "gt"
        gt_file = gt_dir / "sequence1" / "gt" / "gt.txt"
        gt_file.parent.mkdir(parents=True)
        gt_file.write_text("1,1,10,10,20,20,1,1\n")
        (gt_dir / ".notes.txt").write_text("scratch\n")
        tracker_dir = tmp_path / "trackers"
        tracker_dir.mkdir()
        (tracker_dir / "sequence1.txt").write_text("1,1,10,10,20,20,0.9,1\n")

        result = evaluate_mot_sequences(gt_dir, tracker_dir, metrics=["CLEAR"])

        assert list(result.sequences) == ["sequence1"]

    def test_seqmap_selects_hidden_mot_sequence(self, tmp_path: Path) -> None:
        """A `seqmap` still evaluates a MOT-layout sequence directory whose name starts with `.`.

        Discovery skips hidden directories, so the seqmap is the documented way to include one. Layout detection must
        keep seeing such a directory as MOT; otherwise a `gt_dir` holding only hidden sequences is read as flat and the
        seqmap entry resolves to a missing `.5.txt`.
        """
        gt_dir = tmp_path / "gt"
        gt_file = gt_dir / ".5" / "gt" / "gt.txt"
        gt_file.parent.mkdir(parents=True)
        gt_file.write_text("1,1,10,10,20,20,1,1\n")
        tracker_dir = tmp_path / "trackers"
        tracker_dir.mkdir()
        (tracker_dir / ".5.txt").write_text("1,1,10,10,20,20,0.9,1\n")
        seqmap = tmp_path / "seqmap.txt"
        seqmap.write_text("name\n.5\n")

        result = evaluate_mot_sequences(gt_dir, tracker_dir, seqmap=seqmap, metrics=["CLEAR"])

        assert list(result.sequences) == [".5"]
