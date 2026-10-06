# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import pytest

from trackers.eval import evaluate_mot_sequence


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


class TestEvaluateMOTSequenceUnscorableGroundTruth:
    """Ground truth that filters down to nothing is reported instead of scored as zeros.

    Only pedestrian-class (1) rows marked for consideration are scored, mirroring TrackEval. Ground truth whose rows are
    all dropped by that filter cannot produce a meaningful score, and returning zeros looks like a tracker that found
    nothing rather than a ground-truth file the evaluator cannot use.
    """

    def test_tracker_output_used_as_ground_truth_raises(self, tmp_path: Path) -> None:
        """Files written by `_MOTOutput` carry class -1, so they cannot be reused as ground truth."""
        gt_path = tmp_path / "gt.txt"
        tracker_path = tmp_path / "tracker.txt"
        gt_path.write_text("1,1,100,200,50,60,0.9000,-1,-1,-1\n2,1,105,205,50,60,0.9000,-1,-1,-1\n")
        tracker_path.write_text("1,10,102,202,50,60,0.9,1\n2,10,107,207,50,60,0.9,1\n")

        with pytest.raises(ValueError, match="no scored ground-truth"):
            evaluate_mot_sequence(gt_path=gt_path, tracker_path=tracker_path)

    def test_all_ignored_ground_truth_raises(self, tmp_path: Path) -> None:
        """Ground truth where every row is marked ignored (confidence 0) is also unscorable."""
        gt_path = tmp_path / "gt.txt"
        tracker_path = tmp_path / "tracker.txt"
        gt_path.write_text("1,1,100,200,50,60,0,1\n2,1,105,205,50,60,0,1\n")
        tracker_path.write_text("1,10,102,202,50,60,0.9,1\n")

        with pytest.raises(ValueError, match="no scored ground-truth"):
            evaluate_mot_sequence(gt_path=gt_path, tracker_path=tracker_path)

    def test_partially_filtered_ground_truth_is_scored(self, tmp_path: Path) -> None:
        """A file keeping at least one scored row evaluates normally."""
        gt_path = tmp_path / "gt.txt"
        tracker_path = tmp_path / "tracker.txt"
        gt_path.write_text("1,1,100,200,50,60,1,1\n1,2,150,250,40,50,1,8\n")
        tracker_path.write_text("1,10,102,202,50,60,0.9,1\n")

        result = evaluate_mot_sequence(gt_path=gt_path, tracker_path=tracker_path)

        assert result.CLEAR is not None
