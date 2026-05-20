# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""CLI-level tests for trackers/scripts/tune.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trackers.scripts.tune import add_tune_subparser, run_tune, tune


def _make_parser() -> tuple[argparse.ArgumentParser, argparse._SubParsersAction]:
    """Return a top-level parser with a subparsers group."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    return parser, subparsers


class TestAddTuneSubparser:
    @pytest.fixture
    def minimal_args(self) -> argparse.Namespace:
        """Parsed args with only required flags."""
        parser, subparsers = _make_parser()
        add_tune_subparser(subparsers)
        return parser.parse_args(["tune", "--tracker", "sort", "--gt-dir", "/gt", "--detections-dir", "/det"])

    def test_registers_tune_subcommand(self) -> None:
        """tune subcommand is accessible under the 'tune' name."""
        parser, subparsers = _make_parser()
        add_tune_subparser(subparsers)
        args = parser.parse_args(["tune", "--tracker", "sort", "--gt-dir", "/gt", "--detections-dir", "/det"])
        assert args.func is run_tune

    def test_required_args_parsed(self) -> None:
        """--tracker, --gt-dir, and --detections-dir are required and parsed."""
        parser, subparsers = _make_parser()
        add_tune_subparser(subparsers)
        args = parser.parse_args(
            [
                "tune",
                "--tracker",
                "bytetrack",
                "--gt-dir",
                "/data/gt",
                "--detections-dir",
                "/data/det",
            ]
        )
        assert args.tracker == "bytetrack"
        assert args.gt_dir == Path("/data/gt")
        assert args.detections_dir == Path("/data/det")

    @pytest.mark.parametrize(
        "flag,expected",
        [
            ("objective", "HOTA"),
            ("n_trials", 100),
            ("threshold", 0.5),
            ("seqmap", None),
            ("output", None),
        ],
    )
    def test_optional_defaults(self, minimal_args: argparse.Namespace, flag: str, expected: object) -> None:
        """Optional arguments have correct defaults when omitted."""
        assert getattr(minimal_args, flag) == expected

    def test_metrics_default(self, minimal_args: argparse.Namespace) -> None:
        """--metrics defaults to ['CLEAR'] when not supplied."""
        assert minimal_args.metrics == ["CLEAR"]

    def test_output_flag_short_form(self) -> None:
        """-o is an alias for --output."""
        parser, subparsers = _make_parser()
        add_tune_subparser(subparsers)
        args = parser.parse_args(
            [
                "tune",
                "--tracker",
                "sort",
                "--gt-dir",
                "/gt",
                "--detections-dir",
                "/det",
                "-o",
                "/out/params.json",
            ]
        )
        assert args.output == Path("/out/params.json")


class TestTune:
    def test_returns_1_on_invalid_tracker(self, tmp_path: Path) -> None:
        """Invalid tracker ID causes tune() to return exit code 1."""
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        det_dir = tmp_path / "det"
        det_dir.mkdir()
        result = tune("nonexistent_tracker_xyz", gt_dir, det_dir)
        assert result == 1

    def test_returns_1_on_missing_files(self, tmp_path: Path) -> None:
        """FileNotFoundError from Tuner (missing sequence files) returns exit code 1."""
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        det_dir = tmp_path / "det"
        det_dir.mkdir()
        # bytetrack is registered; empty det_dir → FileNotFoundError via Tuner
        result = tune("bytetrack", gt_dir, det_dir)
        assert result == 1

    def test_returns_1_on_import_error(self, tmp_path: Path) -> None:
        """ImportError (e.g. optuna not installed) causes tune() to return 1."""
        gt_dir = tmp_path / "gt"
        det_dir = tmp_path / "det"
        with patch(
            "trackers.tune.Tuner",
            side_effect=ImportError("optuna is required"),
        ):
            result = tune("bytetrack", gt_dir, det_dir)
        assert result == 1

    def test_returns_0_on_success(self, tmp_path: Path) -> None:
        """tune() returns 0 when Tuner.run() completes without error."""
        gt_dir = tmp_path / "gt"
        det_dir = tmp_path / "det"
        mock_tuner = MagicMock()
        mock_tuner.run.return_value = {"high_thresh": 0.6}
        mock_tuner.study = None
        with patch("trackers.tune.Tuner", return_value=mock_tuner):
            result = tune("bytetrack", gt_dir, det_dir)
        assert result == 0

    def test_writes_json_output_on_success(self, tmp_path: Path) -> None:
        """Best parameters are written to the output JSON file on success."""
        gt_dir = tmp_path / "gt"
        det_dir = tmp_path / "det"
        output_path = tmp_path / "out" / "params.json"
        best = {"high_thresh": 0.6, "match_thresh": 0.8}
        mock_tuner = MagicMock()
        mock_tuner.run.return_value = best
        mock_tuner.study = None
        with patch("trackers.tune.Tuner", return_value=mock_tuner):
            result = tune("bytetrack", gt_dir, det_dir, output=output_path)
        assert result == 0
        assert output_path.exists()
        assert json.loads(output_path.read_text()) == best

    def test_returns_1_on_oserror_writing_output(self, tmp_path: Path) -> None:
        """OSError while writing output file returns exit code 1."""
        gt_dir = tmp_path / "gt"
        det_dir = tmp_path / "det"
        output_path = tmp_path / "params.json"
        mock_tuner = MagicMock()
        mock_tuner.run.return_value = {"high_thresh": 0.6}
        mock_tuner.study = None
        with (
            patch("trackers.tune.Tuner", return_value=mock_tuner),
            patch.object(Path, "write_text", side_effect=OSError("permission denied")),
        ):
            result = tune("bytetrack", gt_dir, det_dir, output=output_path)
        assert result == 1

    def test_returns_1_on_tuner_run_exception(self, tmp_path: Path) -> None:
        """Exception from tuner.run() causes tune() to return exit code 1."""
        gt_dir = tmp_path / "gt"
        det_dir = tmp_path / "det"
        mock_tuner = MagicMock()
        mock_tuner.run.side_effect = RuntimeError("optimization failed")
        with patch("trackers.tune.Tuner", return_value=mock_tuner):
            result = tune("bytetrack", gt_dir, det_dir)
        assert result == 1


class TestRunTune:
    def test_delegates_to_tune_with_namespace_args(self, tmp_path: Path) -> None:
        """run_tune() passes all argparse.Namespace fields to tune() correctly."""
        gt_dir = tmp_path / "gt"
        det_dir = tmp_path / "det"
        output_path = tmp_path / "params.json"
        args = argparse.Namespace(
            tracker="sort",
            gt_dir=gt_dir,
            detections_dir=det_dir,
            objective="MOTA",
            n_trials=50,
            metrics=["CLEAR", "HOTA"],
            threshold=0.3,
            seqmap=None,
            fixed_params=None,
            images_dir=None,
            no_enqueue_defaults=False,
            seed=None,
            output=output_path,
        )
        with patch("trackers.scripts.tune.tune", return_value=0) as mock_tune:
            result = run_tune(args)
        assert result == 0
        mock_tune.assert_called_once_with(
            tracker="sort",
            gt_dir=gt_dir,
            detections_dir=det_dir,
            objective="MOTA",
            n_trials=50,
            metrics=["CLEAR", "HOTA"],
            threshold=0.3,
            seqmap=None,
            fixed_params=None,
            images_dir=None,
            enqueue_defaults=True,
            seed=None,
            output=output_path,
        )
