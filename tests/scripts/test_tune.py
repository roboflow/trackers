# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""CLI-level tests for trackers/cli/tune.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from trackers.cli.__main__ import cli
from trackers.cli.tune import tune


class TestTuneCommand:
    """Click CLI surface for the tune subcommand."""

    def test_missing_required_args_exits_nonzero(self) -> None:
        """tune without required flags exits with a non-zero code."""
        runner = CliRunner()
        result = runner.invoke(cli, ["tune"])
        assert result.exit_code != 0

    def test_tracker_flag_accepted(self, tmp_path: Path) -> None:
        """--tracker, --gt-dir, --detections-dir are parsed without error when Tuner raises."""
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        det_dir = tmp_path / "det"
        det_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["tune", "--tracker", "bytetrack", "--gt-dir", str(gt_dir), "--detections-dir", str(det_dir)],
        )
        # bytetrack with empty dirs → exit 1 from tune(), not a click error
        assert result.exit_code in (0, 1)

    @pytest.mark.parametrize("objective", ["MOTA", "HOTA", "IDF1"])
    def test_objective_choices_accepted(self, tmp_path: Path, objective: str) -> None:
        """Valid --objective values are accepted (exit comes from Tuner, not click)."""
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        det_dir = tmp_path / "det"
        det_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "tune",
                "--tracker",
                "bytetrack",
                "--gt-dir",
                str(gt_dir),
                "--detections-dir",
                str(det_dir),
                "--objective",
                objective,
            ],
        )
        assert result.exit_code in (0, 1)

    def test_invalid_objective_rejected(self, tmp_path: Path) -> None:
        """Unknown --objective value exits with click usage error (code 2)."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "tune",
                "--tracker",
                "bytetrack",
                "--gt-dir",
                str(tmp_path),
                "--detections-dir",
                str(tmp_path),
                "--objective",
                "UNKNOWN",
            ],
        )
        assert result.exit_code == 2

    def test_n_trials_flag(self, tmp_path: Path) -> None:
        """--n-trials is forwarded to tune()."""
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        det_dir = tmp_path / "det"
        det_dir.mkdir()
        mock_tuner = MagicMock()
        mock_tuner.run.return_value = {"high_thresh": 0.6}
        mock_tuner.study = None
        runner = CliRunner()
        with patch("trackers.tune.Tuner", return_value=mock_tuner) as mock_cls:
            runner.invoke(
                cli,
                [
                    "tune",
                    "--tracker",
                    "bytetrack",
                    "--gt-dir",
                    str(gt_dir),
                    "--detections-dir",
                    str(det_dir),
                    "--n-trials",
                    "50",
                ],
            )
        _, kwargs = mock_cls.call_args
        assert kwargs.get("n_trials") == 50

    def test_output_flag_writes_json(self, tmp_path: Path) -> None:
        """-o writes best parameters to a JSON file."""
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        det_dir = tmp_path / "det"
        det_dir.mkdir()
        output_path = tmp_path / "params.json"
        best = {"high_thresh": 0.6}
        mock_tuner = MagicMock()
        mock_tuner.run.return_value = best
        mock_tuner.study = None
        runner = CliRunner()
        with patch("trackers.tune.Tuner", return_value=mock_tuner):
            result = runner.invoke(
                cli,
                [
                    "tune",
                    "--tracker",
                    "bytetrack",
                    "--gt-dir",
                    str(gt_dir),
                    "--detections-dir",
                    str(det_dir),
                    "-o",
                    str(output_path),
                ],
            )
        assert result.exit_code == 0
        assert output_path.exists()
        assert json.loads(output_path.read_text()) == best


class TestTune:
    """Unit tests for the tune() helper function (no CLI layer)."""

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
        result = tune("bytetrack", gt_dir, det_dir)
        assert result == 1

    def test_returns_1_on_import_error(self, tmp_path: Path) -> None:
        """ImportError (e.g. optuna not installed) causes tune() to return 1."""
        gt_dir = tmp_path / "gt"
        det_dir = tmp_path / "det"
        with patch("trackers.tune.Tuner", side_effect=ImportError("optuna is required")):
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
