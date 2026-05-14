# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for trackers/cli/tune.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from trackers.cli.tune import tune


class TestTune:
    def test_returns_1_on_invalid_tracker(self, tmp_path: Path) -> None:
        """Invalid tracker ID causes tune() to return exit code 1."""
        result = tune("nonexistent_tracker_xyz", tmp_path / "gt", tmp_path / "det")
        assert result == 1

    def test_returns_1_on_missing_files(self, tmp_path: Path) -> None:
        """FileNotFoundError from Tuner (empty det_dir) returns exit code 1."""
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        det_dir = tmp_path / "det"
        det_dir.mkdir()
        result = tune("bytetrack", gt_dir, det_dir)
        assert result == 1

    def test_returns_1_on_import_error(self, tmp_path: Path) -> None:
        """ImportError (e.g. optuna not installed) causes tune() to return 1."""
        with patch(
            "trackers.tune.Tuner",
            side_effect=ImportError("optuna is required"),
        ):
            result = tune("bytetrack", tmp_path / "gt", tmp_path / "det")
        assert result == 1

    def test_returns_0_on_success(self, tmp_path: Path) -> None:
        """tune() returns 0 when Tuner.run() completes without error."""
        mock_tuner = MagicMock()
        mock_tuner.run.return_value = {"high_thresh": 0.6}
        mock_tuner.study = None
        with patch("trackers.tune.Tuner", return_value=mock_tuner):
            result = tune("bytetrack", tmp_path / "gt", tmp_path / "det")
        assert result == 0

    def test_writes_json_output_on_success(self, tmp_path: Path) -> None:
        """Best parameters are written to the output JSON file on success."""
        output_path = tmp_path / "out" / "params.json"
        best = {"high_thresh": 0.6, "match_thresh": 0.8}
        mock_tuner = MagicMock()
        mock_tuner.run.return_value = best
        mock_tuner.study = None
        with patch("trackers.tune.Tuner", return_value=mock_tuner):
            result = tune("bytetrack", tmp_path / "gt", tmp_path / "det", output=output_path)
        assert result == 0
        assert output_path.exists()
        assert json.loads(output_path.read_text()) == best

    def test_returns_1_on_oserror_writing_output(self, tmp_path: Path) -> None:
        """OSError while writing output file returns exit code 1."""
        output_path = tmp_path / "params.json"
        mock_tuner = MagicMock()
        mock_tuner.run.return_value = {"high_thresh": 0.6}
        mock_tuner.study = None
        with (
            patch("trackers.tune.Tuner", return_value=mock_tuner),
            patch.object(Path, "write_text", side_effect=OSError("permission denied")),
        ):
            result = tune("bytetrack", tmp_path / "gt", tmp_path / "det", output=output_path)
        assert result == 1

    def test_returns_1_on_tuner_run_exception(self, tmp_path: Path) -> None:
        """Exception from tuner.run() causes tune() to return exit code 1."""
        mock_tuner = MagicMock()
        mock_tuner.run.side_effect = RuntimeError("optimization failed")
        with patch("trackers.tune.Tuner", return_value=mock_tuner):
            result = tune("bytetrack", tmp_path / "gt", tmp_path / "det")
        assert result == 1
