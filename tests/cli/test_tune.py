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

from trackers.cli.tune import TrackerSelection, tune_command


class TestTune:
    def test_returns_1_on_invalid_tracker(self, tmp_path: Path) -> None:
        """Invalid tracker ID causes tune_command() to return exit code 1."""
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        det_dir = tmp_path / "det"
        det_dir.mkdir()
        result = tune_command(TrackerSelection(name="nonexistent_tracker_xyz"), gt_dir, det_dir)
        assert result == 1

    def test_returns_1_on_missing_files(self, tmp_path: Path) -> None:
        """FileNotFoundError from Tuner (missing sequence files) returns exit code 1."""
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        det_dir = tmp_path / "det"
        det_dir.mkdir()
        # bytetrack is registered; empty det_dir → FileNotFoundError via Tuner
        result = tune_command(TrackerSelection(name="bytetrack"), gt_dir, det_dir)
        assert result == 1

    def test_returns_1_on_import_error(self, tmp_path: Path) -> None:
        """ImportError (e.g. optuna not installed) causes tune_command() to return 1."""
        gt_dir = tmp_path / "gt"
        det_dir = tmp_path / "det"
        with patch(
            "trackers.tune.Tuner",
            side_effect=ImportError("optuna is required"),
        ):
            result = tune_command(TrackerSelection(name="bytetrack"), gt_dir, det_dir)
        assert result == 1

    def test_returns_0_on_success(self, tmp_path: Path) -> None:
        """tune_command() returns 0 when Tuner.run() completes without error."""
        gt_dir = tmp_path / "gt"
        det_dir = tmp_path / "det"
        mock_tuner = MagicMock()
        mock_tuner.run.return_value = {"high_thresh": 0.6}
        mock_tuner.study = None
        with patch("trackers.tune.Tuner", return_value=mock_tuner):
            result = tune_command(TrackerSelection(name="bytetrack"), gt_dir, det_dir)
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
            result = tune_command(TrackerSelection(name="bytetrack"), gt_dir, det_dir, output=output_path)
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
            result = tune_command(TrackerSelection(name="bytetrack"), gt_dir, det_dir, output=output_path)
        assert result == 1

    def test_returns_1_on_tuner_run_exception(self, tmp_path: Path) -> None:
        """Exception from tuner.run() causes tune_command() to return exit code 1."""
        gt_dir = tmp_path / "gt"
        det_dir = tmp_path / "det"
        mock_tuner = MagicMock()
        mock_tuner.run.side_effect = RuntimeError("optimization failed")
        with patch("trackers.tune.Tuner", return_value=mock_tuner):
            result = tune_command(TrackerSelection(name="bytetrack"), gt_dir, det_dir)
        assert result == 1


class TestCliInvocation:
    """tune_command() is wired into the jsonargparse CLI with the expected args."""

    @staticmethod
    def _invoke(args: list[str], spy: list[dict]) -> object:
        """Run jsonargparse.CLI() with a recording spy for `tune_command`.

        The spy mirrors the real signature so jsonargparse can introspect it.
        """
        from jsonargparse import CLI

        from trackers.cli.tune import tune_command as real_tune

        def spy_tune(
            tracker: str,
            gt_dir: Path,
            detections_dir: Path,
            objective: str = "HOTA",
            n_trials: int = 100,
            metrics: list[str] | None = None,
            threshold: float = 0.5,
            seqmap: Path | None = None,
            fixed_params: dict | None = None,
            images_dir: Path | None = None,
            enqueue_defaults: bool = True,
            seed: int | None = None,
            output: Path | None = None,
        ) -> int:
            spy.append(
                dict(
                    tracker=tracker,
                    gt_dir=gt_dir,
                    detections_dir=detections_dir,
                    objective=objective,
                    n_trials=n_trials,
                    metrics=metrics,
                    threshold=threshold,
                    seqmap=seqmap,
                    fixed_params=fixed_params,
                    images_dir=images_dir,
                    enqueue_defaults=enqueue_defaults,
                    seed=seed,
                    output=output,
                )
            )
            return 0

        # Copy the docstring so jsonargparse's introspection matches the real function.
        spy_tune.__doc__ = real_tune.__doc__
        return CLI({"tune": spy_tune}, as_positional=False, args=args)

    def test_cli_dispatch_to_tune(self, tmp_path: Path) -> None:
        """jsonargparse.CLI() parses the tune subcommand and forwards args."""
        gt_dir = tmp_path / "gt"
        det_dir = tmp_path / "det"
        spy: list[dict] = []
        result = self._invoke(
            [
                "tune",
                "--tracker",
                "sort",
                "--gt_dir",
                str(gt_dir),
                "--detections_dir",
                str(det_dir),
                "--objective",
                "MOTA",
                "--n_trials",
                "50",
            ],
            spy,
        )
        assert result == 0
        assert len(spy) == 1
        assert spy[0]["tracker"] == "sort"
        assert spy[0]["gt_dir"] == gt_dir
        assert spy[0]["detections_dir"] == det_dir
        assert spy[0]["objective"] == "MOTA"
        assert spy[0]["n_trials"] == 50

    @pytest.mark.parametrize(
        "flag,arg_value,attr,expected",
        [
            ("--objective", "HOTA", "objective", "HOTA"),
            ("--n_trials", "100", "n_trials", 100),
            ("--threshold", "0.5", "threshold", 0.5),
        ],
    )
    def test_cli_defaults(
        self,
        tmp_path: Path,
        flag: str,
        arg_value: str,
        attr: str,
        expected: object,
    ) -> None:
        """Optional flags carry their declared defaults when invoked via CLI."""
        gt_dir = tmp_path / "gt"
        det_dir = tmp_path / "det"
        spy: list[dict] = []
        self._invoke(
            [
                "tune",
                "--tracker",
                "sort",
                "--gt_dir",
                str(gt_dir),
                "--detections_dir",
                str(det_dir),
                flag,
                arg_value,
            ],
            spy,
        )
        assert spy[0][attr] == expected
