# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""CLI-level tests for trackers/cli/download.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from trackers.cli.__main__ import cli
from trackers.cli.download import _print_available
from trackers.datasets.download import _DEFAULT_CACHE_DIR, _DEFAULT_OUTPUT_DIR


class TestDownloadCommand:
    """Argument parsing and routing for the download subcommand."""

    def test_list_flag_exits_zero(self) -> None:
        """--list prints datasets and exits 0."""
        runner = CliRunner()
        with patch("trackers.cli.download._print_available") as mock_print:
            result = runner.invoke(cli, ["download", "--list"])
        assert result.exit_code == 0
        mock_print.assert_called_once()

    def test_list_takes_precedence_over_dataset(self) -> None:
        """--list wins over positional dataset argument."""
        runner = CliRunner()
        with patch("trackers.cli.download._print_available") as mock_print:
            result = runner.invoke(cli, ["download", "mot17", "--list"])
        assert result.exit_code == 0
        mock_print.assert_called_once()

    def test_missing_dataset_exits_nonzero(self) -> None:
        """No dataset and no --list exits with non-zero code and error message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["download"])
        assert result.exit_code != 0
        assert "Please specify a dataset" in result.output

    def test_dataset_positional_accepted(self) -> None:
        """Dataset positional argument is forwarded to download_dataset."""
        runner = CliRunner()
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            result = runner.invoke(cli, ["download", "mot17"])
        assert result.exit_code == 0
        mock_dl.assert_called_once_with(
            dataset="mot17",
            split=None,
            asset=None,
            output=_DEFAULT_OUTPUT_DIR,
            cache_dir=_DEFAULT_CACHE_DIR,
        )

    @pytest.mark.parametrize(
        "split_arg,expected_splits",
        [
            ("train", ["train"]),
            ("train,val", ["train", "val"]),
            ("train,val,test", ["train", "val", "test"]),
        ],
    )
    def test_split_comma_parsing(self, split_arg: str, expected_splits: list[str]) -> None:
        """--split values are split on commas and whitespace-stripped."""
        runner = CliRunner()
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            result = runner.invoke(cli, ["download", "mot17", "--split", split_arg, "--asset", "annotations"])
        assert result.exit_code == 0
        mock_dl.assert_called_once_with(
            dataset="mot17",
            split=expected_splits,
            asset=["annotations"],
            output=_DEFAULT_OUTPUT_DIR,
            cache_dir=_DEFAULT_CACHE_DIR,
        )

    @pytest.mark.parametrize(
        "asset_arg,expected_assets",
        [
            ("annotations", ["annotations"]),
            ("frames,annotations", ["frames", "annotations"]),
            ("frames,annotations,detections", ["frames", "annotations", "detections"]),
        ],
    )
    def test_asset_comma_parsing(self, asset_arg: str, expected_assets: list[str]) -> None:
        """--asset values are split on commas and whitespace-stripped."""
        runner = CliRunner()
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            result = runner.invoke(cli, ["download", "sportsmot", "--split", "train", "--asset", asset_arg])
        assert result.exit_code == 0
        mock_dl.assert_called_once_with(
            dataset="sportsmot",
            split=["train"],
            asset=expected_assets,
            output=_DEFAULT_OUTPUT_DIR,
            cache_dir=_DEFAULT_CACHE_DIR,
        )

    def test_none_splits_and_assets_when_omitted(self) -> None:
        """When --split and --asset are omitted, None is forwarded."""
        runner = CliRunner()
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            result = runner.invoke(cli, ["download", "mot17"])
        assert result.exit_code == 0
        mock_dl.assert_called_once_with(
            dataset="mot17",
            split=None,
            asset=None,
            output=_DEFAULT_OUTPUT_DIR,
            cache_dir=_DEFAULT_CACHE_DIR,
        )

    def test_output_directory_forwarded(self) -> None:
        """-o value is forwarded to download_dataset."""
        runner = CliRunner()
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            result = runner.invoke(cli, ["download", "mot17", "-o", "/custom/path"])
        assert result.exit_code == 0
        mock_dl.assert_called_once_with(
            dataset="mot17",
            split=None,
            asset=None,
            output="/custom/path",
            cache_dir=_DEFAULT_CACHE_DIR,
        )

    def test_cache_dir_forwarded(self) -> None:
        """--cache-dir value is forwarded to download_dataset."""
        runner = CliRunner()
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            result = runner.invoke(cli, ["download", "mot17", "--cache-dir", "./cache"])
        assert result.exit_code == 0
        mock_dl.assert_called_once_with(
            dataset="mot17",
            split=None,
            asset=None,
            output=_DEFAULT_OUTPUT_DIR,
            cache_dir="./cache",
        )

    def test_exception_from_download_exits_nonzero(self) -> None:
        """Exception from download_dataset is caught and exits non-zero."""
        runner = CliRunner()
        with patch("trackers.datasets.download.download_dataset", side_effect=ValueError("bad dataset")):
            result = runner.invoke(cli, ["download", "mot17"])
        assert result.exit_code != 0

    def test_split_with_spaces_stripped(self) -> None:
        """--split with spaces around commas strips whitespace."""
        runner = CliRunner()
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            result = runner.invoke(cli, ["download", "mot17", "--split", "train , val", "--asset", "annotations"])
        assert result.exit_code == 0
        mock_dl.assert_called_once_with(
            dataset="mot17",
            split=["train", "val"],
            asset=["annotations"],
            output=_DEFAULT_OUTPUT_DIR,
            cache_dir=_DEFAULT_CACHE_DIR,
        )


class TestPrintAvailable:
    """Output of --list."""

    def test_prints_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_print_available runs without raising."""
        _print_available()
        capsys.readouterr()
