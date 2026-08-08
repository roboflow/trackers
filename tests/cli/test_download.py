# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""CLI-level tests for trackers/cli/download.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from trackers.cli.download import _print_available, download_command
from trackers.datasets.download import _DEFAULT_CACHE_DIR, _DEFAULT_OUTPUT_DIR


class TestDownload:
    """Execution of the download subcommand."""

    def test_list_triggers_print(self) -> None:
        """list_available=True calls _print_available and returns 0."""
        with patch("trackers.cli.download._print_available") as mock_print:
            rc = download_command(list_available=True)
            assert rc == 0
            mock_print.assert_called_once()

    def test_list_takes_precedence_over_dataset(self) -> None:
        """list_available=True wins over dataset argument."""
        with patch("trackers.cli.download._print_available") as mock_print:
            rc = download_command(name="mot17", list_available=True)
            assert rc == 0
            mock_print.assert_called_once()

    def test_missing_dataset_exits_with_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No dataset and no list_available prints error to stderr and returns 1."""
        rc = download_command()
        captured = capsys.readouterr()
        assert rc == 1
        assert "Please specify a dataset" in captured.err

    @pytest.mark.parametrize(
        "split_arg,expected_splits",
        [
            ("train", ["train"]),
            ("train,val", ["train", "val"]),
            ("train,val,test", ["train", "val", "test"]),
        ],
    )
    def test_split_comma_parsing(self, split_arg: str, expected_splits: list[str]) -> None:
        """Split values are split on commas and whitespace-stripped."""
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = download_command(name="mot17", split=split_arg, asset="annotations")
            assert rc == 0
            mock_dl.assert_called_once_with(
                name="mot17",
                split=expected_splits,
                asset=["annotations"],
                output=_DEFAULT_OUTPUT_DIR,
                cache_dir=_DEFAULT_CACHE_DIR,
            )

    @pytest.mark.parametrize(
        "split_arg,expected_splits",
        [
            ("train,", ["train", ""]),
            (",train", ["", "train"]),
            ("train,,val", ["train", "", "val"]),
        ],
    )
    def test_split_comma_parsing_boundary(self, split_arg: str, expected_splits: list[str]) -> None:
        """Split handles malformed comma inputs gracefully."""
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = download_command(name="mot17", split=split_arg, asset="annotations")
            assert rc == 0
            mock_dl.assert_called_once_with(
                name="mot17",
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
        """Asset values are split on commas and whitespace-stripped."""
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = download_command(name="sportsmot", split="train", asset=asset_arg)
            assert rc == 0
            mock_dl.assert_called_once_with(
                name="sportsmot",
                split=["train"],
                asset=expected_assets,
                output=_DEFAULT_OUTPUT_DIR,
                cache_dir=_DEFAULT_CACHE_DIR,
            )

    def test_none_splits_and_assets_when_omitted(self) -> None:
        """When split and asset are omitted, None is forwarded."""
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = download_command(name="mot17")
            assert rc == 0
            mock_dl.assert_called_once_with(
                name="mot17",
                split=None,
                asset=None,
                output=_DEFAULT_OUTPUT_DIR,
                cache_dir=_DEFAULT_CACHE_DIR,
            )

    def test_output_directory_forwarded(self) -> None:
        """Output value is forwarded to download_dataset."""
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = download_command(name="mot17", output="/custom/path")
            assert rc == 0
            mock_dl.assert_called_once_with(
                name="mot17",
                split=None,
                asset=None,
                output="/custom/path",
                cache_dir=_DEFAULT_CACHE_DIR,
            )

    def test_value_error_returns_exit_code(self) -> None:
        """ValueError from download_dataset is caught and returns 1."""
        with patch(
            "trackers.datasets.download.download_dataset",
            side_effect=ValueError("bad dataset"),
        ):
            rc = download_command(name="mot17")
            assert rc == 1

    def test_split_with_spaces_stripped(self) -> None:
        """Split with spaces around commas strips whitespace."""
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = download_command(name="mot17", split="train , val", asset="annotations")
            assert rc == 0
            mock_dl.assert_called_once_with(
                name="mot17",
                split=["train", "val"],
                asset=["annotations"],
                output=_DEFAULT_OUTPUT_DIR,
                cache_dir=_DEFAULT_CACHE_DIR,
            )


class TestPrintAvailable:
    """Output of list_available."""

    def test_prints_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_print_available runs without raising and does not leak output."""
        _print_available()
        capsys.readouterr()
