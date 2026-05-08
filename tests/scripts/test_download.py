# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from trackers.datasets.download import _DEFAULT_CACHE_DIR, _DEFAULT_OUTPUT_DIR
from trackers.scripts.download import (
    _print_available,
    _run_download,
    add_download_subparser,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse argv through a fresh download subparser and return the namespace."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    add_download_subparser(subparsers)
    return parser.parse_args(argv)


class TestSubparserRegistration:
    """Argument parsing and help strings."""

    def test_list_flag(self) -> None:
        """--list sets the flag to True."""
        args = _parse_args(["download", "--list"])
        assert args.list is True

    def test_list_flag_default_false(self) -> None:
        """--list is False when omitted."""
        args = _parse_args(["download", "mot17"])
        assert args.list is False

    def test_split_flag_accepts_comma_separated(self) -> None:
        """--split accepts comma-separated values."""
        args = _parse_args(["download", "mot17", "--split", "train,val"])
        assert args.split == "train,val"

    def test_asset_flag_accepts_comma_separated(self) -> None:
        """--asset accepts comma-separated values."""
        args = _parse_args(["download", "mot17", "--asset", "frames,annotations"])
        assert args.asset == "frames,annotations"

    def test_output_directory_short_flag(self) -> None:
        """-o sets the output directory."""
        args = _parse_args(["download", "mot17", "-o", "./datasets"])
        assert args.output == "./datasets"

    def test_cache_dir_flag(self) -> None:
        """--cache-dir sets the cache directory."""
        args = _parse_args(["download", "mot17", "--cache-dir", "./cache"])
        assert args.cache_dir == "./cache"

    def test_dataset_positional(self) -> None:
        """Dataset is captured as positional argument."""
        args = _parse_args(["download", "sportsmot"])
        assert args.dataset == "sportsmot"


class TestRunDownload:
    """Execution of the download subcommand."""

    def test_list_triggers_print(self) -> None:
        """--list calls _print_available and returns 0."""
        args = _parse_args(["download", "--list"])

        with patch("trackers.scripts.download._print_available") as mock_print:
            rc = _run_download(args)
            assert rc == 0
            mock_print.assert_called_once()

    def test_list_takes_precedence_over_dataset(self) -> None:
        """--list wins over dataset positional."""
        args = _parse_args(["download", "mot17", "--list"])

        with patch("trackers.scripts.download._print_available") as mock_print:
            rc = _run_download(args)
            assert rc == 0
            mock_print.assert_called_once()

    def test_missing_dataset_exits_with_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No dataset and no --list prints error to stderr and returns 1."""
        args = _parse_args(["download"])
        rc = _run_download(args)
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
        """--split values are split on commas and whitespace-stripped."""
        args = _parse_args(["download", "mot17", "--split", split_arg, "--asset", "annotations"])

        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = _run_download(args)
            assert rc == 0
            mock_dl.assert_called_once_with(
                dataset="mot17",
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
        """--split handles malformed comma inputs gracefully."""
        args = _parse_args(["download", "mot17", "--split", split_arg, "--asset", "annotations"])

        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = _run_download(args)
            assert rc == 0
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
        args = _parse_args(["download", "sportsmot", "--split", "train", "--asset", asset_arg])

        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = _run_download(args)
            assert rc == 0
            mock_dl.assert_called_once_with(
                dataset="sportsmot",
                split=["train"],
                asset=expected_assets,
                output=_DEFAULT_OUTPUT_DIR,
                cache_dir=_DEFAULT_CACHE_DIR,
            )

    def test_none_splits_and_assets_when_omitted(self) -> None:
        """When --split and --asset are omitted, None is forwarded."""
        args = _parse_args(["download", "mot17"])

        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = _run_download(args)
            assert rc == 0
            mock_dl.assert_called_once_with(
                dataset="mot17",
                split=None,
                asset=None,
                output=_DEFAULT_OUTPUT_DIR,
                cache_dir=_DEFAULT_CACHE_DIR,
            )

    def test_output_directory_forwarded(self) -> None:
        """-o value is forwarded to download_dataset."""
        args = _parse_args(["download", "mot17", "-o", "/custom/path"])

        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = _run_download(args)
            assert rc == 0
            mock_dl.assert_called_once_with(
                dataset="mot17",
                split=None,
                asset=None,
                output="/custom/path",
                cache_dir=_DEFAULT_CACHE_DIR,
            )

    def test_value_error_returns_exit_code(self) -> None:
        """ValueError from download_dataset is caught and returns 1."""
        args = _parse_args(["download", "mot17"])

        with patch(
            "trackers.datasets.download.download_dataset",
            side_effect=ValueError("bad dataset"),
        ):
            rc = _run_download(args)
            assert rc == 1

    def test_split_with_spaces_stripped(self) -> None:
        """--split with spaces around commas strips whitespace."""
        args = _parse_args(["download", "mot17", "--split", "train , val", "--asset", "annotations"])

        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = _run_download(args)
            assert rc == 0
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
        """_print_available runs without raising and does not leak output."""
        _print_available()
        capsys.readouterr()
