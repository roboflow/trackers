# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from unittest.mock import patch

import pytest

from trackers.cli.download import _print_available, download
from trackers.datasets.download import _DEFAULT_CACHE_DIR, _DEFAULT_OUTPUT_DIR


class TestDownloadList:
    def test_list_triggers_print(self) -> None:
        """list_available=True calls _print_available and returns 0."""
        with patch("trackers.cli.download._print_available") as mock_print:
            rc = download(list_available=True)
        assert rc == 0
        mock_print.assert_called_once()

    def test_list_takes_precedence_over_dataset(self) -> None:
        """list_available=True wins over a provided dataset name."""
        with patch("trackers.cli.download._print_available") as mock_print:
            rc = download(dataset="mot17", list_available=True)
        assert rc == 0
        mock_print.assert_called_once()

    def test_prints_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_print_available runs without raising."""
        _print_available()
        capsys.readouterr()


class TestDownloadMissingDataset:
    def test_missing_dataset_returns_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No dataset and no list_ prints to stderr and returns 1."""
        rc = download()
        captured = capsys.readouterr()
        assert rc == 1
        assert "Please specify a dataset" in captured.err


class TestDownloadExecution:
    @pytest.mark.parametrize(
        "split_arg,expected_splits",
        [
            ("train", ["train"]),
            ("train,val", ["train", "val"]),
            ("train,val,test", ["train", "val", "test"]),
        ],
    )
    def test_split_comma_parsing(self, split_arg: str, expected_splits: list[str]) -> None:
        """split values are split on commas and whitespace-stripped."""
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = download(dataset="mot17", split=split_arg, asset="annotations")
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
        ],
    )
    def test_asset_comma_parsing(self, asset_arg: str, expected_assets: list[str]) -> None:
        """asset values are split on commas."""
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = download(dataset="sportsmot", split="train", asset=asset_arg)
        assert rc == 0
        mock_dl.assert_called_once_with(
            dataset="sportsmot",
            split=["train"],
            asset=expected_assets,
            output=_DEFAULT_OUTPUT_DIR,
            cache_dir=_DEFAULT_CACHE_DIR,
        )

    def test_empty_split_and_asset_pass_none(self) -> None:
        """Empty split/asset strings forward None to download_dataset."""
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = download(dataset="mot17")
        assert rc == 0
        mock_dl.assert_called_once_with(
            dataset="mot17",
            split=None,
            asset=None,
            output=_DEFAULT_OUTPUT_DIR,
            cache_dir=_DEFAULT_CACHE_DIR,
        )

    def test_custom_output_forwarded(self) -> None:
        """output value is forwarded to download_dataset."""
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = download(dataset="mot17", output="/custom/path")
        assert rc == 0
        mock_dl.assert_called_once_with(
            dataset="mot17",
            split=None,
            asset=None,
            output="/custom/path",
            cache_dir=_DEFAULT_CACHE_DIR,
        )

    def test_exception_returns_error(self) -> None:
        """Exception from download_dataset is caught and returns 1."""
        with patch(
            "trackers.datasets.download.download_dataset",
            side_effect=ValueError("bad dataset"),
        ):
            rc = download(dataset="mot17")
        assert rc == 1

    def test_split_whitespace_stripped(self) -> None:
        """Whitespace around commas in split is stripped."""
        with patch("trackers.datasets.download.download_dataset") as mock_dl:
            rc = download(dataset="mot17", split="train , val", asset="annotations")
        assert rc == 0
        mock_dl.assert_called_once_with(
            dataset="mot17",
            split=["train", "val"],
            asset=["annotations"],
            output=_DEFAULT_OUTPUT_DIR,
            cache_dir=_DEFAULT_CACHE_DIR,
        )
