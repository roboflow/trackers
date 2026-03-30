# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import pytest

from trackers.datasets.download import (
    _resolve_assets,
    _resolve_dataset,
    _resolve_splits,
)
from trackers.datasets.manifest import Dataset

_SAMPLE_SPLITS: dict[str, dict[str, dict[str, str]]] = {
    "train": {
        "frames": {"url": "https://example.com/train-frames.zip", "md5": "abc"},
        "annotations": {"url": "https://example.com/train-ann.zip", "md5": "def"},
    },
    "val": {
        "frames": {"url": "https://example.com/val-frames.zip", "md5": "ghi"},
    },
}

_SAMPLE_ASSETS = _SAMPLE_SPLITS["train"]


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        # Enum member - passes through unchanged
        (Dataset.MOT17, Dataset.MOT17),
        # Second enum member - passes through unchanged
        (Dataset.SPORTSMOT, Dataset.SPORTSMOT),
        # Lowercase string - converted to matching enum
        ("mot17", Dataset.MOT17),
        # Uppercase string - normalized and converted
        ("MOT17", Dataset.MOT17),
        # String for second dataset - converted to matching enum
        ("sportsmot", Dataset.SPORTSMOT),
    ],
)
def test_resolve_dataset_with_valid_input(
    dataset: str | Dataset, expected: Dataset
) -> None:
    assert _resolve_dataset(dataset) is expected


@pytest.mark.parametrize(
    "dataset",
    [
        # Unknown name - not a registered dataset
        "nonexistent",
        # Empty string - no dataset matches
        "",
    ],
)
def test_resolve_dataset_rejects_unknown(dataset: str) -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        _resolve_dataset(dataset)


@pytest.mark.parametrize(
    ("split", "expected"),
    [
        # None - defaults to all available splits
        (None, ["train", "val"]),
        # Single split - passes through unchanged
        (["val"], ["val"]),
        # Multiple splits - passes through unchanged
        (["train", "val"], ["train", "val"]),
        # Empty list - returns empty
        ([], []),
    ],
)
def test_resolve_splits_with_valid_input(
    split: list[str] | None, expected: list[str]
) -> None:
    result = _resolve_splits(split, _SAMPLE_SPLITS, dataset_name="mot17")
    assert result == expected


@pytest.mark.parametrize(
    ("split", "match"),
    [
        # Unknown split - not in available splits
        (["nonexistent"], "Invalid split 'nonexistent'"),
        # Mixed valid and invalid - fails on invalid entry
        (["train", "bad"], "Invalid split 'bad'"),
    ],
)
def test_resolve_splits_rejects_unknown(split: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _resolve_splits(split, _SAMPLE_SPLITS, dataset_name="mot17")


@pytest.mark.parametrize(
    ("requested", "expected_keys"),
    [
        # None - defaults to all available assets
        (None, ["frames", "annotations"]),
        # Empty list - defaults to all available assets
        ([], ["frames", "annotations"]),
        # Single asset - filters to requested
        (["annotations"], ["annotations"]),
        # Multiple assets - preserves request order
        (["frames", "annotations"], ["frames", "annotations"]),
    ],
)
def test_resolve_assets_with_valid_input(
    requested: list[str] | None,
    expected_keys: list[str],
) -> None:
    result = _resolve_assets(
        requested,
        _SAMPLE_ASSETS,
        split_name="train",
        dataset_name="mot17",
    )
    assert list(result.keys()) == expected_keys


@pytest.mark.parametrize(
    ("requested", "available", "split_name", "dataset_name", "match"),
    [
        # Missing asset type - not present in split
        (
            ["detections"],
            _SAMPLE_ASSETS,
            "train",
            "mot17",
            "Asset 'detections' not available",
        ),
        # Error context - message includes split and dataset name
        (
            ["annotations"],
            {"frames": {"url": "x"}},
            "val",
            "sportsmot",
            r"split 'val'.*dataset 'sportsmot'",
        ),
    ],
)
def test_resolve_assets_rejects_unknown(
    requested: list[str],
    available: dict[str, dict[str, str]],
    split_name: str,
    dataset_name: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _resolve_assets(
            requested,
            available,
            split_name=split_name,
            dataset_name=dataset_name,
        )
