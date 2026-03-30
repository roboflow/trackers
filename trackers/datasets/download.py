# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from trackers.datasets.manifest import (
    _DATASETS,
    Dataset,
    DatasetAsset,
    DatasetSplit,
)
from trackers.utils.downloader import _download_file, _extract_zip
from trackers.utils.general import _normalize_list

_DEFAULT_OUTPUT_DIR = "."
_DEFAULT_CACHE_DIR = "~/.cache/trackers"


def _resolve_dataset(dataset: str | Dataset) -> Dataset:
    """Validate and convert a dataset identifier to a ``Dataset`` enum."""
    if isinstance(dataset, Dataset):
        dataset_key = dataset
    else:
        try:
            dataset_key = Dataset(dataset.lower())
        except ValueError:
            raise ValueError(f"Unknown dataset: {dataset}")

    if dataset_key not in _DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key.value}")

    return dataset_key


def _resolve_splits(
    split: list[str] | None,
    splits_dict: dict[str, dict[str, dict[str, Any]]],
    *,
    dataset_name: str,
) -> list[str]:
    """Return validated split names, defaulting to all available splits."""
    if split is None:
        return list(splits_dict.keys())

    for split_name in split:
        if split_name not in splits_dict:
            raise ValueError(
                f"Invalid split '{split_name}' for dataset '{dataset_name}'"
            )
    return split


def _resolve_assets(
    requested_assets: list[str] | None,
    available_assets: dict[str, dict[str, Any]],
    *,
    split_name: str,
    dataset_name: str,
) -> dict[str, dict[str, Any]]:
    """Return validated asset entries for a single split."""
    if not requested_assets:
        return available_assets

    selected: dict[str, dict[str, Any]] = {}
    for asset_type in requested_assets:
        if asset_type not in available_assets:
            raise ValueError(
                f"Asset '{asset_type}' not available for "
                f"split '{split_name}' in dataset"
                f" '{dataset_name}'"
            )
        selected[asset_type] = available_assets[asset_type]
    return selected


def download_dataset(
    *,
    dataset: str | Dataset,
    split: DatasetSplit | str | list[DatasetSplit | str] | None = None,
    asset: DatasetAsset | str | list[DatasetAsset | str] | None = None,
    output: str = _DEFAULT_OUTPUT_DIR,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> None:
    """Download benchmark tracking datasets from the official GCP bucket.

    Downloads ZIP files into a persistent cache directory and extracts
    them into the output directory. Cached ZIPs are reused across runs
    so that re-extraction after deleting the output directory does not
    require re-downloading.

    Args:
        dataset: Dataset to download, as a `Dataset` enum or string
            name. Case-insensitive.
        split: Splits to download. If `None`, all available splits
            are downloaded.
        asset: Asset types to download. If `None`, all available
            assets for each split are downloaded.
        output: Directory where dataset files will be extracted.
            Defaults to the current working directory.
        cache_dir: Directory for caching downloaded ZIP files.
            Cached ZIPs are verified by MD5 and reused when valid.

    Raises:
        ValueError: If `dataset`, `split`, or `asset` contains an
            unrecognized value.

    Examples:
        Using enums for type-safe dataset, split, and asset selection:

        >>> from trackers import Dataset, DatasetAsset, DatasetSplit, download_dataset
        >>> download_dataset(  # doctest: +SKIP
        ...     dataset=Dataset.MOT17,
        ...     split=[DatasetSplit.TRAIN, DatasetSplit.VAL],
        ...     asset=[DatasetAsset.ANNOTATIONS],
        ... )

        Using plain strings for quick, interactive use:

        >>> from trackers import download_dataset
        >>> download_dataset(  # doctest: +SKIP
        ...     dataset="mot17",
        ...     split=["train"],
        ...     asset=["frames", "annotations"],
        ...     output="./datasets",
        ... )
    """
    dataset_key = _resolve_dataset(dataset)

    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_cache_dir = Path(cache_dir).expanduser().resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    splits_dict = cast(
        dict[str, dict[str, dict[str, Any]]],
        _DATASETS[dataset_key]["splits"],
    )

    splits = _resolve_splits(
        _normalize_list(split), splits_dict, dataset_name=dataset_key.value
    )
    requested_assets = _normalize_list(asset)

    for split_name in splits:
        selected_assets = _resolve_assets(
            requested_assets,
            splits_dict[split_name],
            split_name=split_name,
            dataset_name=dataset_key.value,
        )

        for asset_type, item in selected_assets.items():
            url: str = item["url"]
            md5: str | None = item.get("md5")

            zip_name = Path(url).name
            cached_zip = resolved_cache_dir / zip_name

            label = f"{dataset_key.value}:{split_name}:{asset_type}"
            print(f"[download] {label}")
            was_downloaded = _download_file(url, cached_zip, md5=md5)
            if not was_downloaded:
                print(f"  using cached {zip_name}")

            print(f"[extract] {label}")
            _extract_zip(cached_zip, output_dir)

            print(f"[done] {label}")
