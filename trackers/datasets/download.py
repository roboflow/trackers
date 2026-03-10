# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from trackers.datasets.manifest import _DATASETS
from trackers.utils.downloader import _download_file, _extract_zip

_DEFAULT_OUTPUT_DIR = "."
_DEFAULT_CACHE_DIR = "~/.cache/trackers"


def download_dataset(
    *,
    dataset: str,
    split: str | None = None,
    content: str | None = None,
    output: str = _DEFAULT_OUTPUT_DIR,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> None:
    """Download benchmark tracking datasets from the official GCP bucket.

    Downloads ZIP files into a persistent cache directory and extracts
    them into the output directory. Cached ZIPs are reused across runs
    so that re-extraction after deleting the output directory does not
    require re-downloading.

    Args:
        dataset: Name of the dataset to download (e.g. `"mot17"`,
            `"sportsmot"`). Case-insensitive.
        split: Comma-separated list of splits to download (e.g.
            `"train"`, `"train,val"`). If `None`, all available splits
            are downloaded.
        content: Comma-separated list of content types to download (e.g.
            `"annotations"`, `"frames,detections"`). If `None`, all
            available content types for each split are downloaded.
        output: Directory where dataset files will be extracted. Defaults
            to the current working directory.
        cache_dir: Directory for caching downloaded ZIP files. Defaults
            to `~/.cache/trackers`. Cached ZIPs are verified by MD5
            checksum and reused when valid.

    Raises:
        ValueError: If `dataset` is not a recognized dataset name, if
            `split` contains a split not available for the dataset, or
            if `content` contains a content type not available for the
            requested split.

    Examples:
        >>> from trackers import download_dataset
        >>> download_dataset(  # doctest: +SKIP
        ...     dataset="mot17", split="train", content="annotations",
        ... )
    """
    dataset = dataset.lower()
    if dataset not in _DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_cache_dir = Path(cache_dir).expanduser().resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    splits_dict = cast(
        dict[str, dict[str, dict[str, Any]]],
        _DATASETS[dataset]["splits"],
    )

    if split:
        splits: list[str] = [s.strip() for s in split.split(",")]
    else:
        splits = list(splits_dict.keys())

    if content:
        requested_content: list[str] = [c.strip() for c in content.split(",")]
    else:
        requested_content = []

    for split_name in splits:
        if split_name not in splits_dict:
            raise ValueError(f"Invalid split '{split_name}' for dataset '{dataset}'")

        available_content: dict[str, dict[str, Any]] = splits_dict[split_name]

        if requested_content:
            selected_content: dict[str, dict[str, Any]] = {}
            for content_type in requested_content:
                if content_type not in available_content:
                    raise ValueError(
                        f"Content '{content_type}' not available for "
                        f"split '{split_name}' in dataset '{dataset}'"
                    )
                selected_content[content_type] = available_content[content_type]
        else:
            selected_content = available_content

        for kind, item in selected_content.items():
            url: str = item["url"]
            md5: str | None = item.get("md5")

            zip_name = Path(url).name
            cached_zip = resolved_cache_dir / zip_name

            print(f"[download] {dataset}:{split_name}:{kind}")
            was_downloaded = _download_file(url, cached_zip, md5=md5)
            if not was_downloaded:
                print(f"  using cached {zip_name}")

            print(f"[extract] {dataset}:{split_name}:{kind}")
            _extract_zip(cached_zip, output_dir)

            print(f"[done] {dataset}:{split_name}:{kind}")
