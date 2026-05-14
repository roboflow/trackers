#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

from trackers.datasets.download import _DEFAULT_CACHE_DIR, _DEFAULT_OUTPUT_DIR
from trackers.datasets.manifest import _DATASETS


def download(
    dataset: str = "",
    list_available: bool = False,
    split: str = "",
    asset: str = "",
    output: str = _DEFAULT_OUTPUT_DIR,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> int:
    """Download benchmark tracking datasets.

    Args:
        dataset: Dataset name (e.g. mot17, sportsmot).
        list_available: List available datasets, splits, and asset types.
        split: Comma-separated splits to download (e.g. train,val,test).
        asset: Comma-separated assets to download (e.g. annotations,frames,detections).
        output: Output directory.
        cache_dir: Cache directory for downloaded ZIPs.

    Returns:
        Exit code: 0 on success, 1 on error.

    Examples:
        >>> download() == 1
        True
    """
    if list_available:
        _print_available()
        return 0

    if not dataset:
        print("Please specify a dataset name or use --list.", file=sys.stderr)
        return 1

    from trackers.datasets.download import download_dataset

    split_list = [s.strip() for s in split.split(",")] if split else None
    asset_list = [a.strip() for a in asset.split(",")] if asset else None

    try:
        download_dataset(
            dataset=dataset,
            split=split_list,
            asset=asset_list,
            output=output,
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def _print_available() -> None:
    """Print available datasets, splits, and asset types."""
    console = Console()
    for name, dataset_info in _DATASETS.items():
        description = dataset_info.get("description", "")
        splits_dict: dict[str, dict] = dataset_info.get("splits", {})

        max_split_len = max(len(s) for s in splits_dict) if splits_dict else 0
        split_lines = [
            f"{split:<{max_split_len}}   {', '.join(assets.keys())}" for split, assets in splits_dict.items()
        ]

        body = f"{description}\n\n" + "\n".join(split_lines)
        console.print(Panel(body, title=name.value, title_align="left"))
        console.print()
