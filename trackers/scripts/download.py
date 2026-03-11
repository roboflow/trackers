#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

from trackers.datasets.download import _DEFAULT_CACHE_DIR, _DEFAULT_OUTPUT_DIR
from trackers.datasets.manifest import _DATASETS


def add_download_subparser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add the download subcommand to the argument parser."""
    parser = subparsers.add_parser(
        "download",
        help="Download benchmark tracking datasets.",
        description="Download tracking datasets from the official trackers bucket.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets, splits, and asset types.",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name (e.g. mot17, sportsmot).",
    )
    parser.add_argument(
        "--split",
        help="Comma-separated splits to download (e.g. train,val,test). "
        "If omitted, all available splits are downloaded.",
    )
    parser.add_argument(
        "--asset",
        help="Comma-separated assets to download: annotations,frames,detections. "
        "If omitted, all available assets are downloaded.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=_DEFAULT_OUTPUT_DIR,
        help="Output directory (default: current directory).",
    )
    parser.add_argument(
        "--cache-dir",
        default=_DEFAULT_CACHE_DIR,
        help="Cache directory for downloaded ZIPs (default: ~/.cache/trackers).",
    )

    parser.set_defaults(func=_run_download)


def _run_download(args: argparse.Namespace) -> int:
    """Execute the download subcommand."""
    if args.list:
        _print_available()
        return 0

    if not args.dataset:
        print("Please specify a dataset name or use --list.", file=sys.stderr)
        return 1

    from trackers.datasets.download import download_dataset

    split_list = [s.strip() for s in args.split.split(",")] if args.split else None
    asset_list = [a.strip() for a in args.asset.split(",")] if args.asset else None

    try:
        download_dataset(
            dataset=args.dataset,
            split=split_list,
            asset=asset_list,
            output=args.output,
            cache_dir=args.cache_dir,
        )
    except ValueError as e:
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
            f"{split:<{max_split_len}}   {', '.join(assets.keys())}"
            for split, assets in splits_dict.items()
        ]

        body = f"{description}\n\n" + "\n".join(split_lines)
        console.print(Panel(body, title=name.value, title_align="left"))
        console.print()
