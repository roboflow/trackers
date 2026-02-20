#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

from trackers.datasets.manifest import DATASETS
from trackers.utils.downloader import download_file, extract_zip


def add_download_subparser(subparsers):
    parser = subparsers.add_parser(
        "download",
        help="Download benchmark tracking datasets.",
        description="Download tracking datasets from the official trackers bucket.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets, splits, and content types.",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name (e.g. mot17). Warning: only MOT17 is supported currently.",
    )
    parser.add_argument(
        "--split",
        help="List of splits to download (e.g. train,val,test). "
        "If omitted, all available splits are downloaded.",
    )
    parser.add_argument(
        "--content",
        help="List of content to download: annotations,frames,detections. "
        "If omitted, all available content is downloaded.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./datasets",
        help="Output directory (default: ./datasets).",
    )

    parser.set_defaults(func=run_download)


def run_download(args) -> int:
    if args.list:
        _print_available()
        return 0

    if not args.dataset:
        sys.exit("Please specify a dataset name or use --list.")

    dataset = args.dataset.lower()
    if dataset not in DATASETS:
        sys.exit(f"Unknown dataset: {dataset}")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = DATASETS[dataset]

    # Parse splits
    if args.split:
        splits: List[str] = [s.strip() for s in args.split.split(",")]
    else:
        splits = list(ds["splits"].keys())

    # Parse content
    if args.content:
        requested_content: List[str] = [c.strip() for c in args.content.split(",")]
    else:
        requested_content = []

    for split in splits:
        if split not in ds["splits"]:
            sys.exit(f"Invalid split '{split}' for dataset '{dataset}'")

        available_content: Dict[str, dict] = ds["splits"][split]

        # Resolve which content to download
        if requested_content:
            selected_content: Dict[str, dict] = {}
            for c in requested_content:
                if c not in available_content:
                    sys.exit(
                        f"Error: content '{c}' is not available for split '{split}' "
                        f"in dataset '{dataset}'"
                    )
                selected_content[c] = available_content[c]
        else:
            selected_content = available_content

        for kind, item in selected_content.items():
            url = item["url"]
            md5 = item.get("md5")

            # marker file = source of truth
            marker = output_dir / f".{dataset}-{split}-{kind}.complete"
            if marker.exists():
                print(f"[skip] {dataset}:{split}:{kind} already downloaded")
                continue

            zip_name = url.split("/")[-1]
            zip_path = output_dir / zip_name

            print(f"[download] {dataset}:{split}:{kind}")
            download_file(url, zip_path, md5=md5)
            extract_zip(zip_path, output_dir)

            # mark completion only after successful extraction
            marker.touch()

    return 0


def _print_available():
    print("\nAvailable datasets:\n")
    for name, ds in DATASETS.items():
        print(f"{name}: {ds.get('description', '')}")
        for split, contents in ds["splits"].items():
            kinds = ", ".join(contents.keys())
            print(f"  - {split}: {kinds}")
        print()
