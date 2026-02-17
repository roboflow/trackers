# trackers/scripts/download.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

from trackers.io.datasets import DATASETS
from trackers.utils.downloader import download_with_progress, extract_zip


def add_download_subparser(subparsers):
    parser = subparsers.add_parser(
        "download",
        help="Download benchmark tracking datasets.",
        description="Download tracking datasets from the trackers bucket",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets, splits, and content types.",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name (e.g. mot17, sportsmot). Warning : only MOT is suppported",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        help="Dataset split to download.",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Download only ground-truth annotations.",
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
    splits = [args.split] if args.split else list(ds["splits"].keys())

    for split in splits:
        if split not in ds["splits"]:
            sys.exit(f"Invalid split '{split}' for dataset '{dataset}'")

        contents: Dict[str, str] = ds["splits"][split]

        if args.annotations_only:
            if "annotations" not in contents:
                print(f"[skip] {dataset}:{split} has no annotations")
                continue
            contents = {"annotations": contents["annotations"]}

        for kind, url in contents.items():
            zip_name = url.split("/")[-1]
            zip_path = output_dir / zip_name

            if zip_path.exists():
                print(f"[skip] {zip_name} already exists")
                continue

            print(f"[download] {dataset}:{split}:{kind}")
            download_with_progress(url, zip_path)
            extract_zip(zip_path, output_dir)

    return 0


def _print_available():
    print("\nAvailable datasets:\n")
    for name, ds in DATASETS.items():
        print(f"{name}: {ds.get('description', '')}")
        for split, contents in ds["splits"].items():
            kinds = ", ".join(contents.keys())
            print(f"  - {split}: {kinds}")
        print()
