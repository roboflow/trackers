# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel

from trackers.datasets.download import _DEFAULT_CACHE_DIR, _DEFAULT_OUTPUT_DIR
from trackers.datasets.manifest import _DATASETS


@click.command("download")
@click.argument("dataset", required=False, default=None)
@click.option("--list", "show_list", is_flag=True, help="List available datasets, splits, and asset types.")
@click.option(
    "--split",
    default=None,
    help="Comma-separated splits to download (e.g. train,val,test). If omitted, all available splits are downloaded.",
)
@click.option(
    "--asset",
    default=None,
    help=(
        "Comma-separated assets to download: annotations,frames,detections."
        " If omitted, all available assets are downloaded."
    ),
)
@click.option("-o", "--output", default=_DEFAULT_OUTPUT_DIR, help="Output directory (default: current directory).")
@click.option(
    "--cache-dir",
    "cache_dir",
    default=_DEFAULT_CACHE_DIR,
    help="Cache directory for downloaded ZIPs (default: ~/.cache/trackers).",
)
def download_command(
    dataset: str | None,
    show_list: bool,
    split: str | None,
    asset: str | None,
    output: str,
    cache_dir: str,
) -> None:
    """Download benchmark tracking datasets."""
    if show_list:
        _print_available()
        return

    if not dataset:
        raise click.UsageError("Please specify a dataset name or use --list.")

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
        raise click.ClickException(str(e)) from e


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
