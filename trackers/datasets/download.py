#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


from __future__ import annotations

from pathlib import Path
from typing import Any

from trackers.datasets.manifest import DATASETS
from trackers.utils.downloader import download_file, extract_zip


def download(
    *,
    dataset: str,
    split: str | None = None,
    content: str | None = None,
    output: str = "./data",
) -> None:
    """
    Download benchmark tracking datasets.

    Example:
        >>> from trackers.datasets.download import download
        >>> download(dataset="mot17", split="train", content="frames")
    """

    dataset = dataset.lower()
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")

    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from typing import cast

    splits_dict = cast(
        dict[str, dict[str, dict[str, Any]]],
        DATASETS[dataset]["splits"],
    )

    # Resolve splits (ALWAYS list[str])
    if split:
        splits: list[str] = [s.strip() for s in split.split(",")]
    else:
        splits = list(splits_dict.keys())

    # Resolve content (ALWAYS list[str])
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
            for c in requested_content:
                if c not in available_content:
                    raise ValueError(
                        f"Content '{c}' not available for split '{split_name}' "
                        f"in dataset '{dataset}'"
                    )
                selected_content[c] = available_content[c]
        else:
            selected_content = available_content

        for kind, item in selected_content.items():
            url: str = item["url"]
            md5: str | None = item.get("md5")

            marker = output_dir / f".{dataset}-{split_name}-{kind}.complete"
            if marker.exists():
                print(f"[skip] {dataset}:{split_name}:{kind} already downloaded")
                continue

            zip_name = url.split("/")[-1]
            zip_path = output_dir / zip_name

            print(f"[download] {dataset}:{split_name}:{kind}")
            download_file(url, zip_path, md5=md5)
            extract_zip(zip_path, output_dir)

            marker.touch()
            print(f"[complete] {dataset}:{split_name}:{kind}")
