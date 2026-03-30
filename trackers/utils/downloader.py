# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TransferSpeedColumn,
)

_CHUNK_SIZE_BYTES = 8192
_DEFAULT_TIMEOUT_SECONDS = 30


def _compute_md5(file_path: Path) -> str:
    """Compute MD5 hex digest of a file."""
    hash_md5 = hashlib.md5(usedforsecurity=False)
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(_CHUNK_SIZE_BYTES), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _download_file(
    url: str,
    destination: Path,
    *,
    md5: str | None = None,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Download a file with a progress bar and optional MD5 verification.

    If `destination` already exists and `md5` is provided, the existing
    file's checksum is verified. The download is skipped when the
    checksum matches.

    Returns:
        `True` if the file was downloaded, `False` if a cached version
        was used.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and md5 is not None:
        if _compute_md5(destination) == md5:
            return False
        destination.unlink()

    temp_path = destination.with_suffix(".tmp")

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
    )
    with open(temp_path, "wb") as file_handle, progress:
        task = progress.add_task(destination.name, total=total or None)
        for chunk in response.iter_content(chunk_size=_CHUNK_SIZE_BYTES):
            if chunk:
                file_handle.write(chunk)
                progress.update(task, advance=len(chunk))

    temp_path.rename(destination)

    if md5 is not None:
        actual = _compute_md5(destination)
        if actual != md5:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"MD5 checksum mismatch for {destination.name}: "
                f"expected {md5}, got {actual}"
            )

    return True


def _extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract a ZIP archive into `output_dir`."""
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(output_dir)
