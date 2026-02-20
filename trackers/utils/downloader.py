#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def _verify_md5(file_path: Path, expected_md5: str) -> None:
    """Verify MD5 checksum of a downloaded file.

    Deletes the file and raises RuntimeError if the checksum does not match.
    """
    hash_md5 = hashlib.md5(usedforsecurity=False)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)

    actual_md5 = hash_md5.hexdigest()

    if actual_md5 != expected_md5:
        file_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 checksum mismatch for {file_path.name}: "
            f"expected {expected_md5}, got {actual_md5}"
        )


def download_file(
    url: str,
    dst: Path,
    *,
    md5: str | None = None,
    timeout: int = 30,
) -> None:
    """Download a file with a progress bar and optional MD5 verification."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with (
        open(tmp, "wb") as f,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dst.name,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    tmp.rename(dst)

    if md5 is not None:
        _verify_md5(dst, md5)


def extract_zip(
    zip_path: Path,
    output_dir: Path,
    *,
    cleanup: bool = True,
) -> None:
    """Extract a ZIP archive and optionally delete it afterwards."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    if cleanup:
        zip_path.unlink(missing_ok=True)
