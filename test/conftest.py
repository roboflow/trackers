# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Any

# Test data URLs and folder names
DATASETS: dict[str, tuple[str, str]] = {
    "sportsmot_flat": (
        "https://storage.googleapis.com/com-roboflow-marketing/"
        "trackers/sportsmot-flat-20260203.zip",
        "sportsmot-flat",
    ),
    "sportsmot_mot17": (
        "https://storage.googleapis.com/com-roboflow-marketing/"
        "trackers/sportsmot-mot17-20260203.zip",
        "sportsmot-mot17",
    ),
    "dancetrack_flat": (
        "https://storage.googleapis.com/com-roboflow-marketing/"
        "trackers/dancetrack-flat-20260203.zip",
        "dancetrack-flat",
    ),
    "dancetrack_mot17": (
        "https://storage.googleapis.com/com-roboflow-marketing/"
        "trackers/dancetrack-mot17-20260203.zip",
        "dancetrack-mot17",
    ),
}

CACHE_DIR = Path.home() / ".cache" / "trackers-test"


def _download_test_data(dataset_key: str) -> tuple[Path, dict[str, Any]]:
    """Download and cache MOT test data for a given dataset.

    Args:
        dataset_key: Key from DATASETS dict (e.g., "sportsmot_flat").

    Returns:
        Tuple of (data_path, expected_results).

    Raises:
        pytest.skip: If download fails or data is unavailable.
    """
    if dataset_key not in DATASETS:
        pytest.skip(f"Unknown dataset: {dataset_key}")

    url, folder_name = DATASETS[dataset_key]

    cache_path = CACHE_DIR / folder_name
    zip_path = CACHE_DIR / f"{folder_name}.zip"
    expected_path = cache_path / "expected_results.json"

    if cache_path.exists() and expected_path.exists():
        with open(expected_path) as f:
            return cache_path, json.load(f)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, zip_path)  # noqa: S310
    except Exception as e:
        pytest.skip(f"Failed to download {dataset_key} test data: {e}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_path)
    except Exception as e:
        if zip_path.exists():
            zip_path.unlink()
        pytest.skip(f"Failed to extract {dataset_key} test data: {e}")

    if zip_path.exists():
        zip_path.unlink()

    if not expected_path.exists():
        for p in cache_path.rglob("expected_results.json"):
            expected_path = p
            cache_path = p.parent
            break
        else:
            shutil.rmtree(cache_path, ignore_errors=True)
            pytest.skip(
                f"{dataset_key} extraction failed: expected_results.json not found"
            )

    with open(expected_path) as f:
        return cache_path, json.load(f)


@pytest.fixture(scope="session")
def sportsmot_flat_data() -> tuple[Path, dict[str, Any]]:
    """Fixture providing SportsMOT flat format test data."""
    return _download_test_data("sportsmot_flat")


@pytest.fixture(scope="session")
def sportsmot_mot17_data() -> tuple[Path, dict[str, Any]]:
    """Fixture providing SportsMOT MOT17 format test data."""
    return _download_test_data("sportsmot_mot17")


@pytest.fixture(scope="session")
def dancetrack_flat_data() -> tuple[Path, dict[str, Any]]:
    """Fixture providing DanceTrack flat format test data."""
    return _download_test_data("dancetrack_flat")


@pytest.fixture(scope="session")
def dancetrack_mot17_data() -> tuple[Path, dict[str, Any]]:
    """Fixture providing DanceTrack MOT17 format test data."""
    return _download_test_data("dancetrack_mot17")
