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

SPORTSMOT_DATA_URL = (
    "https://storage.googleapis.com/com-roboflow-marketing/"
    "trackers/sportsmot-mot-test-data.zip"
)
DANCETRACK_DATA_URL = (
    "https://storage.googleapis.com/com-roboflow-marketing/"
    "trackers/dancetrack-mot-test-data.zip"
)
CACHE_DIR = Path.home() / ".cache" / "trackers-test"

# Dataset configurations: (url, cache_folder_name)
DATASETS = {
    "sportsmot": (SPORTSMOT_DATA_URL, "sportsmot-mot-test-data"),
    "dancetrack": (DANCETRACK_DATA_URL, "dancetrack-mot-test-data"),
}


def _get_test_data(dataset_name: str) -> tuple[Path, dict[str, Any]]:
    """Download and cache MOT test data for a given dataset.

    Args:
        dataset_name: Name of the dataset ("sportsmot" or "dancetrack").

    Returns:
        Tuple of (data_path, expected_results).

    Raises:
        pytest.skip: If download fails or data is unavailable.
    """
    if dataset_name not in DATASETS:
        pytest.skip(f"Unknown dataset: {dataset_name}")

    url, folder_name = DATASETS[dataset_name]
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
        pytest.skip(f"Failed to download {dataset_name} test data: {e}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_path)
    except Exception as e:
        if zip_path.exists():
            zip_path.unlink()
        pytest.skip(f"Failed to extract {dataset_name} test data: {e}")

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
                f"{dataset_name} extraction failed: expected_results.json not found"
            )

    with open(expected_path) as f:
        return cache_path, json.load(f)


@pytest.fixture(scope="session")
def sportsmot_test_data() -> tuple[Path, dict[str, Any]]:
    """Fixture providing SportsMOT test data path and expected results."""
    return _get_test_data("sportsmot")


@pytest.fixture(scope="session")
def dancetrack_test_data() -> tuple[Path, dict[str, Any]]:
    """Fixture providing DanceTrack test data path and expected results."""
    return _get_test_data("dancetrack")


@pytest.fixture
def test_data(dataset_name: str) -> tuple[Path, dict[str, Any]]:
    """Fixture providing test data for the current dataset."""
    return _get_test_data(dataset_name)


def _get_all_test_cases() -> list[tuple[str, str]]:
    """Get all (dataset_name, sequence_name) pairs for parametrization."""
    test_cases = []
    for dataset_name in DATASETS:
        try:
            _, expected = _get_test_data(dataset_name)
            # New format: sequences are top-level keys (no "sequences" wrapper)
            sequences = sorted(expected.keys())
            test_cases.extend((dataset_name, seq) for seq in sequences)
        except Exception:  # noqa: S112
            # Skip datasets that fail to download during collection
            continue
    return test_cases


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize tests that need dataset_name and sequence_name."""
    has_dataset = "dataset_name" in metafunc.fixturenames
    has_sequence = "sequence_name" in metafunc.fixturenames
    if has_dataset and has_sequence:
        test_cases = _get_all_test_cases()
        metafunc.parametrize(["dataset_name", "sequence_name"], test_cases)
