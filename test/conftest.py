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

SOCCERNET_DATA_URL = (
    "https://storage.googleapis.com/com-roboflow-marketing/"
    "trackers/soccernet-mot-test-data.zip"
)
CACHE_DIR = Path.home() / ".cache" / "trackers-test"


def _get_soccernet_data() -> tuple[Path, dict[str, Any]]:
    """Download and cache SoccerNet MOT test data.

    Returns:
        Tuple of (data_path, expected_results).

    Raises:
        pytest.skip: If download fails or data is unavailable.
    """
    cache_path = CACHE_DIR / "soccernet-mot-test-data"
    zip_path = CACHE_DIR / "soccernet-mot-test-data.zip"
    expected_path = cache_path / "expected_results.json"

    if cache_path.exists() and expected_path.exists():
        with open(expected_path) as f:
            return cache_path, json.load(f)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(SOCCERNET_DATA_URL, zip_path)  # noqa: S310
    except Exception as e:
        pytest.skip(f"Failed to download test data: {e}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(CACHE_DIR)
    except Exception as e:
        if zip_path.exists():
            zip_path.unlink()
        pytest.skip(f"Failed to extract test data: {e}")

    if zip_path.exists():
        zip_path.unlink()

    if not expected_path.exists():
        for p in cache_path.rglob("expected_results.json"):
            expected_path = p
            cache_path = p.parent
            break
        else:
            shutil.rmtree(cache_path, ignore_errors=True)
            pytest.skip("Test data extraction failed: expected_results.json not found")

    with open(expected_path) as f:
        return cache_path, json.load(f)


@pytest.fixture(scope="session")
def soccernet_test_data() -> tuple[Path, dict[str, Any]]:
    """Fixture providing SoccerNet MOT test data path and expected results."""
    return _get_soccernet_data()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize tests that need sequence_name."""
    if "sequence_name" in metafunc.fixturenames:
        _, expected = _get_soccernet_data()
        sequences = sorted(expected["sequences"].keys())
        metafunc.parametrize("sequence_name", sequences)
