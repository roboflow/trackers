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

# URL to the SoccerNet MOT test data hosted on GCS
SOCCERNET_DATA_URL = (
    "https://storage.googleapis.com/com-roboflow-marketing/"
    "trackers/soccernet-mot-test-data.zip"
)

# Cache directory for test data
CACHE_DIR = Path.home() / ".cache" / "trackers-test"


@pytest.fixture(scope="session")
def soccernet_test_data() -> tuple[Path, dict[str, Any]]:
    """Download and cache SoccerNet MOT test data.

    Downloads the test data zip file from GCS if not already cached,
    extracts it, and returns the path to the data along with expected
    results from TrackEval.

    Returns:
        Tuple of (data_path, expected_results) where:
        - data_path: Path to the extracted test data directory
        - expected_results: Dict containing expected metrics from TrackEval

    Raises:
        pytest.skip: If download fails or data is unavailable.
    """
    cache_path = CACHE_DIR / "soccernet-mot-test-data"
    zip_path = CACHE_DIR / "soccernet-mot-test-data.zip"
    expected_path = cache_path / "expected_results.json"

    # Return cached data if available
    if cache_path.exists() and expected_path.exists():
        with open(expected_path) as f:
            expected = json.load(f)
        return cache_path, expected

    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download zip file
    try:
        urllib.request.urlretrieve(SOCCERNET_DATA_URL, zip_path)  # noqa: S310
    except Exception as e:
        pytest.skip(f"Failed to download test data: {e}")

    # Extract zip file
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(CACHE_DIR)
    except Exception as e:
        # Clean up partial download
        if zip_path.exists():
            zip_path.unlink()
        pytest.skip(f"Failed to extract test data: {e}")

    # Clean up zip file after extraction
    if zip_path.exists():
        zip_path.unlink()

    # Verify extraction
    if not expected_path.exists():
        # Try to find expected_results.json in nested directory
        for p in cache_path.rglob("expected_results.json"):
            expected_path = p
            cache_path = p.parent
            break
        else:
            shutil.rmtree(cache_path, ignore_errors=True)
            pytest.skip("Test data extraction failed: expected_results.json not found")

    # Load expected results
    with open(expected_path) as f:
        expected = json.load(f)

    return cache_path, expected


# List of all 49 SoccerNet sequences for parametrization
SOCCERNET_SEQUENCES = [
    "SNMOT-116",
    "SNMOT-117",
    "SNMOT-118",
    "SNMOT-119",
    "SNMOT-120",
    "SNMOT-121",
    "SNMOT-122",
    "SNMOT-123",
    "SNMOT-124",
    "SNMOT-125",
    "SNMOT-126",
    "SNMOT-127",
    "SNMOT-128",
    "SNMOT-129",
    "SNMOT-130",
    "SNMOT-131",
    "SNMOT-132",
    "SNMOT-133",
    "SNMOT-134",
    "SNMOT-135",
    "SNMOT-136",
    "SNMOT-137",
    "SNMOT-138",
    "SNMOT-139",
    "SNMOT-140",
    "SNMOT-141",
    "SNMOT-142",
    "SNMOT-143",
    "SNMOT-144",
    "SNMOT-145",
    "SNMOT-146",
    "SNMOT-147",
    "SNMOT-148",
    "SNMOT-149",
    "SNMOT-150",
    "SNMOT-187",
    "SNMOT-188",
    "SNMOT-189",
    "SNMOT-190",
    "SNMOT-191",
    "SNMOT-192",
    "SNMOT-193",
    "SNMOT-194",
    "SNMOT-195",
    "SNMOT-196",
    "SNMOT-197",
    "SNMOT-198",
    "SNMOT-199",
    "SNMOT-200",
]
