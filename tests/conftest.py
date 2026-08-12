# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
import zipfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from typing import Any

# Test data URLs and folder names
DATASETS: dict[str, tuple[str, str]] = {
    "sportsmot_flat": (
        "https://storage.googleapis.com/com-roboflow-marketing/trackers/sportsmot-flat-20260203.zip",
        "sportsmot-flat",
    ),
    "sportsmot_mot17": (
        "https://storage.googleapis.com/com-roboflow-marketing/trackers/sportsmot-mot17-20260203.zip",
        "sportsmot-mot17",
    ),
    "dancetrack_flat": (
        "https://storage.googleapis.com/com-roboflow-marketing/trackers/dancetrack-flat-20260203.zip",
        "dancetrack-flat",
    ),
    "dancetrack_mot17": (
        "https://storage.googleapis.com/com-roboflow-marketing/trackers/dancetrack-mot17-20260203.zip",
        "dancetrack-mot17",
    ),
}

CACHE_DIR = Path.home() / ".cache" / "trackers-test"

# Pinned NVIDIA PhysicalAI-SmartSpaces dataset revision for multicamera fixtures.
MULTICAMERA_HF_REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
MULTICAMERA_HF_REVISION = "1eebcf0f74a510994fe4c886f4fa77fbc6724ea8"


def _require_test_data() -> bool:
    """Return True when missing external fixtures must fail instead of skip."""
    return os.environ.get("TRACKERS_REQUIRE_TEST_DATA", "").strip() in {"1", "true", "True"}


def _unavailable(message: str) -> None:
    if _require_test_data():
        raise RuntimeError(message)
    pytest.skip(message)


def hf_fixture_file(
    filename: str,
    *,
    repo_id: str = MULTICAMERA_HF_REPO_ID,
    revision: str = MULTICAMERA_HF_REVISION,
) -> Path:
    """Download a single Hugging Face dataset file for tests.

    Mirrors the ``reid`` package's ``hf_hub_download`` error wrapping, with
    ``repo_type=\"dataset\"`` because multicamera fixtures live in a dataset
    repo rather than a model repo.

    Args:
        filename: Path within the dataset repository.
        repo_id: Hugging Face dataset repository id.
        revision: Immutable git revision SHA.

    Returns:
        Local path to the cached file.

    Raises:
        RuntimeError: When ``TRACKERS_REQUIRE_TEST_DATA`` is set and the
            download fails; otherwise the failure becomes ``pytest.skip``.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
    except ImportError as exc:
        _unavailable(
            "huggingface_hub is required for multicamera integration fixtures. Install the dev dependency group."
        )
        raise AssertionError("unreachable") from exc

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    except (HfHubHTTPError, EntryNotFoundError, OSError) as exc:
        _unavailable(
            f"Failed to download Hugging Face fixture {filename!r} (repo_id={repo_id!r}, revision={revision!r}): {exc}"
        )
        raise AssertionError("unreachable") from exc
    return Path(path)


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Reset random state before each test for reproducibility."""
    import random

    random.seed(42)
    np.random.seed(42)
    with suppress(ImportError):
        import torch

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)


def _download_test_data(dataset_key: str) -> tuple[Path, dict[str, Any]]:
    """Download and cache MOT test data for a given dataset.

    Args:
        dataset_key: Key from DATASETS dict (e.g., "sportsmot_flat").

    Returns:
        Tuple of (data_path, expected_results).

    Raises:
        pytest.skip: If download fails or data is unavailable.
        RuntimeError: If download fails and ``TRACKERS_REQUIRE_TEST_DATA`` is set.
    """
    if dataset_key not in DATASETS:
        _unavailable(f"Unknown dataset: {dataset_key}")

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
        _unavailable(f"Failed to download {dataset_key} test data: {e}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_path)
    except Exception as e:
        if zip_path.exists():
            zip_path.unlink()
        _unavailable(f"Failed to extract {dataset_key} test data: {e}")

    if zip_path.exists():
        zip_path.unlink()

    if not expected_path.exists():
        for p in cache_path.rglob("expected_results.json"):
            expected_path = p
            cache_path = p.parent
            break
        else:
            shutil.rmtree(cache_path, ignore_errors=True)
            _unavailable(f"{dataset_key} extraction failed: expected_results.json not found")

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
