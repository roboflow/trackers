# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Root pytest hooks.

When ``trackers[reid]`` optional deps are missing (e.g. integration CI with only
``uv sync --group dev``), skip collecting modules that import them so the rest
of the suite still collects. NumPy-only ReID tests always run.

Heavy ReID tests run in the main test workflow, which installs ``--extra reid``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REID_EXTRA_MODULES = ("timm", "huggingface_hub", "safetensors")

# Source modules that import the optional ReID stack at module level.
_HEAVY_SRC_MARKERS = (
    "/trackers/core/reid/architectures/",
    "/trackers/core/reid/model.py",
    "/trackers/core/reid/models/",
)

# Test modules that require the optional ReID stack.
_HEAVY_TEST_NAMES = frozenset(
    {
        "test_reid_model.py",
    }
)


def _reid_extra_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in _REID_EXTRA_MODULES)


def _is_heavy_reid_path(path: Path) -> bool:
    normalized = path.as_posix()
    if any(marker in normalized for marker in _HEAVY_SRC_MARKERS):
        return True
    return path.name in _HEAVY_TEST_NAMES


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if _reid_extra_available() or not _is_heavy_reid_path(collection_path):
        return None
    return True
