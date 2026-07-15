# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Root pytest hooks.

Skip collecting optional ReID modules when ``trackers[reid]`` deps are absent so
workflows that only ``uv sync --group dev`` (no ``--extra reid``) can still
collect and run the rest of the suite, including ``--doctest-modules`` on
``src/``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REID_EXTRA_MODULES = ("timm", "huggingface_hub", "safetensors")


def _reid_extra_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in _REID_EXTRA_MODULES)


def _is_reid_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "reid" in parts and ("trackers" in parts or "tests" in parts)


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if _reid_extra_available() or not _is_reid_path(collection_path):
        return None
    return True
