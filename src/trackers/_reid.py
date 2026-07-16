# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Optional ReID model provider."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trackers.core.reid.model import ReIDModel as ReIDModel

REID_INSTALL_HINT = (
    "ReID features require the optional `trackers[reid]` extra. Install with: pip install 'trackers[reid]'"
)

_MODEL_MODULE = "trackers.core.reid.model"
_OPTIONAL_DEPENDENCY_ROOTS = frozenset(
    {
        "PIL",
        "gdown",
        "huggingface_hub",
        "safetensors",
        "timm",
        "torch",
        "torchvision",
    }
)


def import_reid_model() -> Any:
    """Return the configured ``ReIDModel`` class."""
    try:
        module = importlib.import_module(_MODEL_MODULE)
    except ImportError as exc:
        name = getattr(exc, "name", None)
        if name is not None and name.split(".", 1)[0] in _OPTIONAL_DEPENDENCY_ROOTS:
            raise ImportError(REID_INSTALL_HINT) from exc
        raise
    return module.ReIDModel
