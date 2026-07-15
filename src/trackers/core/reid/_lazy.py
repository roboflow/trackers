# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Optional ReID dependency helpers."""

from __future__ import annotations

import importlib

REID_INSTALL_HINT = (
    "ReID features require the optional `trackers[reid]` extra. Install with: pip install 'trackers[reid]'"
)

# Modules that must be importable for the ReID stack. Checked in order so the
# first missing dependency produces a clear install hint.
_REID_DEPENDENCY_MODULES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "timm",
    "huggingface_hub",
    "safetensors",
    "PIL",
    "gdown",
)


def require_reid_extra() -> None:
    """Raise a clear error when any ReID optional dependency is missing."""
    for module_name in _REID_DEPENDENCY_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(REID_INSTALL_HINT) from exc
        if module is None:
            raise ImportError(REID_INSTALL_HINT)


def import_reid_symbol(module_name: str, attr_name: str) -> object:
    """Import a symbol from a heavy ReID submodule with a friendly fallback."""
    require_reid_extra()
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except ImportError as exc:
        raise ImportError(REID_INSTALL_HINT) from exc
