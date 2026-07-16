# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import importlib

__all__ = [
    "ReIDEvaluator",
    "ReIDMetrics",
    "ReIDModel",
    "ReIDPreprocessing",
    "ReIDResult",
    "ReIDSplit",
    "compute_reid_metrics",
    "load_market1501",
    "load_msmt17",
]

REID_INSTALL_HINT = (
    "ReID features require the optional `trackers[reid]` extra. Install with: pip install 'trackers[reid]'"
)

_REID_OPTIONAL_ROOTS = frozenset(
    {
        "torch",
        "torchvision",
        "timm",
        "huggingface_hub",
        "safetensors",
        "PIL",
        "gdown",
    }
)

# NumPy-only symbols — safe to import without torch/timm/HF.
from trackers.core.reid.eval.datasets import ReIDSplit, load_market1501, load_msmt17
from trackers.core.reid.eval.evaluator import ReIDEvaluator, ReIDResult
from trackers.core.reid.eval.metrics import ReIDMetrics, compute_reid_metrics

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ReIDModel": ("trackers.core.reid.model", "ReIDModel"),
    "ReIDPreprocessing": ("trackers.core.reid.models.preprocessing", "ReIDPreprocessing"),
}


def _is_optional_reid_import_error(exc: ImportError) -> bool:
    name = getattr(exc, "name", None)
    if name is None:
        return False
    root = name.split(".", 1)[0]
    return root in _REID_OPTIONAL_ROOTS


def _import_reid_symbol(module_name: str, attr_name: str) -> object:
    """Import a heavy ReID symbol, rewriting missing-extra errors only."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except ImportError as exc:
        if _is_optional_reid_import_error(exc):
            raise ImportError(REID_INSTALL_HINT) from exc
        raise


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = _import_reid_symbol(module_name, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
