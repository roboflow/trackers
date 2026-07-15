# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import importlib

__all__ = [
    "FeatureBank",
    "ReIDEvaluator",
    "ReIDMetrics",
    "ReIDModel",
    "ReIDPreprocessing",
    "ReIDResult",
    "ReIDSplit",
    "appearance_similarity",
    "compute_reid_metrics",
    "load_market1501",
    "load_msmt17",
    "resolve_model_card",
]

REID_INSTALL_HINT = (
    "ReID features require the optional `trackers[reid]` extra. Install with: pip install 'trackers[reid]'"
)

# NumPy-only symbols — safe to import without torch/timm/HF.
from trackers.core.reid.appearance import appearance_similarity
from trackers.core.reid.eval.datasets import ReIDSplit, load_market1501, load_msmt17
from trackers.core.reid.eval.metrics import ReIDMetrics, compute_reid_metrics
from trackers.core.reid.feature_bank import FeatureBank

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ReIDModel": ("trackers.core.reid.model", "ReIDModel"),
    "ReIDPreprocessing": ("trackers.core.reid.models.preprocessing", "ReIDPreprocessing"),
    "resolve_model_card": ("trackers.core.reid.models.registry", "resolve_model_card"),
    "ReIDEvaluator": ("trackers.core.reid.eval.evaluator", "ReIDEvaluator"),
    "ReIDResult": ("trackers.core.reid.eval.evaluator", "ReIDResult"),
}


def _import_reid_symbol(module_name: str, attr_name: str) -> object:
    """Import a heavy ReID symbol, rewriting missing-extra errors."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except ImportError as exc:
        raise ImportError(REID_INSTALL_HINT) from exc


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = _import_reid_symbol(module_name, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
