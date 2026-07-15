# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from trackers.core.reid._lazy import import_reid_symbol

__all__ = [
    "FASTREID_MOT17_SBS50",
    "MARKET1501_GALLERY_JUNK_PIDS",
    "FeatureBank",
    "ReIDEvaluator",
    "ReIDMetrics",
    "ReIDModel",
    "ReIDPreprocessing",
    "ReIDResult",
    "ReIDSplit",
    "appearance_similarity",
    "build_architecture",
    "compute_reid_metrics",
    "list_architectures",
    "load_market1501",
    "load_msmt17",
    "resolve_model_card",
]

# NumPy-only symbols — safe to import without torch/timm/HF.
from trackers.core.reid.distance import appearance_similarity
from trackers.core.reid.eval.datasets import (
    MARKET1501_GALLERY_JUNK_PIDS,
    ReIDSplit,
    load_market1501,
    load_msmt17,
)
from trackers.core.reid.eval.metrics import ReIDMetrics, compute_reid_metrics
from trackers.core.reid.feature_bank import FeatureBank

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ReIDModel": ("trackers.core.reid.model", "ReIDModel"),
    "ReIDPreprocessing": ("trackers.core.reid.models.preprocessing", "ReIDPreprocessing"),
    "resolve_model_card": ("trackers.core.reid.models.registry", "resolve_model_card"),
    "FASTREID_MOT17_SBS50": ("trackers.core.reid.models.registry", "FASTREID_MOT17_SBS50"),
    "build_architecture": ("trackers.core.reid.architectures", "build_architecture"),
    "list_architectures": ("trackers.core.reid.architectures", "list_architectures"),
    "ReIDEvaluator": ("trackers.core.reid.eval.evaluator", "ReIDEvaluator"),
    "ReIDResult": ("trackers.core.reid.eval.evaluator", "ReIDResult"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = import_reid_symbol(module_name, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
