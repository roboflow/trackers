# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from trackers.core.reid.eval.datasets import ReIDSplit, load_market1501, load_msmt17
from trackers.core.reid.eval.metrics import ReIDMetrics, compute_reid_metrics

__all__ = [
    "ReIDEvaluator",
    "ReIDMetrics",
    "ReIDResult",
    "ReIDSplit",
    "compute_reid_metrics",
    "load_market1501",
    "load_msmt17",
]


def __getattr__(name: str) -> object:
    if name == "ReIDEvaluator":
        from trackers.core.reid._lazy import import_reid_symbol

        value = import_reid_symbol("trackers.core.reid.eval.evaluator", "ReIDEvaluator")
        globals()["ReIDEvaluator"] = value
        return value
    if name == "ReIDResult":
        from trackers.core.reid._lazy import import_reid_symbol

        value = import_reid_symbol("trackers.core.reid.eval.evaluator", "ReIDResult")
        globals()["ReIDResult"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
