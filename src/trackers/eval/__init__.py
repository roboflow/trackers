# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Evaluation metrics and utilities for tracking benchmarks."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from trackers.eval.box import box_ioa, box_iou
from trackers.eval.clear import aggregate_clear_metrics, compute_clear_metrics
from trackers.eval.hota import aggregate_hota_metrics, compute_hota_metrics
from trackers.eval.identity import aggregate_identity_metrics, compute_identity_metrics
from trackers.eval.results import (
    BenchmarkResult,
    CLEARMetrics,
    HOTAMetrics,
    IdentityMetrics,
    SequenceResult,
)

if TYPE_CHECKING:
    from trackers.eval.evaluate import evaluate_mot_sequence, evaluate_mot_sequences
    from trackers.eval.multicamera import (
        MulticameraBenchmarkResult,
        SceneMeanHOTA,
        evaluate_multicamera_scene,
        evaluate_multicamera_scenes,
    )

_LAZY_MODULES = {
    "evaluate_mot_sequence": "trackers.eval.evaluate",
    "evaluate_mot_sequences": "trackers.eval.evaluate",
    "MulticameraBenchmarkResult": "trackers.eval.multicamera",
    "SceneMeanHOTA": "trackers.eval.multicamera",
    "evaluate_multicamera_scene": "trackers.eval.multicamera",
    "evaluate_multicamera_scenes": "trackers.eval.multicamera",
}


def __getattr__(name: str) -> object:
    """Lazy imports for modules that read MOT files, to avoid circular imports."""
    module_name = _LAZY_MODULES.get(name)
    if module_name is not None:
        return getattr(importlib.import_module(module_name), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_MODULES))


__all__ = [
    "BenchmarkResult",
    "CLEARMetrics",
    "HOTAMetrics",
    "IdentityMetrics",
    "MulticameraBenchmarkResult",
    "SceneMeanHOTA",
    "SequenceResult",
    "aggregate_clear_metrics",
    "aggregate_hota_metrics",
    "aggregate_identity_metrics",
    "box_ioa",
    "box_iou",
    "compute_clear_metrics",
    "compute_hota_metrics",
    "compute_identity_metrics",
    "evaluate_mot_sequence",
    "evaluate_mot_sequences",
    "evaluate_multicamera_scene",
    "evaluate_multicamera_scenes",
]
