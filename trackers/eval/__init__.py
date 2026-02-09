# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Evaluation metrics and utilities for tracking benchmarks."""

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


def __getattr__(name: str):
    """Lazy imports for evaluate functions to avoid circular imports."""
    if name in ("evaluate_mot_sequence", "evaluate_mot_sequences"):
        from trackers.eval import evaluate as _evaluate
        return getattr(_evaluate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BenchmarkResult",
    "CLEARMetrics",
    "HOTAMetrics",
    "IdentityMetrics",
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
]
