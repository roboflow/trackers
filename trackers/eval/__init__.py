# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from trackers.eval.box import box_ioa, box_iou
from trackers.eval.clear import aggregate_clear_metrics, compute_clear_metrics
from trackers.eval.evaluate import evaluate_benchmark, evaluate_mot_sequence
from trackers.eval.hota import aggregate_hota_metrics, compute_hota_metrics
from trackers.eval.identity import aggregate_identity_metrics, compute_identity_metrics
from trackers.eval.io import (
    MOTFrameData,
    MOTSequenceData,
    load_mot_file,
    prepare_mot_sequence,
)
from trackers.eval.results import (
    BenchmarkResult,
    CLEARMetrics,
    HOTAMetrics,
    IdentityMetrics,
    SequenceResult,
)

__all__ = [
    "BenchmarkResult",
    "CLEARMetrics",
    "HOTAMetrics",
    "IdentityMetrics",
    "MOTFrameData",
    "MOTSequenceData",
    "SequenceResult",
    "aggregate_clear_metrics",
    "aggregate_hota_metrics",
    "aggregate_identity_metrics",
    "box_ioa",
    "box_iou",
    "compute_clear_metrics",
    "compute_hota_metrics",
    "compute_identity_metrics",
    "evaluate_benchmark",
    "evaluate_mot_sequence",
    "load_mot_file",
    "prepare_mot_sequence",
]
