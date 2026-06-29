# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from trackers.core.reid.architectures import build_architecture, list_architectures
from trackers.core.reid.distance import appearance_similarity
from trackers.core.reid.eval.datasets import ReidSplit, load_market1501, load_msmt17
from trackers.core.reid.eval.evaluator import ReidEvaluator, ReidResult
from trackers.core.reid.eval.metrics import ReidMetrics, compute_reid_metrics
from trackers.core.reid.feature_bank import FeatureBank
from trackers.core.reid.model import ReIDModel
from trackers.core.reid.models.loaders import KeyReport, resolve_weights
from trackers.core.reid.models.preprocessing import ReIDPreprocessing
from trackers.core.reid.models.registry import ModelCard, resolve_model_card

__all__ = [
    "FeatureBank",
    "KeyReport",
    "ModelCard",
    "ReIDModel",
    "ReIDPreprocessing",
    "ReidEvaluator",
    "ReidMetrics",
    "ReidResult",
    "ReidSplit",
    "appearance_similarity",
    "build_architecture",
    "compute_reid_metrics",
    "list_architectures",
    "load_market1501",
    "load_msmt17",
    "resolve_model_card",
    "resolve_weights",
]
