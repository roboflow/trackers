# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Appearance-ReID association plus offline threshold selection and plotting tools."""

from __future__ import annotations

from trackers.core.reid.appearance import (
    appearance_similarity,
    extract_detection_embeddings,
    extract_ground_truth_embeddings,
)
from trackers.core.reid.encoder import ReIDEncoder
from trackers.core.reid.feature_bank import FeatureBank
from trackers.core.reid.thresholds import (
    DEFAULT_FRAME_GAP_BANDS,
    AppearanceDistances,
    ThresholdLines,
    plot_appearance_distances,
    plot_frame_gap_sweep,
    roc_auc,
    sample_appearance_distances,
    sweep_frame_gap,
)

__all__ = [
    "DEFAULT_FRAME_GAP_BANDS",
    "AppearanceDistances",
    "FeatureBank",
    "ReIDEncoder",
    "ThresholdLines",
    "appearance_similarity",
    "extract_detection_embeddings",
    "extract_ground_truth_embeddings",
    "plot_appearance_distances",
    "plot_frame_gap_sweep",
    "roc_auc",
    "sample_appearance_distances",
    "sweep_frame_gap",
]
