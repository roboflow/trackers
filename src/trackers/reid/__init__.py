# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Appearance association, long-term identity joining and offline ReID threshold tools."""

from __future__ import annotations

from trackers.reid.appearance import (
    appearance_similarity,
    extract_detection_embeddings,
    extract_ground_truth_embeddings,
)
from trackers.reid.config import ReIDConfig
from trackers.reid.encoder import ReIDEncoder
from trackers.reid.feature_bank import FeatureBank
from trackers.reid.long_term import LongTermReID
from trackers.reid.long_term_config import LongTermReIDConfig
from trackers.reid.plots import ThresholdLines, plot_appearance_distances, plot_frame_gap_sweep
from trackers.reid.sports_model import prepare_sports_reid_model
from trackers.reid.threshold_selection import (
    DEFAULT_FRAME_GAP_BANDS,
    AppearanceDistances,
    roc_auc,
    sample_appearance_distances,
    sweep_frame_gap,
)

__all__ = [
    "DEFAULT_FRAME_GAP_BANDS",
    "AppearanceDistances",
    "FeatureBank",
    "LongTermReID",
    "LongTermReIDConfig",
    "ReIDConfig",
    "ReIDEncoder",
    "ThresholdLines",
    "appearance_similarity",
    "extract_detection_embeddings",
    "extract_ground_truth_embeddings",
    "plot_appearance_distances",
    "plot_frame_gap_sweep",
    "prepare_sports_reid_model",
    "roc_auc",
    "sample_appearance_distances",
    "sweep_frame_gap",
]
