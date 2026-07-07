# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared appearance–IoU fusion helpers for weighted-fusion trackers.

``fuse_botsort_reid_association`` mirrors NirAharon/BoT-SORT ``bot_sort.py`` ReID
cost fusion. ``fuse_weighted_first_stage`` is kept for deferred ByteTrack+ReID.
"""

from __future__ import annotations

import numpy as np


def mask_appearance_by_iou_proximity(
    iou_similarity: np.ndarray,
    appearance_similarity: np.ndarray,
    proximity_iou: float = 0.5,
) -> np.ndarray:
    """Zero appearance similarity where IoU is below *proximity_iou*."""
    masked = appearance_similarity.copy()
    masked[iou_similarity < proximity_iou] = 0.0
    return masked


def fuse_weighted_first_stage(
    iou_similarity: np.ndarray,
    appearance_similarity: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Blend IoU and appearance similarity for first-stage association."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    return (1.0 - weight) * iou_similarity + weight * appearance_similarity


def fuse_botsort_reid_association(
    iou_similarity_raw: np.ndarray,
    iou_similarity_fused: np.ndarray,
    appearance_similarity: np.ndarray,
    *,
    proximity_thresh: float,
    appearance_thresh: float,
) -> np.ndarray:
    """Fuse IoU and appearance the way BoT-SORT ``bot_sort.py`` does.

    Computes ``min(score_fused_iou_cost, halved_appearance_cost)`` with
    proximity and appearance caps, then returns the corresponding similarity
    matrix (``1 - cost``).
    """
    d_iou = 1.0 - iou_similarity_fused
    d_iou_raw = 1.0 - iou_similarity_raw
    d_app = 0.5 * (1.0 - appearance_similarity)
    d_app = np.where(d_app > appearance_thresh, 1.0, d_app)
    d_app = np.where(d_iou_raw > proximity_thresh, 1.0, d_app)
    fused_cost = np.minimum(d_iou, d_app)
    return 1.0 - fused_cost
