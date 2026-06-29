# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared appearance–IoU fusion helpers for weighted-fusion trackers."""

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
