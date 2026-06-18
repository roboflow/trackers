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
    """Zero appearance similarity where raw IoU is below a proximity threshold.

    Matches the BoT-SORT / DeepSORT pattern of only trusting appearance when
    boxes already overlap sufficiently in image space.

    Args:
        iou_similarity: Raw IoU similarity matrix, shape ``(T, N)``.
        appearance_similarity: Cosine similarity matrix, shape ``(T, N)``.
        proximity_iou: Minimum IoU required before appearance is considered.

    Returns:
        Appearance matrix with out-of-proximity entries set to ``0.0``.
    """
    masked = appearance_similarity.copy()
    masked[iou_similarity < proximity_iou] = 0.0
    return masked


def fuse_weighted_first_stage(
    iou_similarity: np.ndarray,
    appearance_similarity: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Blend IoU and appearance similarity for first-stage association.

    Implements the standard DeepSORT/JDE-style weighted fusion used by
    ByteTrack and other trackers that combine geometry and appearance
    into a single score matrix.

    Args:
        iou_similarity: IoU (or score-fused IoU) similarity matrix,
            shape ``(T, N)``.
        appearance_similarity: Cosine similarity matrix, shape ``(T, N)``.
        weight: Appearance weight in ``[0, 1]``.  ``0.0`` keeps IoU only;
            ``1.0`` keeps appearance only.

    Returns:
        Fused similarity matrix of shape ``(T, N)``.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    return (1.0 - weight) * iou_similarity + weight * appearance_similarity
