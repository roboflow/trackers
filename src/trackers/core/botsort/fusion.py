# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
#
# Adapted from NirAharon/BoT-SORT (MIT)
# Copyright (c) 2022 Nir Aharon
# Source: https://github.com/NirAharon/BoT-SORT
# Reference: tracker/bot_sort.py (ReID appearance-IoU cost fusion)
# ------------------------------------------------------------------------

"""Appearance-IoU fusion for BoT-SORT ReID association."""

from __future__ import annotations

import numpy as np


def fuse_botsort_reid_association(
    association_similarity: np.ndarray,
    appearance_similarity: np.ndarray,
    *,
    proximity_threshold: float,
    appearance_threshold: float,
    proximity_iou_similarity: np.ndarray | None = None,
) -> np.ndarray:
    """Fuse IoU and appearance the way BoT-SORT ``bot_sort.py`` does.

    Computes ``min(association_cost, capped_appearance_cost)`` with proximity
    and appearance gates, then returns the corresponding similarity matrix
    (``1 - cost``).

    ``proximity_iou_similarity`` is the standard-IoU gate (defaults to
    ``association_similarity``). Pass it separately when association uses
    GIoU/DIoU/CIoU so proximity still uses plain IoU.
    """
    if proximity_iou_similarity is None:
        proximity_iou_similarity = association_similarity

    d_iou = 1.0 - association_similarity
    d_iou_proximity = 1.0 - proximity_iou_similarity
    d_app = 0.5 * (1.0 - appearance_similarity)
    d_app = np.where(d_app > appearance_threshold, 1.0, d_app)
    d_app = np.where(d_iou_proximity > proximity_threshold, 1.0, d_app)
    fused_cost = np.minimum(d_iou, d_app)
    return 1.0 - fused_cost
