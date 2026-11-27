# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Box-only appearance-extraction gating, independent of segmentation masks."""

from __future__ import annotations

import numpy as np


def foreground_overlap(box: np.ndarray, reference_boxes: np.ndarray) -> float:
    """Return the largest target-area fraction covered by one lower-bottom box.

    Only finite, positive-area reference boxes with ``y2 > box[3]`` participate.
    The target itself and equal-bottom boxes therefore contribute nothing. This
    is a perspective heuristic, not a visibility measurement: it uses neither
    intersection-over-union nor the sum/union of several occluders.

    Args:
        box: Target ``xyxy`` detector box with positive area.
        reference_boxes: All current ``xyxy`` detector boxes, shape ``(N, 4)``.

    Returns:
        Maximum intersection divided by the target area, or zero when no
        valid lower-bottom box overlaps it.
    """
    boxes = np.asarray(reference_boxes)
    area = float((box[2] - box[0]) * (box[3] - box[1]))
    if len(boxes) == 0 or area <= 0:
        return 0.0
    valid = np.isfinite(boxes).all(axis=1)
    valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 3] > box[3])
    boxes = boxes[valid]
    if len(boxes) == 0:
        return 0.0
    intersection = np.maximum(0, np.minimum(boxes[:, 2:], box[2:]) - np.maximum(boxes[:, :2], box[:2]))
    return float(np.max(intersection[:, 0] * intersection[:, 1]) / area)
