# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified and adapted from Hybrid-SORT https://github.com/ymzis69/HybridSORT
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

# Every per-corner array uses this corner order: (x1, y1), (x1, y2), (x2, y1), (x2, y2).
_CORNER_X_INDICES = np.array([0, 0, 2, 2])
_CORNER_Y_INDICES = np.array([1, 3, 1, 3])


def _box_corners(boxes: np.ndarray) -> np.ndarray:
    """Return the four corners of each box.

    Args:
        boxes: Boxes of shape `(..., 4)` in `[x1, y1, x2, y2]` format.

    Returns:
        Corners of shape `(..., 4, 2)` holding `[x, y]` for each corner.
    """
    return np.stack([boxes[..., _CORNER_X_INDICES], boxes[..., _CORNER_Y_INDICES]], axis=-1)


def _summed_corner_directions(from_boxes: np.ndarray, to_box: np.ndarray) -> np.ndarray:
    """Sum, per box corner, the unit directions from several earlier observations to one box.

    Args:
        from_boxes: Earlier bounding boxes of shape `(k, 4)` in `[x1, y1, x2, y2]` format.
        to_box: Later bounding box `[x1, y1, x2, y2]`.

    Returns:
        Array of shape `(4, 2)` holding, for each corner in the module's corner
            order, the sum over `from_boxes` of the normalized `[dy, dx]`
            direction towards `to_box`. The sum is not re-normalized.
    """
    dx = to_box[_CORNER_X_INDICES] - from_boxes[:, _CORNER_X_INDICES]
    dy = to_box[_CORNER_Y_INDICES] - from_boxes[:, _CORNER_Y_INDICES]
    norm = np.sqrt(dx * dx + dy * dy) + 1e-6
    return np.column_stack(((dy / norm).sum(axis=0), (dx / norm).sum(axis=0)))


def _build_corner_direction_consistency_matrix(
    corner_velocities: np.ndarray,
    reference_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    velocity_mask: np.ndarray,
) -> np.ndarray:
    """Build Hybrid-SORT's robust observation-centric momentum (ROCM) score matrix.

    OC-SORT's momentum term compares a tracklet's motion direction with the
    direction implied by a candidate match, using box centers. ROCM makes the
    same comparison for each of the four box corners and averages the four
    scores, which is less sensitive to a single noisy box edge.

    Args:
        corner_velocities: Array of shape `(n_tracklets, 4, 2)` holding each
            tracklet's per-corner `[vy, vx]` motion direction. The vectors are
            not required to have unit length; the cosine is clipped to `[-1, 1]`.
        reference_boxes: Array of shape `(n_tracklets, 4)` holding the
            observation each association direction is measured from, in
            `[x1, y1, x2, y2]` format.
        detection_boxes: Array of shape `(n_detections, 4)` in
            `[x1, y1, x2, y2]` format.
        velocity_mask: Array of shape `(n_tracklets, 1)` with `1.0` for
            tracklets that have a velocity estimate and `0.0` otherwise.

    Returns:
        Array of shape `(n_tracklets, n_detections)` with scores in
            `[-0.5, 0.5]`. Higher values mean the candidate detection lies along
            the tracklet's recent direction of motion.
    """
    n_tracklets = corner_velocities.shape[0]
    n_detections = detection_boxes.shape[0]
    if n_tracklets == 0 or n_detections == 0:
        return np.zeros((n_tracklets, n_detections), dtype=np.float64)

    delta = _box_corners(detection_boxes)[np.newaxis] - _box_corners(reference_boxes)[:, np.newaxis]
    norm = np.sqrt(delta[..., 0] ** 2 + delta[..., 1] ** 2) + 1e-6
    direction_x = delta[..., 0] / norm
    direction_y = delta[..., 1] / norm

    velocity_y = corner_velocities[:, np.newaxis, :, 0]
    velocity_x = corner_velocities[:, np.newaxis, :, 1]
    cosine = np.clip(velocity_x * direction_x + velocity_y * direction_y, -1.0, 1.0)
    corner_scores = (np.pi / 2.0 - np.arccos(cosine)) / np.pi

    return velocity_mask * corner_scores.mean(axis=-1)
