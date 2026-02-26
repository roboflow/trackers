# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified and adapted from OC-SORT https://github.com/noahcao/OC_SORT/
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import supervision as sv


def _speed_direction_batch(
    dets: np.ndarray, tracks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized direction vectors from tracks to detections in batch.

    Args:
        dets: Detection bounding boxes `[x1, y1, x2, y2]`, shape (n_dets, 4).
        tracks: Track bounding boxes `[x1, y1, x2, y2]`, shape (n_tracks, 4).

    Returns:
        tuple[np.ndarray, np.ndarray]: (dy, dx) direction vectors,
            each of shape (n_tracks, n_dets).
    """
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx


def _build_direction_consistency_matrix_batch(
    velocities: np.ndarray,
    k_observations: np.ndarray,
    detection_boxes: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Build direction consistency cost matrix (OCM) in batch - vectorized version.

    Computes similarity between tracklet velocity vectors (computed with delta_t
    lookback) and potential association directions from k-previous observations.
    Used in OC-SORT for motion-aware association.

    Args:
        velocities: Array of shape (n_tracklets, 2) with velocity vectors.
        k_observations: Array of shape (n_tracklets, 4) with reference boxes.
        detection_boxes: Array of shape (n_detections, 4).
        valid_mask: Array of shape (n_tracklets, 1) indicating valid velocities.

    Returns:
        np.ndarray: Direction consistency cost matrix (n_tracklets, n_detections).
    """
    n_tracklets = velocities.shape[0]
    n_detections = detection_boxes.shape[0] if len(detection_boxes) > 0 else 0

    if n_tracklets == 0 or n_detections == 0:
        return np.zeros((n_tracklets, n_detections), dtype=np.float32)

    # Compute association directions (from k_observations -> detection) in batch
    Y, X = _speed_direction_batch(detection_boxes, k_observations)

    # Expand velocities for broadcasting
    inertia_Y = velocities[:, 0:1]  # (n_tracklets, 1)
    inertia_X = velocities[:, 1:2]  # (n_tracklets, 1)

    # Compute cosine similarity (dot product of normalized vectors)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, -1.0, 1.0)

    diff_angle = np.arccos(diff_angle_cos)
    angle_diff_cost = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    angle_diff_cost = valid_mask * angle_diff_cost

    return angle_diff_cost.astype(np.float32)


def _get_iou_matrix(
    predicted_boxes: np.ndarray, detection_boxes: np.ndarray
) -> np.ndarray:
    """Build IOU cost matrix between tracks and detections.

    Args:
        predicted_boxes: Array of shape (n_tracks, 4) with predicted bounding boxes.
        detection_boxes: Detection bounding boxes `[x1, y1, x2, y2]`.

    Returns:
        np.ndarray: IOU matrix of shape (n_tracks, n_detections).
    """
    n_tracks = predicted_boxes.shape[0]
    n_detections = detection_boxes.shape[0]
    if n_tracks > 0 and n_detections > 0:
        iou_matrix = sv.box_iou_batch(predicted_boxes, detection_boxes)
    else:
        iou_matrix = np.zeros((n_tracks, n_detections), dtype=np.float32)
    return iou_matrix
