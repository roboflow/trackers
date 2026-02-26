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
    track_boxes: np.ndarray, detection_boxes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized direction vectors from track centers to detection centers.

    For each (track, detection) pair, computes the unit vector pointing from the
    track box center to the detection box center. Used in OC-SORT's direction
    consistency mechanism (OCM) to compare motion directions.

    Args:
        track_boxes: `np.ndarray` of shape `(n_tracks, 4)` containing track
            bounding boxes in `[x1, y1, x2, y2]` format. These serve as the
            origin points for direction vectors.
        detection_boxes: `np.ndarray` of shape `(n_detections, 4)` containing
            detection bounding boxes in `[x1, y1, x2, y2]` format. These serve
            as the target points for direction vectors.

    Returns:
        Tuple of `(dy, dx)` arrays, each of shape `(n_tracks, n_detections)`.
            Values are normalized to unit length. The `dy` component represents
            vertical direction, `dx` represents horizontal direction.
    """
    track_boxes = track_boxes[..., np.newaxis]
    CX1 = (detection_boxes[:, 0] + detection_boxes[:, 2]) / 2.0
    CY1 = (detection_boxes[:, 1] + detection_boxes[:, 3]) / 2.0
    CX2 = (track_boxes[:, 0] + track_boxes[:, 2]) / 2.0
    CY2 = (track_boxes[:, 1] + track_boxes[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx


def _build_direction_consistency_matrix_batch(
    tracklet_velocities: np.ndarray,
    reference_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    velocity_mask: np.ndarray,
) -> np.ndarray:
    """Build direction consistency cost matrix for OC-SORT's OCM mechanism.

    Measures how well each potential (tracklet, detection) association aligns
    with the tracklet's current motion direction. A high score indicates the
    detection lies along the tracklet's expected trajectory, reducing identity
    switches during non-linear motion.

    Args:
        tracklet_velocities: `np.ndarray` of shape `(n_tracklets, 2)` containing
            normalized velocity vectors `[vy, vx]` for each tracklet, computed
            using `delta_t` lookback observations.
        reference_boxes: `np.ndarray` of shape `(n_tracklets, 4)` containing
            bounding boxes in `[x1, y1, x2, y2]` format representing each
            tracklet's k-th previous observation. Used as origin points for
            computing association direction to each detection.
        detection_boxes: `np.ndarray` of shape `(n_detections, 4)` containing
            detection bounding boxes in `[x1, y1, x2, y2]` format.
        velocity_mask: `np.ndarray` of shape `(n_tracklets, 1)` with values
            `1.0` for tracklets with valid velocity estimates and `0.0` for
            newly initialized tracklets lacking motion history.

    Returns:
        `np.ndarray` of shape `(n_tracklets, n_detections)` containing direction
            consistency scores. Higher values indicate better alignment between
            tracklet motion and association direction.
    """
    n_tracklets = tracklet_velocities.shape[0]
    n_detections = detection_boxes.shape[0] if len(detection_boxes) > 0 else 0

    if n_tracklets == 0 or n_detections == 0:
        return np.zeros((n_tracklets, n_detections), dtype=np.float32)

    # Compute association directions (from reference_boxes -> detection) in batch
    Y, X = _speed_direction_batch(reference_boxes, detection_boxes)

    # Expand velocities for broadcasting
    inertia_Y = tracklet_velocities[:, 0:1]  # (n_tracklets, 1)
    inertia_X = tracklet_velocities[:, 1:2]  # (n_tracklets, 1)

    # Compute cosine similarity (dot product of normalized vectors)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, -1.0, 1.0)

    diff_angle = np.arccos(diff_angle_cos)
    angle_diff_cost = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    angle_diff_cost = velocity_mask * angle_diff_cost

    return angle_diff_cost.astype(np.float32)


def _get_iou_matrix(track_boxes: np.ndarray, detection_boxes: np.ndarray) -> np.ndarray:
    """Build IoU matrix between track and detection bounding boxes.

    Computes pairwise Intersection over Union (IoU) scores used as the primary
    cost metric for Hungarian algorithm association in SORT-family trackers.

    Args:
        track_boxes: `np.ndarray` of shape `(n_tracks, 4)` containing track
            bounding boxes in `[x1, y1, x2, y2]` format. Typically predicted
            positions from Kalman filter or last observations.
        detection_boxes: `np.ndarray` of shape `(n_detections, 4)` containing
            detection bounding boxes in `[x1, y1, x2, y2]` format from the
            current frame.

    Returns:
        `np.ndarray` of shape `(n_tracks, n_detections)` containing IoU scores
            in range `[0, 1]`. Higher values indicate greater overlap between
            track and detection boxes.
    """
    n_tracks = track_boxes.shape[0]
    n_detections = detection_boxes.shape[0]
    if n_tracks > 0 and n_detections > 0:
        iou_matrix = sv.box_iou_batch(track_boxes, detection_boxes)
    else:
        iou_matrix = np.zeros((n_tracks, n_detections), dtype=np.float32)
    return iou_matrix
