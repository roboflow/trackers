# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified and adapted from OC-SORT https://github.com/noahcao/OC_SORT/
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from copy import deepcopy
from typing import Sequence, cast

import numpy as np
import supervision as sv

from trackers.core.ocsort.tracklet import OCSORTTracklet


def k_previous_obs(
    observations: dict[int, np.ndarray], cur_age: int, delta_t: int
) -> np.ndarray | None:
    """Get observation from delta_t steps ago from the observations dict.

    Looks back up to delta_t timesteps. Falls back to the most recent
    observation if none found in the lookback window.

    Args:
        observations: Dict mapping age to observed bbox.
        cur_age: Current age of the tracklet.
        delta_t: Number of timesteps to look back.

    Returns:
        The observation from delta_t steps ago, or the most recent one,
        or None if no observations exist.
    """
    if len(observations) == 0:
        return None
    for i in range(delta_t):
        dt = delta_t - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def _speed_direction(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Compute normalized direction vector between two bounding box centers.

    Args:
        bbox1: First bounding box in the form [x1, y1, x2, y2].
        bbox2: Second bounding box in the form [x1, y1, x2, y2].

    Returns:
        np.ndarray: Normalized direction vector [dy, dx] from bbox1 to bbox2.
    """
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def _build_direction_consistency_matrix(
    tracklets: list[OCSORTTracklet],
    detection_boxes: np.ndarray,
) -> np.ndarray:
    """Build direction consistency cost matrix (OCM) between tracklet velocities
    and detection associations.

    Note: This is the non-batch version kept for reference, interpretability and testing
    purposes. Use `build_direction_consistency_matrix_batch` for production.

    Uses tracklet.velocity (computed with delta_t lookback) and k_previous_obs
    as the reference point for association direction, matching the original
    OC-SORT implementation.

    Args:
        tracklets: List of OCSORTTracklet objects.
        detection_boxes: Detection bounding boxes [x1, y1, x2, y2].

    Returns:
        np.ndarray: Direction consistency cost matrix (n_tracklets, n_detections).
    """
    n_tracklets = len(tracklets)
    n_detections = detection_boxes.shape[0]
    direction_consistency_matrix = np.zeros(
        (n_tracklets, n_detections), dtype=np.float32
    )

    for t, tracklet in enumerate(tracklets):
        if tracklet.velocity is None:
            continue

        k_obs = tracklet.get_k_previous_obs()
        if k_obs is None:
            continue

        tracklet_speed = tracklet.velocity

        for d in range(n_detections):
            detection_box = detection_boxes[d]
            association_speed = _speed_direction(k_obs, detection_box)

            cos_sim = np.dot(tracklet_speed, association_speed)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            angle = np.arccos(cos_sim)
            direction_consistency_matrix[t, d] = (np.pi / 2.0 - np.abs(angle)) / np.pi

    return direction_consistency_matrix


def _speed_direction_batch(
    dets: np.ndarray, tracks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized direction vectors from tracks to detections in batch.

    Args:
        dets: Detection bounding boxes [x1, y1, x2, y2], shape (n_dets, 4).
        tracks: Track bounding boxes [x1, y1, x2, y2], shape (n_tracks, 4).

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


def build_direction_consistency_matrix_batch(
    tracklets: list[OCSORTTracklet],
    detection_boxes: np.ndarray,
) -> np.ndarray:
    """Build direction consistency cost matrix (OCM) in batch - vectorized version.

    Computes similarity between tracklet velocity vectors (computed with delta_t
    lookback) and potential association directions from k-previous observations.
    Used in OC-SORT for motion-aware association.

    Args:
        tracklets: List of OCSORTTracklet objects.
        detection_boxes: Detection bounding boxes [x1, y1, x2, y2].

    Returns:
        np.ndarray: Direction consistency cost matrix (n_tracklets, n_detections).
    """
    n_tracklets = len(tracklets)
    n_detections = detection_boxes.shape[0] if len(detection_boxes) > 0 else 0

    if n_tracklets == 0 or n_detections == 0:
        return np.zeros((n_tracklets, n_detections), dtype=np.float32)

    # Use precomputed velocities from tracklets (computed with delta_t lookback)
    velocities = np.array(
        [
            tracklet.velocity if tracklet.velocity is not None else np.array([0.0, 0.0])
            for tracklet in tracklets
        ]
    )

    # Get k-previous observations as reference for association direction
    k_obs_list = [tracklet.get_k_previous_obs() for tracklet in tracklets]
    k_observations = np.array(
        [
            obs if obs is not None else tracklet.last_observation
            for obs, tracklet in zip(k_obs_list, tracklets)
        ]
    )

    # Compute association directions (from k_observations -> detection) in batch
    Y, X = _speed_direction_batch(
        detection_boxes, k_observations
    )  # (n_tracklets, n_detections)

    # Expand velocities for broadcasting
    inertia_Y = velocities[:, 0:1]  # (n_tracklets, 1)
    inertia_X = velocities[:, 1:2]  # (n_tracklets, 1)

    # Compute cosine similarity (dot product of normalized vectors)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, -1.0, 1.0)

    diff_angle = np.arccos(diff_angle_cos)
    angle_diff_cost = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    # Mask out tracklets without velocity (no previous observation)
    valid_mask = np.array(
        [tracklet.velocity is not None for tracklet in tracklets],
        dtype=np.float32,
    )[:, np.newaxis]  # (n_tracklets, 1)

    angle_diff_cost = valid_mask * angle_diff_cost

    return angle_diff_cost.astype(np.float32)


def add_track_id_detections(
    track: OCSORTTracklet,
    detection: sv.Detections,
    updated_detections: list[sv.Detections],
    minimum_consecutive_frames: int,
    frame_count: int,
) -> None:
    """Assign track ID to detection and add to updated_detections list.

    Handles ID assignment based on track maturity:
    - Early frames (frame_count < minimum_consecutive_frames): Assign ID if
      track was just updated and doesn't have an ID yet.
    - Later frames: Assign ID only if track is mature (has enough consecutive
      updates). Immature tracks get tracker_id = -1.

    Args:
        track: The tracklet being processed.
        detection: The detection to assign an ID to.
        updated_detections: List to append the updated detection to.
        minimum_consecutive_frames: Frames required for track maturity.
        frame_count: Current frame number in tracking process.
    """
    new_det = deepcopy(detection)
    new_det = cast(sv.Detections, new_det)
    is_mature = (
        track.number_of_successful_consecutive_updates >= minimum_consecutive_frames
    )
    if frame_count <= minimum_consecutive_frames:
        if track.time_since_update == 0:
            if track.tracker_id == -1:
                track.tracker_id = OCSORTTracklet.get_next_tracker_id()

            new_det.tracker_id = np.array([track.tracker_id])
    else:
        if is_mature:
            # Assign ID now if track just became mature
            if track.tracker_id == -1:
                track.tracker_id = OCSORTTracklet.get_next_tracker_id()
            new_det.tracker_id = np.array([track.tracker_id])
        else:
            new_det.tracker_id = np.array([-1], dtype=int)
    updated_detections.append(new_det)


def get_iou_matrix(
    trackers: Sequence[OCSORTTracklet], detection_boxes: np.ndarray
) -> np.ndarray:
    """Build IOU cost matrix between trackers and detections.

    Args:
        trackers: Sequence of OCSORTTracklet objects.
        detection_boxes: Detection bounding boxes [x1, y1, x2, y2].

    Returns:
        np.ndarray: IOU matrix of shape (n_trackers, n_detections).
    """
    predicted_boxes = np.array([t.get_state_bbox() for t in trackers])
    if len(predicted_boxes) == 0 and len(trackers) > 0:
        predicted_boxes = np.zeros((len(trackers), 4), dtype=np.float32)

    if len(trackers) > 0 and len(detection_boxes) > 0:
        iou_matrix = sv.box_iou_batch(predicted_boxes, detection_boxes)
    else:
        iou_matrix = np.zeros((len(trackers), len(detection_boxes)), dtype=np.float32)

    return iou_matrix


def get_iou_matrix_between_boxes(
    last_observations: np.ndarray, detection_boxes: np.ndarray
) -> np.ndarray:
    """Build IOU cost matrix between two sets of bounding boxes.

    Args:
        last_observations: First set of boxes [x1, y1, x2, y2].
        detection_boxes: Second set of boxes [x1, y1, x2, y2].

    Returns:
        np.ndarray: IOU matrix of shape (n_observations, n_detections).
    """
    if len(last_observations) == 0 and len(last_observations) > 0:
        last_observations = np.zeros((len(last_observations), 4), dtype=np.float32)

    if len(last_observations) > 0 and len(detection_boxes) > 0:
        iou_matrix = sv.box_iou_batch(last_observations, detection_boxes)
    else:
        iou_matrix = np.zeros(
            (len(last_observations), len(detection_boxes)), dtype=np.float32
        )

    return iou_matrix
