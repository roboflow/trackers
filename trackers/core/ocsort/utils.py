# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from OC-SORT https://github.com/noahcao/OC_SORT/
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------

from copy import deepcopy
from typing import Sequence, cast

import numpy as np
import supervision as sv

from trackers.core.ocsort.tracklet import OCSORTTracklet


def speed_direction(bbox1, bbox2):  # from oc sort repo
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def build_direction_consistency_matrix(
    # could we try to vectorize this? yes, lets compare both methods later
    tracklets: list[OCSORTTracklet],
    detection_boxes: np.ndarray,
) -> np.ndarray:
    n_tracklets = len(tracklets)
    n_detections = detection_boxes.shape[0]
    direction_consistency_matrix = np.zeros(
        (n_tracklets, n_detections), dtype=np.float32
    )

    for t, tracklet in enumerate(tracklets):
        if tracklet.previous_to_last_observation is None:  # if there is no previous box
            continue
        # In case the track is lost we use the last known box
        last_observation = tracklet.last_observation
        tracklet_speed = speed_direction(
            tracklet.previous_to_last_observation, last_observation
        )

        for d in range(n_detections):
            detection_box = detection_boxes[d]
            association_speed = speed_direction(last_observation, detection_box)

            # Compute cosine similarity
            cos_sim = np.dot(tracklet_speed, association_speed)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            # Apply the same transformation as original
            angle = np.arccos(cos_sim)
            direction_consistency_matrix[t, d] = (np.pi / 2.0 - np.abs(angle)) / np.pi
            #direction_consistency_matrix[t, d] = cos_sim

    return direction_consistency_matrix


def speed_direction_batch(dets, tracks):  # From oc sort repo
    """Function for calculating direction between detections and tracks in a batch way.

    Args:
        dets: Detected bounding boxes in the form [x1, y1, x2, y2].
        tracks: Tracked bounding boxes in the form [x1, y1, x2, y2]."""
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx  # size: num_track x num_det


def build_direction_consistency_matrix_batch(
    # vectorized version from oc sort repo but adapted to our tracklet class
    tracklets: list[OCSORTTracklet],
    detection_boxes: np.ndarray,
) -> np.ndarray:
    n_tracklets = len(tracklets)
    n_detections = detection_boxes.shape[0] if len(detection_boxes) > 0 else 0
    
    if n_tracklets == 0 or n_detections == 0:
        return np.zeros((n_tracklets, n_detections), dtype=np.float32)
    
    # Compute tracklet velocities (from previous_to_last -> last_observation)
    velocities = np.array(
        [
            speed_direction(
                tracklet.previous_to_last_observation, tracklet.last_observation
            )
            if tracklet.previous_to_last_observation is not None
            else np.array([0.0, 0.0])
            for tracklet in tracklets
        ]
    )  # shape: (n_tracklets, 2) where each row is [dy, dx]
    
    # Get last observations as array for batch direction computation
    last_obs = np.array([tracklet.last_observation for tracklet in tracklets])
    
    # Compute association directions (from last_observation -> detection) in batch
    # speed_direction_batch expects (dets, tracks) and returns (dy, dx) each of shape (n_tracks, n_dets)
    Y, X = speed_direction_batch(detection_boxes, last_obs)  # (n_tracklets, n_detections)
    
    # Expand velocities for broadcasting
    inertia_Y = velocities[:, 0:1]  # (n_tracklets, 1)
    inertia_X = velocities[:, 1:2]  # (n_tracklets, 1)
    
    # Compute cosine similarity (dot product of normalized vectors)
    diff_angle_cos = inertia_X * X + inertia_Y * Y  # (n_tracklets, n_detections)
    diff_angle_cos = np.clip(diff_angle_cos, -1.0, 1.0)
    
    # Apply same transformation as non-batch version
    diff_angle = np.arccos(diff_angle_cos)
    angle_diff_cost = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    # Mask out tracklets without previous observation
    valid_mask = np.array(
        [tracklet.previous_to_last_observation is not None for tracklet in tracklets],
        dtype=np.float32
    )[:, np.newaxis]  # (n_tracklets, 1)
    
    angle_diff_cost = valid_mask * angle_diff_cost

    return angle_diff_cost.astype(np.float32)


def add_track_id_detections(
    track: OCSORTTracklet,
    detection: sv.Detections,
    updated_detections: list[sv.Detections],
    minimum_consecutive_frames: int,
    frame_count: int
) -> sv.Detections:
    """
    The function prepares the updated Detections with track IDs.
    If a tracker is "mature" (>= `minimum_consecutive_frames`) or recently updated,
    it is assigned an ID to the detection that just updated it.

    Args:
        detections: The latest set of object detections.
        tracklets: List of OCSORTTracklet objects.
        updated_detections: List of detections in which we add detections with assigned track IDs.
        minimum_consecutive_frames: The number of consecutive frames required for a track to be considered "mature".
        frame_count: The current frame count in the tracking process.    
    Returns:
        sv.Detections: A copy of the detections with `tracker_id` set
            for each detection that is tracked.
    """  # noqa: E501
    new_det = deepcopy(detection)
    # Add cast to clarify type for mypy
    new_det = cast(sv.Detections, new_det)  # ADDED cast
    is_mature = (
                    track.number_of_successful_consecutive_updates
                    >= minimum_consecutive_frames
                )
    if frame_count < minimum_consecutive_frames:
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
    """
    Build IOU cost matrix between detections and predicted bounding boxes

    Args:
        detection_boxes: Detected bounding boxes in the
            form [x1, y1, x2, y2].

    Returns:
        np.ndarray: IOU cost matrix.
    """
    predicted_boxes = np.array([t.get_state_bbox() for t in trackers])
    if len(predicted_boxes) == 0 and len(trackers) > 0:
        # Handle case where get_state_bbox might return empty array
        predicted_boxes = np.zeros((len(trackers), 4), dtype=np.float32)

    if len(trackers) > 0 and len(detection_boxes) > 0:
        iou_matrix = sv.box_iou_batch(predicted_boxes, detection_boxes)
    else:
        iou_matrix = np.zeros((len(trackers), len(detection_boxes)), dtype=np.float32)

    return iou_matrix


def get_iou_matrix_between_boxes(
    last_observations: np.ndarray, detection_boxes: np.ndarray
) -> np.ndarray:
    """
    Build IOU cost matrix between detections and predicted bounding boxes

    Args:
        last_observations: Last observed bounding boxes of tracks in the
            form [x1, y1, x2, y2].
        detection_boxes: Detected bounding boxes in the
            form [x1, y1, x2, y2].

    Returns:
        np.ndarray: IOU cost matrix.
    """
    if len(last_observations) == 0 and len(last_observations) > 0:
        # Handle case where get_state_bbox might return empty array
        last_observations = np.zeros((len(last_observations), 4), dtype=np.float32)

    if len(last_observations) > 0 and len(detection_boxes) > 0:
        iou_matrix = sv.box_iou_batch(last_observations, detection_boxes)
    else:
        iou_matrix = np.zeros(
            (len(last_observations), len(detection_boxes)), dtype=np.float32
        )

    return iou_matrix
