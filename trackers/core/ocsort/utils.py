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
    # vectorized version from oc sort repo but adapted to our tracklet class, still needs testing and is not currently used# noqa: E501
    tracklets: list[OCSORTTracklet],
    detection_boxes: np.ndarray,
) -> np.ndarray:
    velocities = np.array(
        [
            speed_direction(
                tracklet.previous_to_last_observation, tracklet.last_observation
            )
            if tracklet.previous_to_last_observation is not None
            else (0, 0)
            for tracklet in tracklets
        ]
    )
    Y, X = speed_direction_batch(detection_boxes, tracklets)  # these will break
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(len(tracklets))
    # valid_mask[np.where(previous_obs[:,4]<0)] = 0
    previous_to_last_observation_is_none = np.array(
        [tracklet.previous_to_last_observation is None for tracklet in tracklets]
    )
    valid_mask[previous_to_last_observation_is_none] = 0
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = valid_mask * diff_angle
    angle_diff_cost = angle_diff_cost.T  # check if this transpose is needed
    return angle_diff_cost


def add_track_id_detections(
    track: OCSORTTracklet,
    detection: sv.Detections,
    updated_detections: list[sv.Detections],
) -> sv.Detections:
    """
    The function prepares the updated Detections with track IDs.
    If a tracker is "mature" (>= `minimum_consecutive_frames`) or recently updated,
    it is assigned an ID to the detection that just updated it.

    Args:
        detections: The latest set of object detections.
        tracklets: List of OCSORTTracklet objects.
        updated_detections: List of detections in which we add detections with assigned track IDs.
    Returns:
        sv.Detections: A copy of the detections with `tracker_id` set
            for each detection that is tracked.
    """  # noqa: E501
    new_det = deepcopy(detection)
    # Add cast to clarify type for mypy
    new_det = cast(sv.Detections, new_det)  # ADDED cast
    new_det.tracker_id = np.array([track.tracker_id])
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
