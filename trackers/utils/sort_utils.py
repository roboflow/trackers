# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from copy import deepcopy
from typing import List, Sequence, Set, TypeVar, Union

import numpy as np
import supervision as sv

from trackers.core.bytetrack.kalman_box_tracker import ByteTrackKalmanBoxTracker
from trackers.core.sort.kalman_box_tracker import SORTKalmanBoxTracker

KalmanBoxTrackerType = TypeVar(
    "KalmanBoxTrackerType", bound=Union[SORTKalmanBoxTracker, ByteTrackKalmanBoxTracker]
)


def get_alive_trackers(
    trackers: Sequence[KalmanBoxTrackerType],
    minimum_consecutive_frames: int,
    maximum_frames_without_update: int,
) -> List[KalmanBoxTrackerType]:
    """
    Remove dead or immature lost tracklets and get alive trackers
    that are within `maximum_frames_without_update` AND (it's mature OR
    it was just updated).

    Args:
        trackers (Sequence[KalmanBoxTrackerType]): List of KalmanBoxTracker objects.
        minimum_consecutive_frames (int): Number of consecutive frames that an object
            must be tracked before it is considered a 'valid' track.
        maximum_frames_without_update (int): Maximum number of frames without update
            before a track is considered dead.

    Returns:
        List[KalmanBoxTrackerType]: List of alive trackers.
    """
    alive_trackers = []
    for tracker in trackers:
        is_mature = tracker.number_of_successful_updates >= minimum_consecutive_frames
        is_active = tracker.time_since_update == 0
        if tracker.time_since_update < maximum_frames_without_update and (
            is_mature or is_active
        ):
            alive_trackers.append(tracker)
    return alive_trackers


def get_iou_matrix(
    trackers: Sequence[KalmanBoxTrackerType], detection_boxes: np.ndarray
) -> np.ndarray:
    """
    Build IOU cost matrix between detections and predicted bounding boxes

    Args:
        detection_boxes (np.ndarray): Detected bounding boxes in the
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


def update_detections_with_track_ids(
    trackers: Sequence[KalmanBoxTrackerType],
    detections: sv.Detections,
    detection_boxes: np.ndarray,
    minimum_iou_threshold: float,
    minimum_consecutive_frames: int,
) -> sv.Detections:
    """
    The function prepares the updated Detections with track IDs.
    If a tracker is "mature" (>= `minimum_consecutive_frames`) or recently updated,
    it is assigned an ID to the detection that just updated it.

    Args:
        trackers (Sequence[SORTKalmanBoxTracker]): List of SORTKalmanBoxTracker objects.
        detections (sv.Detections): The latest set of object detections.
        detection_boxes (np.ndarray): Detected bounding boxes in the
            form [x1, y1, x2, y2].
        minimum_iou_threshold (float): IOU threshold for associating detections to
            existing tracks.
        minimum_consecutive_frames (int): Number of consecutive frames that an object
            must be tracked before it is considered a 'valid' track.

    Returns:
        sv.Detections: A copy of the detections with `tracker_id` set
            for each detection that is tracked.
    """
    # Re-run association in the same way (could also store direct mapping)
    final_tracker_ids = [-1] * len(detection_boxes)

    # Recalculate predicted_boxes based on current trackers after some may have
    # been removed
    predicted_boxes = np.array([t.get_state_bbox() for t in trackers])
    iou_matrix_final = np.zeros((len(trackers), len(detection_boxes)), dtype=np.float32)

    # Ensure predicted_boxes is properly shaped before the second iou calculation
    if len(predicted_boxes) == 0 and len(trackers) > 0:
        predicted_boxes = np.zeros((len(trackers), 4), dtype=np.float32)

    if len(trackers) > 0 and len(detection_boxes) > 0:
        iou_matrix_final = sv.box_iou_batch(predicted_boxes, detection_boxes)

    row_indices, col_indices = np.where(iou_matrix_final > minimum_iou_threshold)
    sorted_pairs = sorted(
        zip(row_indices, col_indices),
        key=lambda x: iou_matrix_final[x[0], x[1]],
        reverse=True,
    )
    used_rows: Set[int] = set()
    used_cols: Set[int] = set()
    for row, col in sorted_pairs:
        # Double check index is in range
        if row < len(trackers):
            tracker_obj = trackers[int(row)]
            # Only assign if the track is "mature" or is new but has enough hits
            if (int(row) not in used_rows) and (int(col) not in used_cols):
                if (
                    tracker_obj.number_of_successful_updates
                    >= minimum_consecutive_frames
                ):
                    # If tracker is mature but still has ID -1, assign a new ID
                    if tracker_obj.tracker_id == -1:
                        tracker_obj.tracker_id = (
                            SORTKalmanBoxTracker.get_next_tracker_id()
                        )
                    final_tracker_ids[int(col)] = tracker_obj.tracker_id
                used_rows.add(int(row))
                used_cols.add(int(col))

    # Assign tracker IDs to the returned Detections
    updated_detections = deepcopy(detections)
    updated_detections.tracker_id = np.array(final_tracker_ids)

    return updated_detections
