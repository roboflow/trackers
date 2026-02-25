# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import supervision as sv

from trackers.utils.base_tracklet import BaseTracklet

def get_alive_tracklets(
    tracklets: Sequence[BaseTracklet],
    minimum_consecutive_frames: int,
    maximum_frames_without_update: int,
) -> list[BaseTracklet]:
    """
    Remove dead or immature lost tracklets and get alive trackers
    that are within `maximum_frames_without_update` AND (it's mature OR
    it was just updated).

    Args:
        tracklets: List of KalmanBoxTracker objects.
        minimum_consecutive_frames: Number of consecutive frames that an object
            must be tracked before it is considered a 'valid' track.
        maximum_frames_without_update: Maximum number of frames without update
            before a track is considered dead.

    Returns:
        List of alive tracklets.
    """
    alive_tracklets = []
    for tracklet in tracklets:
        is_mature = tracklet.number_of_successful_consecutive_updates >= minimum_consecutive_frames
        is_active = tracklet.time_since_update == 0
        if tracklet.time_since_update < maximum_frames_without_update and (
            is_mature or is_active
        ):
            alive_tracklets.append(tracklet)
    return alive_tracklets


def get_iou_matrix(
    tracks: Sequence[BaseTracklet], detection_boxes: np.ndarray
) -> np.ndarray:
    """
    Build IOU cost matrix between detections and predicted bounding boxes

    Args:
        tracks: List of KalmanBoxTracker objects.
        detection_boxes: Detected bounding boxes in the
            form [x1, y1, x2, y2].

    Returns:
        IOU cost matrix.
    """
    predicted_boxes = np.array([t.get_state_bbox() for t in tracks])
    if len(predicted_boxes) == 0 and len(tracks) > 0:
        # Handle case where get_state_bbox might return empty array
        predicted_boxes = np.zeros((len(tracks), 4), dtype=np.float32)

    if len(tracks) > 0 and len(detection_boxes) > 0:
        iou_matrix = sv.box_iou_batch(predicted_boxes, detection_boxes)
    else:
        iou_matrix = np.zeros((len(tracks), len(detection_boxes)), dtype=np.float32)

    return iou_matrix


def update_detections_with_track_ids(
    tracks: Sequence[BaseTracklet],
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
        tracks: List of BaseTracklet objects.
        detections: The latest set of object detections.
        detection_boxes: Detected bounding boxes in the
            form [x1, y1, x2, y2].
        minimum_iou_threshold: IOU threshold for associating detections to
            existing tracks.
        minimum_consecutive_frames: Number of consecutive frames that an object
            must be tracked before it is considered a 'valid' track.

    Returns:
        A copy of the detections with `tracker_id` set
            for each detection that is tracked.
    """
    # Re-run association in the same way (could also store direct mapping)
    final_tracker_ids = [-1] * len(detection_boxes)

    # Recalculate predicted_boxes based on current tracks after some may have
    # been removed
    predicted_boxes = np.array([t.get_state_bbox() for t in tracks])
    iou_matrix_final = np.zeros((len(tracks), len(detection_boxes)), dtype=np.float32)

    # Ensure predicted_boxes is properly shaped before the second iou calculation
    if len(predicted_boxes) == 0 and len(tracks) > 0:
        predicted_boxes = np.zeros((len(tracks), 4), dtype=np.float32)

    if len(tracks) > 0 and len(detection_boxes) > 0:
        iou_matrix_final = sv.box_iou_batch(predicted_boxes, detection_boxes)

    row_indices, col_indices = np.where(iou_matrix_final > minimum_iou_threshold)
    sorted_pairs = sorted(
        zip(row_indices, col_indices),
        key=lambda x: iou_matrix_final[x[0], x[1]],
        reverse=True,
    )
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for row, col in sorted_pairs:
        # Double check index is in range
        if row < len(tracks):
            tracker_obj = tracks[int(row)]
            # Only assign if the track is "mature" or is new but has enough hits
            if (int(row) not in used_rows) and (int(col) not in used_cols):
                if (
                    tracker_obj.number_of_successful_consecutive_updates
                    >= minimum_consecutive_frames
                ):
                    # If tracker is mature but still has ID -1, assign a new ID
                    if tracker_obj.tracker_id == -1:
                        tracker_obj.tracker_id = (
                            type(tracker_obj).get_next_tracker_id()
                        )
                    final_tracker_ids[int(col)] = tracker_obj.tracker_id
                used_rows.add(int(row))
                used_cols.add(int(col))

    # Assign tracker IDs to the returned Detections
    updated_detections = deepcopy(detections)
    updated_detections.tracker_id = np.array(final_tracker_ids)

    return updated_detections
