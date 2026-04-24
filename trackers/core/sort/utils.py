# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import supervision as sv

from trackers.core.sort.tracklet import SORTTracklet
from trackers.utils.base_tracklet import BaseTracklet

T_SORTTracklet = TypeVar("T_SORTTracklet", bound="SORTTracklet")


def _get_alive_tracklets(
    tracklets: Sequence[T_SORTTracklet],
    minimum_consecutive_frames: int,
    maximum_frames_without_update: int,
) -> list[T_SORTTracklet]:
    """
    Remove dead or immature lost tracklets and get alive trackers
    that are within `maximum_frames_without_update` AND (it's mature OR
    it was just updated).

    Note:
        SORT uses total `number_of_successful_updates` (cumulative) for maturity,
        unlike ByteTrack which uses `number_of_successful_consecutive_updates`.
        This means a briefly-lost-and-recovered track retains its maturity in SORT
        but resets to immature in ByteTrack.

    Args:
        tracklets: List of SORTTracklet objects.
        minimum_consecutive_frames: Number of consecutive frames that an object
            must be tracked before it is considered a 'valid' track.
        maximum_frames_without_update: Maximum number of frames without update
            before a track is considered dead.

    Returns:
        List of alive tracklets.
    """
    alive_tracklets = []
    for tracklet in tracklets:
        is_mature = tracklet.number_of_successful_updates >= minimum_consecutive_frames
        is_active = tracklet.time_since_update == 0
        if tracklet.time_since_update < maximum_frames_without_update and (
            is_mature or is_active
        ):
            alive_tracklets.append(tracklet)
    return alive_tracklets


def _get_iou_matrix(
    tracks: Sequence[BaseTracklet], detection_boxes: np.ndarray
) -> np.ndarray:
    """
    Build IOU cost matrix between detections and predicted bounding boxes

    Args:
        tracks: List of BaseTracklet objects.
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
