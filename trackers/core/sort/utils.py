# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections.abc import Sequence
from copy import deepcopy
from typing import TypeVar

import numpy as np
import supervision as sv

from trackers.core.bytetrack.kalman import ByteTrackKalmanBoxTracker
from trackers.core.sort.kalman import SORTKalmanBoxTracker

KalmanBoxTrackerType = TypeVar(
    "KalmanBoxTrackerType", bound=SORTKalmanBoxTracker | ByteTrackKalmanBoxTracker
)


def get_alive_trackers(
    trackers: Sequence[KalmanBoxTrackerType],
    minimum_consecutive_frames: int,
    maximum_frames_without_update: int,
) -> list[KalmanBoxTrackerType]:
    """
    Remove dead or immature lost tracklets and get alive trackers
    that are within `maximum_frames_without_update` AND (it's mature OR
    it was just updated).

    Args:
        trackers: List of KalmanBoxTracker objects.
        minimum_consecutive_frames: Number of consecutive frames that an object
            must be tracked before it is considered a 'valid' track.
        maximum_frames_without_update: Maximum number of frames without update
            before a track is considered dead.

    Returns:
        List of alive trackers.
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


def _compute_diou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute Distance IoU (DIoU) between two sets of boxes.

    DIoU = IoU - d^2 / c^2 where d is the Euclidean distance between box
    centers and c is the diagonal length of the smallest enclosing box.
    Ranges from -1 to 1; penalizes center displacement directly.

    Reference: Zheng et al., "Distance-IoU Loss", AAAI 2020.

    Args:
        boxes_a: Array of shape ``(M, 4)`` in ``[x1, y1, x2, y2]`` format.
        boxes_b: Array of shape ``(N, 4)`` in ``[x1, y1, x2, y2]`` format.

    Returns:
        DIoU matrix of shape ``(M, N)``.
    """
    # Intersection coordinates
    x1_inter = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1_inter = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2_inter = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2_inter = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    inter_area = np.maximum(x2_inter - x1_inter, 0) * np.maximum(y2_inter - y1_inter, 0)

    # Areas of individual boxes
    area_a = (
        (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    ).reshape(-1, 1)
    area_b = (
        (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    ).reshape(1, -1)

    union_area = area_a + area_b - inter_area
    iou = np.where(union_area > 0, inter_area / union_area, 0.0)

    # Center distance squared
    cx_a = ((boxes_a[:, 0] + boxes_a[:, 2]) / 2).reshape(-1, 1)
    cy_a = ((boxes_a[:, 1] + boxes_a[:, 3]) / 2).reshape(-1, 1)
    cx_b = ((boxes_b[:, 0] + boxes_b[:, 2]) / 2).reshape(1, -1)
    cy_b = ((boxes_b[:, 1] + boxes_b[:, 3]) / 2).reshape(1, -1)
    d_sq = (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2

    # Enclosing box diagonal squared
    x1_c = np.minimum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1_c = np.minimum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2_c = np.maximum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2_c = np.maximum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)
    c_sq = (x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2

    diou = iou - np.where(c_sq > 0, d_sq / c_sq, 0.0)

    return diou.astype(np.float32)


def get_iou_matrix(
    trackers: Sequence[KalmanBoxTrackerType], detection_boxes: np.ndarray
) -> np.ndarray:
    """Build DIoU similarity matrix between tracked and detected boxes.

    Uses Distance IoU (DIoU) instead of standard IoU to penalize center
    displacement and recover association signal for near-miss pairs.

    Args:
        trackers: List of KalmanBoxTracker objects.
        detection_boxes: Detected bounding boxes in the
            form [x1, y1, x2, y2].

    Returns:
        DIoU similarity matrix.
    """
    predicted_boxes = np.array([t.get_state_bbox() for t in trackers])
    if len(predicted_boxes) == 0 and len(trackers) > 0:
        # Handle case where get_state_bbox might return empty array
        predicted_boxes = np.zeros((len(trackers), 4), dtype=np.float32)

    if len(trackers) > 0 and len(detection_boxes) > 0:
        iou_matrix = _compute_diou_matrix(predicted_boxes, detection_boxes)
    else:
        iou_matrix = np.zeros((len(trackers), len(detection_boxes)), dtype=np.float32)

    return iou_matrix


def interpolate_mot_gaps(
    lines: list[str],
    max_gap: int = 20,
) -> list[str]:
    """Fill short gaps in MOT-format output via linear bbox interpolation.

    For each track that disappears for up to ``max_gap`` consecutive frames
    and then reappears, linearly interpolate the bounding box coordinates
    between the last observation before the gap and the first observation after.

    Args:
        lines: MOT-format lines, each ``"frame,id,x,y,w,h,conf,-1,-1,-1"``.
        max_gap: Maximum gap length (in frames) to interpolate. Gaps longer
            than this are left unfilled. ``0`` disables interpolation.

    Returns:
        Augmented list of MOT-format lines including interpolated entries.

    Examples:
        >>> obs = ["1,1,10,20,30,40,0.9,-1,-1,-1", "3,1,16,26,30,40,0.8,-1,-1,-1"]
        >>> result = interpolate_mot_gaps(obs, max_gap=2)
        >>> any("2,1," in r for r in result)
        True
    """
    if not lines or max_gap <= 0:
        return lines

    tracks: dict[int, list[tuple[int, float, float, float, float, float]]] = {}
    for line in lines:
        parts = line.split(",")
        if len(parts) < 7:
            continue
        frame = int(parts[0])
        tid = int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        conf = float(parts[6])
        tracks.setdefault(tid, []).append((frame, x, y, w, h, conf))

    for tid in tracks:
        tracks[tid].sort(key=lambda t: t[0])

    interp_lines: list[str] = []
    for tid, obs in tracks.items():
        for i in range(len(obs) - 1):
            f1, x1, y1, w1, h1, c1 = obs[i]
            f2, x2, y2, w2, h2, c2 = obs[i + 1]
            gap = f2 - f1
            if gap <= 1 or gap > max_gap + 1:
                continue
            for j in range(1, gap):
                alpha = j / gap
                fx = x1 + alpha * (x2 - x1)
                fy = y1 + alpha * (y2 - y1)
                fw = w1 + alpha * (w2 - w1)
                fh = h1 + alpha * (h2 - h1)
                fc = min(c1, c2)
                interp_lines.append(
                    f"{f1 + j},{tid},{fx:.2f},{fy:.2f},"
                    f"{fw:.2f},{fh:.2f},{fc:.4f},-1,-1,-1"
                )

    return lines + interp_lines


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
        trackers: List of SORTKalmanBoxTracker objects.
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
    used_rows: set[int] = set()
    used_cols: set[int] = set()
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
