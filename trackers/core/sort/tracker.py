# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.sort.kalman_box_tracker import SORTKalmanBoxTracker
from trackers.utils.sort_utils import (
    get_alive_trackers,
    get_iou_matrix,
    update_detections_with_track_ids,
)


class SORTTracker(BaseTracker):
    """Track objects using SORT algorithm with Kalman filter and IoU matching.
    Provides simple and fast online tracking using only bounding box geometry
    without appearance features.

    Args:
        lost_track_buffer: `int` specifying number of frames to buffer when a
            track is lost. Increasing this value enhances occlusion handling but
            may increase ID switching for similar objects.
        frame_rate: `float` specifying video frame rate in frames per second.
            Used to scale the lost track buffer for consistent tracking across
            different frame rates.
        track_activation_threshold: `float` specifying minimum detection
            confidence to create new tracks. Higher values reduce false
            positives but may miss low-confidence objects.
        minimum_consecutive_frames: `int` specifying number of consecutive
            frames before a track is considered valid. Before reaching this
            threshold, tracks are assigned `tracker_id` of `-1`.
        minimum_iou_threshold: `float` specifying IoU threshold for associating
            detections to existing tracks. Higher values require more overlap.
    """

    tracker_id = "sort"

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold

        # Active trackers
        self.trackers: list[SORTKalmanBoxTracker] = []

    def _get_associated_indices(
        self, iou_matrix: np.ndarray, detection_boxes: np.ndarray
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """
        Associate detections to trackers based on IOU

        Args:
            iou_matrix: IOU cost matrix.
            detection_boxes: Detected bounding boxes in the form [x1, y1, x2, y2].

        Returns:
            Matched indices, unmatched trackers, unmatched detections.
        """
        matched_indices = []
        unmatched_trackers = set(range(len(self.trackers)))
        unmatched_detections = set(range(len(detection_boxes)))

        if len(self.trackers) > 0 and len(detection_boxes) > 0:
            # Find optimal assignment using scipy.optimize.linear_sum_assignment.
            # Note that it uses a a modified Jonker-Volgenant algorithm with no
            # initialization instead of the Hungarian algorithm as mentioned in the
            # SORT paper.
            row_indices, col_indices = linear_sum_assignment(iou_matrix, maximize=True)
            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] >= self.minimum_iou_threshold:
                    matched_indices.append((row, col))
                    unmatched_trackers.remove(row)
                    unmatched_detections.remove(col)

        return matched_indices, unmatched_trackers, unmatched_detections

    def _spawn_new_trackers(
        self,
        detections: sv.Detections,
        detection_boxes: np.ndarray,
        unmatched_detections: set[int],
    ) -> None:
        """
        Create new trackers only for unmatched detections with confidence
        above threshold.

        Args:
            detections: The latest set of object detections.
            detection_boxes: Detected bounding boxes in the form [x1, y1, x2, y2].
        """
        for detection_idx in unmatched_detections:
            if (
                detections.confidence is None
                or detection_idx >= len(detections.confidence)
                or detections.confidence[detection_idx]
                >= self.track_activation_threshold
            ):
                new_tracker = SORTKalmanBoxTracker(detection_boxes[detection_idx])
                self.trackers.append(new_tracker)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker state with new detections and return tracked objects.
        Performs Kalman filter prediction, IoU-based association, and initializes
        new tracks for unmatched high-confidence detections.

        Args:
            detections: `sv.Detections` containing bounding boxes with shape
                `(N, 4)` in `(x_min, y_min, x_max, y_max)` format and optional
                confidence scores.

        Returns:
            `sv.Detections` with `tracker_id` assigned for each detection.
                Unmatched or immature tracks have `tracker_id` of `-1`.
        """

        if len(self.trackers) == 0 and len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections

        # Convert detections to a (N x 4) array (x1, y1, x2, y2)
        detection_boxes = (
            detections.xyxy if len(detections) > 0 else np.array([]).reshape(0, 4)
        )

        # Predict new locations for existing trackers
        for tracker in self.trackers:
            tracker.predict()

        # Build IOU cost matrix between detections and predicted bounding boxes
        iou_matrix = get_iou_matrix(self.trackers, detection_boxes)

        # Associate detections to trackers based on IOU
        matched_indices, _, unmatched_detections = self._get_associated_indices(
            iou_matrix, detection_boxes
        )

        # Update matched trackers with assigned detections
        for row, col in matched_indices:
            self.trackers[row].update(detection_boxes[col])

        self._spawn_new_trackers(detections, detection_boxes, unmatched_detections)

        # Remove dead trackers
        self.trackers = get_alive_trackers(
            self.trackers,
            self.minimum_consecutive_frames,
            self.maximum_frames_without_update,
        )

        updated_detections = update_detections_with_track_ids(
            self.trackers,
            detections,
            detection_boxes,
            self.minimum_iou_threshold,
            self.minimum_consecutive_frames,
        )

        return updated_detections

    def reset(self) -> None:
        """Reset tracker state by clearing all tracks and resetting ID counter.
        Call this method when switching to a new video or scene.
        """
        self.trackers = []
        SORTKalmanBoxTracker.count_id = 0
