# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from copy import deepcopy

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.ocsort.tracklet import OCSORTTracklet
from trackers.core.ocsort.utils import (
    _build_direction_consistency_matrix_batch,
    _get_iou_matrix,
)
from trackers.utils.state_representations import XCYCSRStateEstimator


class OCSORTTracker(BaseTracker):
    """OC-SORT enhances traditional SORT by shifting to an observation-centric paradigm,
    using detections to correct Kalman filter errors accumulated during occlusions. It
    introduces Observation-Centric Re-Update to generate virtual trajectories for
    parameter refinement upon track reactivation. Association incorporates
    Observation-Centric Momentum, blending IoU with direction consistency from
    historical observations. Short-term recoveries are aided by heuristics linking
    unmatched tracks to prior detections. This rethinking prioritizes real measurements
    over estimations, making OC-SORT particularly adept at handling real-world tracking
    challenges.

    OC-SORT's primary strength is its robustness to non-linear motions and occlusions,
    outperforming baselines on datasets with erratic movements like DanceTrack. It
    maintains extreme efficiency, processing over 700 frames per second on CPUs for
    scalable deployments. The tracker excels in crowded scenes, reducing identity
    switches through momentum-based associations. However, lacking appearance features,
    it can confuse similar objects in overlapping paths. Its linear motion core still
    imposes limits in extreme velocity variations, requiring careful parameter
    selection.

    Args:
        lost_track_buffer: `int` specifying number of frames to buffer when a
            track is lost. Increasing this value enhances occlusion handling but
            may increase ID switching for similar objects.
        frame_rate: `float` specifying video frame rate in frames per second.
            Used to scale the lost track buffer for consistent tracking across
            different frame rates.
        minimum_consecutive_frames: `int` specifying number of consecutive
            frames before a track is considered valid. Before reaching this
            threshold, tracks are assigned `tracker_id` of `-1`.
        minimum_iou_threshold: `float` specifying IoU threshold for associating
            detections to existing tracks. Higher values require more overlap.
        direction_consistency_weight: `float` specifying weight for direction
            consistency in the association cost. Higher values prioritize angle
            alignment between motion and association direction.
        high_conf_det_threshold: `float` specifying threshold for high confidence
            detections. Lower confidence detections are excluded from association.
        delta_t: `int` specifying number of past frames to use for velocity
            estimation. Higher values provide more stable direction estimates
            during occlusion.
    """

    tracker_id = "ocsort"

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
        direction_consistency_weight: float = 0.2,
        high_conf_det_threshold: float = 0.6,
        delta_t: int = 3,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.direction_consistency_weight = direction_consistency_weight
        self.high_conf_det_threshold = high_conf_det_threshold
        self.delta_t = delta_t

        self.tracks: list[OCSORTTracklet] = []
        self.frame_count = 0
        self.state_estimator_class = XCYCSRStateEstimator

    def _get_associated_indices(
        self,
        iou_matrix: np.ndarray,
        direction_consistency_matrix: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Associate detections to tracks based on IOU.

        Args:
            iou_matrix: IOU cost matrix.
            direction_consistency_matrix: Direction of the tracklet consistency cost matrix.

        Returns:
            matched_indices: List of (track_index, detection_index) tuples for
                successful associations that meet the IOU threshold.
            unmatched_tracks: list of track indices that were not matched
                to any detection.
            unmatched_detections: list of detection indices that were not
                matched to any track.
        """  # noqa: E501
        matched_indices = []
        n_tracks, n_detections = iou_matrix.shape
        unmatched_tracks = set(range(n_tracks))
        unmatched_detections = set(range(n_detections))
        if n_tracks > 0 and n_detections > 0:
            # Find optimal assignment using scipy.optimize.linear_sum_assignment.
            cost_matrix = (
                iou_matrix
                + self.direction_consistency_weight * direction_consistency_matrix
            )
            row_indices, col_indices = linear_sum_assignment(cost_matrix, maximize=True)
            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] >= self.minimum_iou_threshold:
                    matched_indices.append((row, col))
                    unmatched_tracks.remove(row)
                    unmatched_detections.remove(col)

        return (
            matched_indices,
            list(unmatched_tracks),
            list(unmatched_detections),
        )

    def _spawn_new_tracklets(
        self,
        detections: sv.Detections,
    ) -> None:
        """
        Create new tracklets only for passed detections. It is used for activating
        new tracklets from unmatched detections after association.

        Args:
            detections: The detections that will start new tracklets.
        """
        for xyxy in detections.xyxy:
            new_tracker = OCSORTTracklet(
                xyxy,
                delta_t=self.delta_t,
                state_estimator_class=self.state_estimator_class,
            )
            self.tracks.append(new_tracker)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker state with new detections and return tracked objects.
        Performs Kalman filter prediction, two-stage association using direction
        consistency and last-observation recovery, and initializes new tracks
        for unmatched high-confidence detections.

        Args:
            detections: `sv.Detections` containing bounding boxes with shape
                `(N, 4)` in `(x_min, y_min, x_max, y_max)` format and optional
                confidence scores.

        Returns:
            `sv.Detections` with `tracker_id` assigned for each detection.
                Unmatched or immature tracks have `tracker_id` of `-1`.
        """

        if len(self.tracks) == 0 and len(detections) == 0:
            result = deepcopy(detections)
            result.tracker_id = np.array([], dtype=int)
            return result

        detections = detections[detections.confidence >= self.high_conf_det_threshold]

        updated_detections: list[
            sv.Detections
        ] = []  # List for returning the updated detections
        # Convert detections to a (N x 4) array (x1, y1, x2, y2)
        detection_boxes = (
            detections.xyxy if len(detections) > 0 else np.array([]).reshape(0, 4)
        )

        # Predict new locations for existing tracks KF
        for tracker in self.tracks:
            tracker.predict()

        # Build IOU cost matrix between detections and predicted bounding boxes
        predicted_boxes = np.array([t.get_state_bbox() for t in self.tracks])

        iou_matrix = _get_iou_matrix(predicted_boxes, detection_boxes)

        # Compute direction consistency matrix
        direction_consistency_matrix = self._compute_direction_consistency_matrix(
            detection_boxes, detections
        )

        # 1st Association of detections to tracks (OCM)
        matched_indices, unmatched_tracks, unmatched_detections = (
            self._get_associated_indices(iou_matrix, direction_consistency_matrix)
        )

        # Update matched trackers with assigned detections
        for row, col in matched_indices:
            self.tracks[row].update(detection_boxes[col])
            self.tracks[row].add_track_id_to_detections(
                detections[col : col + 1],
                updated_detections,
                self.minimum_consecutive_frames,
                self.frame_count,
            )

        # Run 2nd Chance Association (OCR)
        # between the last observation of unmatched tracks to the unmatched observations #noqa: E501
        if len(unmatched_detections) > 0 and len(unmatched_tracks) > 0:
            last_observation_of_tracks = np.array(
                [self.tracks[t_id].last_observation for t_id in unmatched_tracks]
            )

            ocr_iou_matrix = sv.box_iou_batch(
                last_observation_of_tracks, detection_boxes[unmatched_detections]
            )

            ocr_matched_indices, _ocr_unmatched_tracks, ocr_unmatched_detections = (
                self._get_associated_indices(
                    ocr_iou_matrix,
                    np.zeros_like(ocr_iou_matrix),
                )
            )

            for ocr_row, ocr_col in ocr_matched_indices:
                track_idx = unmatched_tracks[ocr_row]
                det_idx = unmatched_detections[ocr_col]
                self.tracks[track_idx].update(detection_boxes[det_idx])

                self.tracks[track_idx].add_track_id_to_detections(
                    detections[det_idx : det_idx + 1],
                    updated_detections,
                    self.minimum_consecutive_frames,
                    self.frame_count,
                )

            # Update OCR-unmatched tracks with None before filtering (marks as lost for re-update) #noqa: E501
            for m in _ocr_unmatched_tracks:
                self.tracks[unmatched_tracks[m]].update(None)

            self.tracks = self._prune_expired_tracklets()
            remaining_detections = detections[unmatched_detections][
                ocr_unmatched_detections
            ]

            self._spawn_new_tracklets(remaining_detections)
            remaining_detections.tracker_id = np.array(
                [-1] * len(remaining_detections), dtype=int
            )
            updated_detections.append(remaining_detections)
        else:
            for track_idx in unmatched_tracks:
                self.tracks[track_idx].update(None)
            self.tracks = self._prune_expired_tracklets()
            remaining_detections = detections[unmatched_detections]

            self._spawn_new_tracklets(remaining_detections)
            remaining_detections.tracker_id = np.array(
                [-1] * len(remaining_detections), dtype=int
            )
            updated_detections.append(remaining_detections)
        final_updated_detections = sv.Detections.merge(updated_detections)
        if len(final_updated_detections) == 0:
            final_updated_detections.tracker_id = np.array([], dtype=int)
        self.frame_count += 1
        return final_updated_detections

    def reset(self) -> None:
        """Reset tracker state by clearing all tracks and resetting ID counter.
        Call this method when switching to a new video or scene.
        """
        self.tracks = []
        self.frame_count = 0
        OCSORTTracklet.count_id = 0

    def _prune_expired_tracklets(self) -> list[OCSORTTracklet]:
        """Remove tracklets that have been lost for too long.

        Returns:
            List of tracklets that are still active.
        """
        return [
            tracklet
            for tracklet in self.tracks
            if tracklet.time_since_update <= self.maximum_frames_without_update
        ]

    def _compute_direction_consistency_matrix(
        self, detection_boxes: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        """Extract arrays and compute the direction consistency matrix for association,
        including confidence scaling."""
        tracklet_velocities = np.array(
            [
                t.velocity if t.velocity is not None else np.array([0.0, 0.0])
                for t in self.tracks
            ]
        )
        reference_boxes = np.array(
            [
                t.get_k_previous_obs()
                if t.get_k_previous_obs() is not None
                else t.last_observation
                for t in self.tracks
            ]
        )
        velocity_mask = np.array(
            [t.velocity is not None for t in self.tracks],
            dtype=np.float32,
        )[:, np.newaxis]
        matrix = _build_direction_consistency_matrix_batch(
            tracklet_velocities=tracklet_velocities,
            reference_boxes=reference_boxes,
            detection_boxes=detection_boxes,
            velocity_mask=velocity_mask,
        )
        matrix *= detections.confidence[np.newaxis, :]
        return matrix
