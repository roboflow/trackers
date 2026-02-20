# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from copy import deepcopy
from typing import cast

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.botsort.cmc import CMC, CMCConfig
from trackers.core.botsort.kalman_box_tracker import BoTSORTKalmanBoxTracker
from trackers.utils.sort_utils import (
    get_alive_trackers,
    get_iou_matrix,
)


class BoTSORTTracker(BaseTracker):
    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.1,
        high_conf_det_threshold: float = 0.6,
        enable_cmc: bool = True,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold
        self.high_conf_det_threshold = high_conf_det_threshold
        self.tracks: list[BoTSORTKalmanBoxTracker] = []

        self.enable_cmc = enable_cmc
        self.cmc = CMC(CMCConfig()) if enable_cmc else None

    def _update_detections(
        self,
        tracks: list[BoTSORTKalmanBoxTracker],
        detections: sv.Detections,
        updated_detections: list[sv.Detections],
        matched_indices: list[tuple[int, int]],
    ) -> list[sv.Detections]:
        # Update matched tracks with assigned detections.
        det_bboxes = detections.xyxy
        for row, col in matched_indices:
            t = tracks[row]
            t.update(det_bboxes[col])
            # If tracker is mature but still has ID -1, assign a new ID
            if (
                t.number_of_successful_updates >= self.minimum_consecutive_frames
                and t.tracker_id == -1
            ):  # Check maturity before assigning ID
                t.tracker_id = BoTSORTKalmanBoxTracker.get_next_tracker_id()

            new_det = deepcopy(detections[col : col + 1])
            # Add cast to clarify type for mypy
            new_det = cast(sv.Detections, new_det)  # ADDED cast
            new_det.tracker_id = np.array([t.tracker_id])
            updated_detections.append(new_det)
        return updated_detections

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
    ) -> sv.Detections:
        if len(self.tracks) == 0 and len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections
        updated_detections: list[
            sv.Detections
        ] = []  # List for returning the updated detections with its new assigned track id # noqa: E501

        # Predict new locations for existing tracks
        for tracker in self.tracks:
            tracker.predict()
        # Assign a default tracker_id with the correct shape
        detections.tracker_id = -np.ones(len(detections))
        # Split into high confidence boxes and lower based on self.high_conf_det_threshold # noqa: E501
        high_prob_detections, low_prob_detections = (
            self._get_high_and_low_probability_detections(detections)
        )

        # CMC (ORB) apply to all predicted tracks before association
        if self.enable_cmc and self.cmc is not None and frame is not None:
            mask_boxes = (
                high_prob_detections.xyxy if len(high_prob_detections) > 0 else None
            )
            H = self.cmc.estimate(frame, mask_boxes)
            self.cmc.apply_to_tracks(self.tracks, H)

        # Step 1: first association, with high confidence boxes
        matched_indices, unmatched_tracks, unmatched_high_prob_detections = (
            self._similarity_step(
                high_prob_detections,
                self.tracks,
            )
        )

        # Update matched tracks with high-confidence detections
        self._update_detections(
            self.tracks,
            high_prob_detections,
            updated_detections,
            matched_indices,
        )

        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]

        # Step 2: associate Low Probability detections with remaining tracks
        matched_indices, unmatched_tracks, unmatched_detections = self._similarity_step(
            low_prob_detections, remaining_tracks
        )

        # Update matched tracks with low-confidence detections
        self._update_detections(
            remaining_tracks,
            low_prob_detections,
            updated_detections,
            matched_indices,
        )

        # Add unmatched low prob predictions to updated predictions
        for det_index in unmatched_detections:
            new_det = deepcopy(low_prob_detections[det_index : det_index + 1])

            new_det.tracker_id = np.array([-1])
            updated_detections.append(new_det)

        self._spawn_new_trackers(
            high_prob_detections,
            high_prob_detections.xyxy,
            unmatched_high_prob_detections,
            updated_detections,
        )

        # Kill lost tracks
        self.tracks = get_alive_trackers(
            trackers=self.tracks,
            maximum_frames_without_update=self.maximum_frames_without_update,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )
        final_updated_detections: sv.Detections = sv.Detections.merge(
            updated_detections
        )
        if len(final_updated_detections) == 0:
            final_updated_detections.tracker_id = np.array([], dtype=int)
        return final_updated_detections

    def _get_high_and_low_probability_detections(
        self, detections: sv.Detections
    ) -> tuple[sv.Detections, sv.Detections]:
        """
        Splits the input detections into high-confidence and low-confidence sets
        based on the `self.high_conf_det_threshold`.

        Args:
            detections: The input detections with confidence scores.

        Returns:
            A tuple containing two `sv.Detections objects`: the first for
                high-confidence detections `(confidence >= threshold)` and the second
                for low-confidence detections `(confidence < threshold)`.
        """
        # Check if confidence scores exist before comparing
        if detections.confidence is not None:
            # Perform element-wise comparison if confidence is a NumPy array
            condition = detections.confidence >= self.high_conf_det_threshold
        else:
            # If no confidence scores, no detections meet the threshold
            # Create a boolean array of False with the same length as detections
            condition = np.zeros(len(detections), dtype=bool)

        high_confidence = detections[condition]
        low_confidence = detections[np.logical_not(condition)]
        return high_confidence, low_confidence

    def _get_associated_indices(
        self,
        similarity_matrix: np.ndarray,
        min_similarity_thresh: float,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """
        Associate detections to tracks based on Similarity (IoU) using the
        Jonker-Volgenant algorithm approach with no initialization instead of the
        Hungarian algorithm as mentioned in the SORT paper, but it solves the
        assignment problem in an optimal way.

        Args:
            similarity_matrix: Similarity matrix between tracks (rows) and detections (columns).
            min_similarity_thresh: Minimum similarity threshold for a valid match.

        Returns:
            Matched indices (list of (tracker_idx, detection_idx)), indices of
                unmatched tracks, indices of unmatched detections.
        """  # noqa: E501
        matched_indices = []
        n_tracks, n_detections = similarity_matrix.shape
        unmatched_tracks = set(range(n_tracks))
        unmatched_detections = set(range(n_detections))

        if n_tracks > 0 and n_detections > 0:
            row_indices, col_indices = linear_sum_assignment(
                similarity_matrix, maximize=True
            )
            for row, col in zip(row_indices, col_indices):
                if similarity_matrix[row, col] >= min_similarity_thresh:
                    matched_indices.append((row, col))
                    unmatched_tracks.remove(row)
                    unmatched_detections.remove(col)

        return matched_indices, unmatched_tracks, unmatched_detections

    def _spawn_new_trackers(
        self,
        detections: sv.Detections,
        detection_boxes: np.ndarray,
        unmatched_detections: set[int],
        updated_detections: list[sv.Detections],
    ):
        """
        Create new trackers for unmatched detections and
            append detections to updated_detections detections.

        Args:
            detections: Current detections.
            detection_boxes: Bounding boxes for detections.
            unmatched_detections: Indices of unmatched detections.
            updated_detections: List with all the detections

        """
        for detection_idx in unmatched_detections:
            # Check for detections.confidence existence and index bounds
            if detections.confidence is not None and detection_idx < len(
                detections.confidence
            ):
                # Assign to a temporary variable with explicit type hint
                confidence_score: float = float(detections.confidence[detection_idx])

                # Use the temporary variable in the comparison
                if confidence_score >= self.track_activation_threshold:
                    # Original logic for high confidence detection

                    new_tracker = BoTSORTKalmanBoxTracker(
                        bbox=detection_boxes[detection_idx]
                    )
                    self.tracks.append(new_tracker)

                    new_det = deepcopy(detections[detection_idx : detection_idx + 1])
                    new_det = cast(sv.Detections, new_det)  # Cast added previously
                    new_det.tracker_id = np.array([-1])
                    updated_detections.append(new_det)
            else:
                pass  # Do nothing, the detection remains unmatched

    def _similarity_step(
        self,
        detections: sv.Detections,
        tracks: list[BoTSORTKalmanBoxTracker],
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """Measures similarity based on IoU between tracks and detections and returns the matches
            and unmatched tracks/detections. Is used for step 1 and 2 of the BYTE algorithm.

        Args:
            detections: The set of object detections.
            tracks: The list of tracks that will be matched to the detections.

        Returns:
            A tuple containing:
                - matched_indices: A list of (tracker_idx, detection_idx) pairs.
                - unmatched_tracks_indices: A set of indices for tracks that
                  were not matched.
                - unmatched_detections_indices: A set of indices for detections
                  that were not matched.
        """  # noqa: E501
        # Build IoU cost matrix between detections and predicted bounding boxes
        similarity_matrix = get_iou_matrix(tracks, detections.xyxy)
        thresh = self.minimum_iou_threshold

        # Associate detections to tracks based on the higher value of the
        # similarity matrix, using the Jonker-Volgenant algorithm (linear_sum_assignment). # noqa: E501
        matched_indices, unmatched_tracks, unmatched_detections = (
            self._get_associated_indices(similarity_matrix, thresh)
        )
        return matched_indices, unmatched_tracks, unmatched_detections

    def reset(self) -> None:
        """Reset tracker state by clearing all tracks and resetting ID counter.
        Call this method when switching to a new video or scene.
        """
        self.tracks = []
        BoTSORTKalmanBoxTracker.count_id = 0
        if self.cmc is not None:
            self.cmc.reset()
