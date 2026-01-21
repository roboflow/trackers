from copy import deepcopy
from typing import cast

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.bytetrack.kalman_box_tracker import ByteTrackKalmanBoxTracker
from trackers.utils.sort_utils import (
    get_alive_trackers,
    get_iou_matrix,
)


class ByteTrackTracker(BaseTracker):
    """Implements ByteTrack.

    ByteTrack is a simple, effective, and generic multi-object tracking method
    that improves upon tracking-by-detection by associating *every* detection box
    instead of discarding low-score ones. This makes it more robust to occlusions. 
    It uses a two-stage association process and builds on established techniques 
    like the Kalman Filter for motion prediction and the Hungarian algorithm for
    data association.

    Args:
        lost_track_buffer (int): Number of frames to buffer when a track is lost.
            Increasing lost_track_buffer enhances occlusion handling, significantly
            improving tracking through occlusions, but may increase the possibility
            of ID switching for objects that disappear.
        frame_rate (float): Frame rate of the video (frames per second).
            Used to calculate the maximum time a track can be lost.
        track_activation_threshold (float): Detection confidence threshold
            for track activation. Only detections with confidence above this
            threshold will create new tracks. Increasing this threshold may
            reduce false positives but may miss real objects with low confidence.
        minimum_consecutive_frames (int): Number of consecutive frames that an object
            must be tracked before it is considered a 'valid'/'active/ track. Increasing
            `minimum_consecutive_frames` prevents the creation of accidental tracks
            from false detection or double detection, but risks missing shorter
            tracks. Before the tracker is considered valid, it will be assigned
            `-1` as its `tracker_id`.
        minimum_iou_threshold (float): IoU threshold for associating detections to existing tracks.
            Prevents the association of lower IoU than the threshold between boxes and tracks.
            A higher value will only associate boxes that have more overlapping area.
        high_conf_boxes_threshold (float): threshold for assigning predicted boxes to high probability class.
            A higher value will classify only higher confidence/probability boxes as 'high probability'
            per the ByteTrack algorithm, which are used in the first similarity step of
            the algorithm.
    """  # noqa: E501

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.1,
        high_conf_boxes_threshold: float = 0.6,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold
        self.high_conf_boxes_threshold = high_conf_boxes_threshold
        self.tracks: list[ByteTrackKalmanBoxTracker] = []

    def _update_detections(
        self,
        tracks: list[ByteTrackKalmanBoxTracker],
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
                t.tracker_id = ByteTrackKalmanBoxTracker.get_next_tracker_id()

            new_det = deepcopy(detections[col : col + 1])
            # Add cast to clarify type for mypy
            new_det = cast(sv.Detections, new_det)  # ADDED cast
            new_det.tracker_id = np.array([t.tracker_id])
            updated_detections.append(new_det)
        return updated_detections

    def update(
        self,
        detections: sv.Detections,
    ) -> sv.Detections:
        """Updates the tracker state with new detections.

        Performs Kalman Filter prediction, associates detections with existing
        tracks based on IoU, updates matched tracks, and initializes new
        tracks for unmatched high-confidence detections.

        Args:
            detections (sv.Detections): The latest set of object detections from a frame.

        Returns:
            sv.Detections: A copy of the input detections, augmented with assigned
                `tracker_id` for each successfully tracked object. Detections not
                associated with a track will not have a `tracker_id`. The order of 
                the detections is not guaranteed to be the same as the input detections.
        """  # noqa: E501

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
        # Split into high confidence boxes and lower based on self.high_conf_boxes_threshold # noqa: E501
        high_prob_detections, low_prob_detections = (
            self._get_high_and_low_probability_detections(detections)
        )

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

    def _get_high_and_low_probability_detections(self, detections: sv.Detections)-> tuple[sv.Detections, sv.Detections]:
        """
        Splits the input detections into high-confidence and low-confidence sets
        based on the `self.high_conf_boxes_threshold`.

        Args:
            detections (sv.Detections): The input detections with confidence scores.

        Returns:
            tuple[sv.Detections, sv.Detections]: A tuple containing two
                sv.Detections objects: the first for high-confidence detections
                (confidence >= threshold) and the second for low-confidence detections
                (confidence < threshold).
        """
        # Check if confidence scores exist before comparing
        if detections.confidence is not None:
            # Perform element-wise comparison if confidence is a NumPy array
            condition = detections.confidence >= self.high_conf_boxes_threshold
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
        Jonker-Volgenant algorithm approach with no initialization instead of the Hungarian algorithm as mentioned in the SORT paper, but
        it solves the assignment problem in an optimal way.

        Args:
            similarity_matrix (np.ndarray): Similarity matrix between tracks (rows) and detections (columns).
            min_similarity_thresh (float): Minimum similarity threshold for a valid match.

        Returns:
            tuple[list[tuple[int, int]], set[int], set[int]]: Matched indices (list of (tracker_idx, detection_idx)),
                indices of unmatched tracks, indices of unmatched detections.
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
            detections (sv.Detections): Current detections.
            detection_boxes (np.ndarray): Bounding boxes for detections.
            unmatched_detections (set[int]): Indices of unmatched detections.
            updated_detections (list[sv.Detections]): List with all the detections

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

                    new_tracker = ByteTrackKalmanBoxTracker(
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
        tracks: list[ByteTrackKalmanBoxTracker],
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """Measures similarity based on IoU between tracks and detections and returns the matches 
            and unmatched tracks/detections. Is used for step 1 and 2 of the BYTE algorithm.

        Args:
            detections (sv.Detections): The set of object detections.
            tracks (list[ByteTrackKalmanBoxTracker]): The list of tracks that will be matched to the detections.

        Returns:
            tuple[list[tuple[int, int]], set[int], set[int]]: A tuple containing:
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
            self._get_associated_indices(
                similarity_matrix, thresh
            )
        )
        return matched_indices, unmatched_tracks, unmatched_detections

    def reset(self) -> None:
        """Resets the tracker's internal state.

        Clears all active tracks and resets the track ID counter.
        """
        self.tracks = []
        ByteTrackKalmanBoxTracker.count_id = 0
