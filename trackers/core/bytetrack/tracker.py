from copy import deepcopy
from typing import Optional, cast

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from trackers.core.base import BaseTrackerWithFeatures
from trackers.core.bytetrack.kalman_box_tracker import ByteTrackKalmanBoxTracker
from trackers.core.reid import ReIDModel
from trackers.utils.bytetrack_utils import (
    fuse_score,
)

from trackers.utils.sort_utils import (
    get_alive_trackers,
    get_iou_matrix,
)


class ByteTrackTracker(BaseTrackerWithFeatures):
    """Implements ByteTrack.

    ByteTrack is a simple, effective, and generic multi-object tracking method
    that improves upon tracking-by-detection by associating *every* detection box
    instead of discarding low-score ones. It uses a two-stage association process
    and builds on established techniques like the Kalman filter for motion prediction
    and the Hungarian algorithm for data association.

    Args:
        reid_model (Optional[ReIDModel]): An instance of a `ReIDModel` to extract
            appearance features or None if want to use IoU matching. Default is None. 
        lost_track_buffer (int): Number of frames to buffer when a track is lost.
            Increasing lost_track_buffer enhances occlusion handling, significantly
            improving tracking through occlusions, but may increase the possibility
            of ID switching for objects with similar appearance.
        frame_rate (float): Frame rate of the video (frames per second).
            Used to calculate the maximum time a track can be lost.
        track_activation_threshold (float): Detection confidence threshold
            for track activation. Only detections with confidence above this
            threshold will create new tracks. Increasing this threshold
            reduces false positives but may miss real objects with low confidence.
        minimum_consecutive_frames (int): Number of consecutive frames that an object
            must be tracked before it is considered a 'valid' track. Increasing
            `minimum_consecutive_frames` prevents the creation of accidental tracks
            from false detection or double detection, but risks missing shorter
            tracks. Before the tracker is considered valid, it will be assigned
            `-1` as its `tracker_id`.
        minimum_iou_threshold (float): IOU threshold for associating detections to existing tracks.
            Prevents the association of low IoU bounding boxes.
            A higher value will only associate boxes that have more overlapping when using IoU metric.
        appearance_threshold (float): Maximum allowed distance for appearance-based
            matching when using 'RE-ID' as the `high_prob_association_metric`.
            Prevents the association of detections which distance to the track isn't lower than the threshold.
            Lower values result in stricter appearance matching.
        distance_metric (str): Distance metric for appearance features (e.g., 'cosine',
            'euclidean'). See `scipy.spatial.distance.cdist`.
        high_prob_boxes_threshold (float): threshold for assigning predicted boxes to high probability class.
            A higher value will classify only higher probability boxes as 'high probability'
            per the ByteTrack algorithm, which are used in the first similarity step of
            the algorithm.  If feature extractor is used, high probability boxes are the
            only ones that are matched using the appearance features.
    """  # noqa: E501

    def __init__(
        self,
        reid_model: Optional[ReIDModel] = None,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.1,
        appearance_threshold: float = 0.5,
        distance_metric: str = "cosine",
        high_prob_boxes_threshold: float = 0.6,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold
        self.high_prob_boxes_threshold = high_prob_boxes_threshold
        self.high_prob_association_metric = "IoU" if reid_model is None else "RE-ID"
        self.reid_model = reid_model
        self.distance_metric = distance_metric
        self.trackers: list[ByteTrackKalmanBoxTracker] = []
        self.appearance_threshold = appearance_threshold

    def _update_detections(
        self,
        trackers: list[ByteTrackKalmanBoxTracker],
        detections: sv.Detections,
        updated_detections: list[sv.Detections],
        matched_indices: list[tuple[int, int]],
        associated_detection_features: Optional[np.ndarray] = None,
    ) -> list[sv.Detections]:
        # Update matched trackers with assigned detections.
        det_bboxes = detections.xyxy
        for row, col in matched_indices:
            t = trackers[row]
            current_detection_feature: Optional[np.ndarray] = None
            if associated_detection_features is not None and col < len(
                associated_detection_features
            ):
                current_detection_feature = associated_detection_features[col]

            t.update(det_bboxes[col], feature=current_detection_feature)
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

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """Updates the tracker state with new detections.

        Performs Kalman filter prediction, associates detections with existing
        trackers based on IOU, updates matched trackers, and initializes new
        trackers for unmatched high-confidence detections.

        Args:
            detections (sv.Detections): The latest set of object detections from a frame.
            frame (np.ndarray): The current video frame, used for extracting
                appearance features from detections.

        Returns:
            sv.Detections: A copy of the input detections, augmented with assigned
                `tracker_id` for each successfully tracked object. Detections not
                associated with a track will not have a `tracker_id`.
        """  # noqa: E501

        if len(self.trackers) == 0 and len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections
        updated_detections: list[
            sv.Detections
        ] = []  # List for returning the updated detections with its new assigned track id # noqa: E501

        # Predict new locations for existing trackers
        for tracker in self.trackers:
            tracker.predict()
        # Assign a default tracker_id with the correct shape
        detections.tracker_id = -np.ones(len(detections))
        # Split into high confidence boxes and lower based on self.high_prob_boxes_threshold # noqa: E501
        high_prob_detections, low_prob_detections = (
            self._get_high_and_low_probability_detections(detections)
        )

        detection_features: Optional[np.ndarray] = (
            self.reid_model.extract_features(high_prob_detections, frame)
            if self.reid_model is not None
            else None
        )

        # Step 1: first association, with high confidence boxes
        matched_indices, unmatched_trackers, unmatched_high_prob_detections = (
            self._similarity_step(
                high_prob_detections,
                self.trackers,
                self.high_prob_association_metric,
                detection_features,
                fuse_enabled=True
            )
        )

        # Update matched trackers with high-confidence detections
        # with associated appearance features
        self._update_detections(
            self.trackers,
            high_prob_detections,
            updated_detections,
            matched_indices,
            detection_features,
        )

        remaining_trackers = [self.trackers[i] for i in unmatched_trackers]

        # Step 2: associate Low Probability detections with remaining trackers
        matched_indices, unmatched_trackers, unmatched_detections = (
            self._similarity_step(low_prob_detections, remaining_trackers, "IoU")
        )

        # Update matched trackers with low-confidence detections
        # without associated appearance features
        self._update_detections(
            remaining_trackers,
            low_prob_detections,
            updated_detections,
            matched_indices,
            None,
        )

        # Add unmatched low prob predictions to updated predictions
        for det_index in unmatched_detections:
            new_det = deepcopy(low_prob_detections[det_index : det_index + 1])

            new_det.tracker_id = np.array([-1])
            updated_detections.append(new_det)

        self._spawn_new_trackers(
            high_prob_detections,
            high_prob_detections.xyxy,
            detection_features,
            unmatched_high_prob_detections,
            updated_detections,
        )

        # Kill lost tracks
        self.trackers = get_alive_trackers(
            trackers=self.trackers,
            maximum_frames_without_update=self.maximum_frames_without_update,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )
        final_updated_detections: sv.Detections = sv.Detections.merge(
            updated_detections
        )
        if len(final_updated_detections) == 0:
            final_updated_detections.tracker_id = np.array([], dtype=int)
        return final_updated_detections

    def _get_high_and_low_probability_detections(self, detections: sv.Detections):
        """
        Splits the input detections into high-confidence and low-confidence sets
        based on the `self.high_prob_boxes_threshold`.

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
            condition = detections.confidence >= self.high_prob_boxes_threshold
        else:
            # If no confidence scores, no detections meet the threshold
            # Create a boolean array of False with the same length as detections
            condition = np.zeros(len(detections), dtype=bool)

        high_confidence = detections[condition]

        not_low = detections.confidence >0.1
        remaining = np.logical_not(condition)


        low_confidence = detections[np.logical_and(remaining, not_low)]
        return high_confidence, low_confidence

    def _get_associated_indices(
        self,
        cost_matrix: np.ndarray,
        detection_boxes: np.ndarray,
        trackers: list[ByteTrackKalmanBoxTracker],
        max_cost_thresh: float,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """
        Associate detections to trackers based on cost (1 - IoU or Distance between appeareance features) using the
        Jonker-Volgenant algorithm approach with no initialization instead of the Hungarian algorithm as mentioned in the SORT paper, but
        it solves the assignment problem in an optimal way.

        Args:
            cost_matrix (np.ndarray): Distance/Cost matrix between trackers (rows) and detections (columns).
            detections (sv.Detections): The set of object detections.
            trackers (list[ByteTrackKalmanBoxTracker]): The list of trackers.
            max_cost_thresh (float): Maximum cost threshold for a valid match. THIS HAS TO BE CHANGED YET TO MATCH ACTUAL BEHAVIOR

        Returns:
            tuple[list[tuple[int, int]], set[int], set[int]]: Matched indices (list of (tracker_idx, detection_idx)),
                indices of unmatched trackers, indices of unmatched detections.
        """  # noqa: E501
        matched_indices = []
        unmatched_trackers = set(range(len(trackers)))
        unmatched_detections = set(range(len(detection_boxes)))

        if len(trackers) > 0 and len(detection_boxes) > 0:
            row_indices, col_indices = linear_sum_assignment(
                cost_matrix, maximize=False
            )
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] <= max_cost_thresh:
                    matched_indices.append((row, col))
                    unmatched_trackers.remove(row)
                    unmatched_detections.remove(col)

        return matched_indices, unmatched_trackers, unmatched_detections

    def _spawn_new_trackers(
        self,
        detections: sv.Detections,
        detection_boxes: np.ndarray,
        detection_features: Optional[np.ndarray],
        unmatched_detections: set[int],
        updated_detections: list[sv.Detections],
    ):
        """
        Create new trackers for unmatched detections and
            append detections to updated_detections detections.

        Args:
            detections (sv.Detections): Current detections.
            detection_boxes (np.ndarray): Bounding boxes for detections.
            detection_features (Optional[np.ndarray]): Features for detections.
            unmatched_detections (set[int]): Indices of unmatched detections.
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
                if (
                    confidence_score >= self.track_activation_threshold
                ):  # Changed to use confidence_score
                    # Original logic for high confidence detection
                    feature: Optional[np.ndarray] = None
                    if (
                        detection_features is not None
                        and len(detection_features) > detection_idx
                    ):
                        feature = detection_features[detection_idx]

                    new_tracker = ByteTrackKalmanBoxTracker(
                        bbox=detection_boxes[detection_idx], feature=feature
                    )
                    self.trackers.append(new_tracker)

                    new_det = deepcopy(detections[detection_idx : detection_idx + 1])
                    new_det = cast(sv.Detections, new_det)  # Cast added previously
                    new_det.tracker_id = np.array([-1])
                    updated_detections.append(new_det)
            else:
                pass  # Do nothing, the detection remains unmatched

    def _similarity_step(
        self,
        detections: sv.Detections,
        trackers: list[ByteTrackKalmanBoxTracker],
        association_metric: str,
        detection_features: Optional[np.ndarray] = None,
        fuse_enabled = False,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """Measures similarity as indicated by the user between tracks and detections and returns the matches and unmatched trackers/detections.
            Is useful for step 1 and 2 of the BYTE algorithm.

        Args:
            detections (sv.Detections): The set of object detections.
            trackers (list[ByteTrackKalmanBoxTracker]): The list of trackers that will we matched to the detections.
            association_metric (str): The metric that will compare the detections with the trackers. Can be either object features (RE-ID) or
                based on location (IoU).
            detection_features (Optional[np.ndarray]): Features extracted from detections, used for 'RE-ID' association. Defaults to None.
            fuse_enabled (bool): Whether to apply score fusion (enforces matches for each detection based on detection confidence)
                to the cost matrix. Defaults to False.
        Returns:
            tuple[list[tuple[int, int]], set[int], set[int]]: A tuple containing:
                - matched_indices: A list of (tracker_idx, detection_idx) pairs.
                - unmatched_trackers_indices: A set of indices for trackers that
                  were not matched.
                - unmatched_detections_indices: A set of indices for detections
                  that were not matched.

        Raises:
            Exception: If an unsupported `association_metric` is provided.
        """  # noqa: E501
        cost_matrix = None
        if association_metric == "IoU":
            # Build IOU cost matrix between detections and predicted bounding boxes
            cost_matrix = 1 - get_iou_matrix(trackers, detections.xyxy)
            if fuse_enabled:
                cost_matrix = fuse_score(cost_matrix, detections)
            thresh = 1- self.minimum_iou_threshold
        elif association_metric == "RE-ID" and detection_features is not None:
            # Build feature distance matrix between detections and predicted bounding boxes # noqa: E501
            cost_matrix = self._get_appearance_distance_matrix(
                detection_features, trackers
            )
            thresh = self.appearance_threshold

        else:
            raise Exception("Your association metric is not supported")
        # Associate detections to trackers based on the lower value of the
        # cost matrix, using the Jonker-Volgenant algorithm (linear_sum_assignment). # noqa: E501
        matched_indices, unmatched_trackers, unmatched_detections = (
            self._get_associated_indices(
                cost_matrix, detections.xyxy, trackers, thresh
            )
        )
        return matched_indices, unmatched_trackers, unmatched_detections

    def reset(self) -> None:
        """Resets the tracker's internal state.

        Clears all active tracks and resets the track ID counter.
        """
        self.trackers = []
        ByteTrackKalmanBoxTracker.count_id = 0

    def _get_appearance_distance_matrix(
        self,
        detection_features: np.ndarray,
        trackers: list[ByteTrackKalmanBoxTracker],
    ) -> np.ndarray:
        """
        Calculate appearance distance matrix between tracks and detections.

        Args:
            detection_features (np.ndarray): Features extracted from current detections.
            trackers (list[ByteTrackKalmanBoxTracker]): trackers to be compared.
        Returns:
            np.ndarray: Appearance distance matrix (rows: trackers, columns: detections).
        """  # noqa: E501

        if len(trackers) == 0 or len(detection_features) == 0:  # Handle empty cases
            return np.zeros((len(trackers), len(detection_features)), dtype=np.float32)
        if len(trackers) > 0 and len(detection_features) > 0:
            track_features = np.array([t.get_feature() for t in trackers])
            distance_matrix = cdist(
                track_features, detection_features, metric=self.distance_metric
            )
            distance_matrix = np.clip(distance_matrix, 0, 1)
        else:
            distance_matrix = np.zeros(
                (len(trackers), len(detection_features)), dtype=np.float32
            )
        return distance_matrix
