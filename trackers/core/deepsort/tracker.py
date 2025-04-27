from typing import List, Optional, Tuple, Union

import numpy as np
import supervision as sv
import torch
from scipy.spatial.distance import cdist

from trackers.core.base import BaseTrackerWithFeatures
from trackers.core.deepsort.feature_extractor import DeepSORTFeatureExtractor
from trackers.core.deepsort.kalman_box_tracker import (
    MAHALANOBIS_THRESHOLD,
    DeepSORTKalmanBoxTracker,
)
from trackers.utils.sort_utils import (
    get_alive_trackers,
    get_iou_matrix,
    update_detections_with_track_ids,
    xyxy_to_xcycarh,
)


class DeepSORTTracker(BaseTrackerWithFeatures):
    """Implements DeepSORT (Deep Simple Online and Realtime Tracking).

    DeepSORT extends SORT by integrating appearance information using a deep
    learning model, improving tracking through occlusions and reducing ID switches.
    It combines motion (Kalman filter) and appearance cues for data association.

    Args:
        feature_extractor (Union[DeepSORTFeatureExtractor, torch.nn.Module, str]):
            A feature extractor model checkpoint URL, model checkpoint path, a model
            instance, or an instance of `DeepSORTFeatureExtractor` to extract
            appearance features. By default, a default model checkpoint is downloaded
            and loaded.
        device (Optional[str]): Device to run the feature extraction
            model on (e.g., 'cpu', 'cuda').
        lost_track_buffer (int): Number of frames to buffer when a track is lost.
            Enhances occlusion handling but may increase ID switches for similar objects.
        frame_rate (float): Frame rate of the video (frames per second).
            Used to calculate the maximum time a track can be lost.
        track_activation_threshold (float): Detection confidence threshold
            for track activation. Higher values reduce false positives
            but might miss objects.
        minimum_consecutive_frames (int): Number of consecutive frames an object
            must be tracked to be considered 'valid'. Prevents spurious tracks but
            may miss short tracks.
        minimum_iou_threshold (float): IOU threshold for gating in the matching cascade.
        appearance_threshold (float): Cosine distance threshold for appearance matching.
            Only matches below this threshold are considered valid.
        appearance_weight (float): Weight (0-1) balancing motion (IOU) and appearance
            distance in the combined matching cost.
        distance_metric (str): Distance metric for appearance features (e.g., 'cosine',
            'euclidean'). See `scipy.spatial.distance.cdist`.
    """  # noqa: E501

    def __init__(
        self,
        feature_extractor: Union[DeepSORTFeatureExtractor, torch.nn.Module, str],
        device: Optional[str] = None,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
        appearance_threshold: float = 0.7,
        appearance_weight: float = 0.5,
        distance_metric: str = "cosine",
    ):
        self.feature_extractor = self._initialize_feature_extractor(
            feature_extractor, device
        )

        self.lost_track_buffer = lost_track_buffer
        self.frame_rate = frame_rate
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold
        self.appearance_threshold = appearance_threshold
        self.appearance_weight = appearance_weight
        self.distance_metric = distance_metric
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(
            self.frame_rate / 30.0 * self.lost_track_buffer
        )

        self.trackers: list[DeepSORTKalmanBoxTracker] = []

    def _initialize_feature_extractor(
        self,
        feature_extractor: Union[DeepSORTFeatureExtractor, torch.nn.Module, str],
        device: Optional[str],
    ) -> DeepSORTFeatureExtractor:
        """
        Initialize the feature extractor based on the input type.

        Args:
            feature_extractor: The feature extractor input, which can be a model path,
                a torch module, or a DeepSORTFeatureExtractor instance.
            device: The device to run the model on.

        Returns:
            DeepSORTFeatureExtractor: The initialized feature extractor.
        """
        if isinstance(feature_extractor, (str, torch.nn.Module)):
            return DeepSORTFeatureExtractor(
                model_or_checkpoint_path=feature_extractor,
                device=device,
            )
        else:
            return feature_extractor

    def _get_appearance_distance_matrix(
        self,
        detection_features: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate appearance distance matrix between tracks and detections.

        Args:
            detection_features (np.ndarray): Features extracted from current detections.

        Returns:
            np.ndarray: Appearance distance matrix.
        """

        if len(self.trackers) == 0 or len(detection_features) == 0:
            return np.zeros((len(self.trackers), len(detection_features)))

        track_features = np.array([t.get_feature() for t in self.trackers])
        distance_matrix = cdist(
            track_features, detection_features, metric=self.distance_metric
        )
        distance_matrix = np.clip(distance_matrix, 0, 1)

        return distance_matrix

    def _get_mahalanobis_distance_matrix(
        self,
        detection_boxes: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Mahalanobis distance matrix between tracks and detections,
        as per Equation 1 in section 2.2 of the [DeepSORT paper](https://arxiv.org/pdf/1703.07402).

        Args:
            detection_boxes (np.ndarray): Detected bounding boxes in the
                form [x1, y1, x2, y2].

        Returns:
            np.ndarray: Mahalanobis distance matrix.
        """
        if len(self.trackers) == 0 or len(detection_boxes) == 0:
            return np.zeros((len(self.trackers), len(detection_boxes)))

        distance_matrix = np.zeros((len(self.trackers), len(detection_boxes)))

        for i, tracker in enumerate(self.trackers):
            measurements = np.array([xyxy_to_xcycarh(box) for box in detection_boxes])

            distance_matrix[i, :] = tracker.compute_gating_distance(measurements)

        return distance_matrix

    def _get_combined_distance_matrix(
        self,
        mahalanobis_matrix: np.ndarray,
        appearance_dist_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Combine Mahalanobis and appearance distances into a single distance matrix,
        as per Equation 5 in section 2.2 of the [DeepSORT paper](https://arxiv.org/pdf/1703.07402).

        Args:
            mahalanobis_matrix (np.ndarray): Mahalanobis distance matrix between
                tracks and detections.
            appearance_dist_matrix (np.ndarray): Appearance distance matrix.

        Returns:
            np.ndarray: Combined distance matrix.
        """
        # Using weighted sum to combine Mahalanobis and appearance distances
        combined_dist = (
            self.appearance_weight * appearance_dist_matrix
            + (1 - self.appearance_weight) * mahalanobis_matrix
        )

        mahalanobis_gate = mahalanobis_matrix > MAHALANOBIS_THRESHOLD
        appearance_gate = appearance_dist_matrix > self.appearance_threshold

        # An association is inadmissible if either metric is above threshold
        invalid_mask = np.logical_or(mahalanobis_gate, appearance_gate)
        combined_dist[invalid_mask] = 1.0  # Mark as infeasible

        return combined_dist

    def _match_tracks_stage(
        self,
        cost_matrix: np.ndarray,
        track_indices: list,
        detection_indices: list,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match tracks with detections for a specific stage of the matching cascade.
        This implements the linear assignment for a specific group of tracks based
        on their maturity.

        Args:
            cost_matrix (np.ndarray): Cost matrix between tracks and detections.
            track_indices (list): Indices of tracks to match.
            detection_indices (list): Indices of detections to match.

        Returns:
            tuple[list[tuple[int, int]], list[int], list[int]]: Matched indices,
                unmatched track indices, unmatched detection indices.
        """
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return [], track_indices, detection_indices

        sub_cost_matrix = cost_matrix[np.ix_(track_indices, detection_indices)]

        # Apply threshold of 1.0 to mark infeasible associations
        # Only consider associations where cost < 1.0
        valid_mask = sub_cost_matrix < 1.0

        if not np.any(valid_mask):
            return [], track_indices, detection_indices

        row_indices, col_indices = np.where(valid_mask)

        indices = np.stack([row_indices, col_indices], axis=1)
        indices = indices[np.argsort(sub_cost_matrix[row_indices, col_indices])]

        matches = []
        unmatched_tracks = list(track_indices)
        unmatched_detections = list(detection_indices)

        matched_track_indices = set()
        matched_detection_indices = set()

        for row, col in indices:
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]

            # Skip if either track or detection is already matched
            if row in matched_track_indices or col in matched_detection_indices:
                continue

            matches.append((track_idx, detection_idx))
            matched_track_indices.add(row)
            matched_detection_indices.add(col)

            if track_idx in unmatched_tracks:
                unmatched_tracks.remove(track_idx)
            if detection_idx in unmatched_detections:
                unmatched_detections.remove(detection_idx)

        return matches, unmatched_tracks, unmatched_detections

    def _get_associated_indices(
        self,
        detection_boxes: np.ndarray,
        detection_features: np.ndarray,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """
        Associate detections to trackers using a cascade matching approach.
        The cascade gives priority to tracks that have been recently seen.

        As per the paper, we use a weighted combination of Mahalanobis distance
        (for motion information) and appearance distance (for visual similarity).

        Args:
            detection_boxes (np.ndarray): Detected bounding boxes in the
                form [x1, y1, x2, y2].
            detection_features (np.ndarray): Features extracted from current detections.

        Returns:
            tuple[list[tuple[int, int]], set[int], set[int]]: Matched indices,
                unmatched trackers, unmatched detections.
        """
        confirmed_tracks = []
        unconfirmed_tracks = []
        for tracker_idx, tracker in enumerate(self.trackers):
            if tracker.number_of_successful_updates >= self.minimum_consecutive_frames:
                confirmed_tracks.append(tracker_idx)
            else:
                unconfirmed_tracks.append(tracker_idx)

        # Get Mahalanobis distance matrix
        mahalanobis_matrix = self._get_mahalanobis_distance_matrix(detection_boxes)

        # Get appearance distance matrix
        appearance_dist_matrix = self._get_appearance_distance_matrix(
            detection_features
        )

        # Combine distances using weighted sum
        combined_dist_matrix = self._get_combined_distance_matrix(
            mahalanobis_matrix, appearance_dist_matrix
        )

        confirmed_matches, unmatched_confirmed, unmatched_detections = (
            self._match_tracks_stage(
                combined_dist_matrix,
                confirmed_tracks,
                list(range(len(detection_features))),
            )
        )

        # Find recently lost confirmed tracks (time_since_update == 1)
        recently_lost = [
            tracker_idx
            for tracker_idx in unmatched_confirmed
            if self.trackers[tracker_idx].time_since_update == 1
        ]

        # Remove recently lost from unmatched_confirmed
        unmatched_confirmed = [
            tracker_idx
            for tracker_idx in unmatched_confirmed
            if tracker_idx not in recently_lost
        ]

        iou_track_candidates = unconfirmed_tracks + recently_lost

        # Match remaining tracks using IoU only
        iou_matrix = get_iou_matrix(
            trackers=self.trackers, detection_boxes=detection_boxes
        )

        iou_matches: list[tuple[int, int]] = []
        if iou_track_candidates and unmatched_detections:
            iou_dist_matrix = 1 - iou_matrix
            iou_dist_matrix_filtered = iou_dist_matrix.copy()
            mask = iou_matrix < self.minimum_iou_threshold
            iou_dist_matrix_filtered[mask] = 1.0

            iou_matches, unmatched_candidates, unmatched_detections = (
                self._match_tracks_stage(
                    iou_dist_matrix_filtered,
                    iou_track_candidates,
                    list(unmatched_detections),
                )
            )
        else:
            unmatched_candidates = iou_track_candidates

        matches = confirmed_matches + iou_matches
        unmatched_tracks = set(unmatched_confirmed).union(set(unmatched_candidates))

        return matches, unmatched_tracks, set(unmatched_detections)

    def _spawn_new_trackers(
        self,
        detections: sv.Detections,
        detection_boxes: np.ndarray,
        detection_features: np.ndarray,
        unmatched_detections: set[int],
    ):
        """
        Create new trackers for unmatched detections with confidence above threshold.

        Args:
            detections (sv.Detections): Current detections.
            detection_boxes (np.ndarray): Bounding boxes for detections.
            detection_features (np.ndarray): Features for detections.
            unmatched_detections (set[int]): Indices of unmatched detections.
        """
        for detection_idx in unmatched_detections:
            if (
                detections.confidence is None
                or detection_idx >= len(detections.confidence)
                or detections.confidence[detection_idx]
                >= self.track_activation_threshold
            ):
                feature = None
                if (
                    detection_features is not None
                    and len(detection_features) > detection_idx
                ):
                    feature = detection_features[detection_idx]

                new_tracker = DeepSORTKalmanBoxTracker(
                    bbox=detection_boxes[detection_idx], feature=feature
                )
                self.trackers.append(new_tracker)

        self.trackers = get_alive_trackers(
            trackers=self.trackers,
            maximum_frames_without_update=self.maximum_frames_without_update,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """Updates the tracker state with new detections and appearance features.

        Extracts appearance features, performs Kalman filter prediction, calculates
        IOU and appearance distance matrices, associates detections with tracks using
        a combined metric, updates matched tracks (position and appearance), and
        initializes new tracks for unmatched high-confidence detections.

        Args:
            detections (sv.Detections): The latest set of object detections.
            frame (np.ndarray): The current video frame, used for extracting
                appearance features from detections.

        Returns:
            sv.Detections: A copy of the input detections, augmented with assigned
                `tracker_id` for each successfully tracked object. Detections not
                associated with a track will not have a `tracker_id`.
        """
        if len(self.trackers) == 0 and len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections

        # Convert detections to a (N x 4) array (x1, y1, x2, y2)
        detection_boxes = (
            detections.xyxy if len(detections) > 0 else np.array([]).reshape(0, 4)
        )

        # Extract appearance features from the frame and detections
        detection_features = self.feature_extractor.extract_features(frame, detections)

        # Predict new locations for existing trackers
        for tracker in self.trackers:
            tracker.predict()

        # Associate detections to trackers using the cascade matching approach
        matched_indices, unmatched_tracks, unmatched_detections = (
            self._get_associated_indices(detection_boxes, detection_features)
        )

        # Update matched trackers with assigned detections
        for row, col in matched_indices:
            self.trackers[row].update(detection_boxes[col])
            if detection_features is not None and len(detection_features) > col:
                self.trackers[row].update_feature(detection_features[col])

        # Create new trackers for unmatched detections with confidence above threshold
        self._spawn_new_trackers(
            detections, detection_boxes, detection_features, unmatched_detections
        )

        # Update detections with tracker IDs
        updated_detections = update_detections_with_track_ids(
            trackers=self.trackers,
            detections=detections,
            detection_boxes=detection_boxes,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            minimum_iou_threshold=self.minimum_iou_threshold,
        )

        return updated_detections

    def reset(self) -> None:
        """Resets the tracker's internal state.

        Clears all active tracks and resets the track ID counter.
        """
        self.trackers = []
        DeepSORTKalmanBoxTracker.count_id = 0
