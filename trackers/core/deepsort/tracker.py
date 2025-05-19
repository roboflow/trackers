from typing import Optional

import numpy as np
import supervision as sv
from scipy.spatial.distance import cdist

from trackers.core.base import BaseTrackerWithFeatures
from trackers.core.deepsort.kalman_box_tracker import DeepSORTKalmanBoxTracker
from trackers.core.reid import ReIDModel
from trackers.utils.sort_utils import (
    get_alive_trackers,
    get_iou_matrix,
    update_detections_with_track_ids,
)


class DeepSORTTracker(BaseTrackerWithFeatures):
    """Implements DeepSORT (Deep Simple Online and Realtime Tracking).

    DeepSORT extends SORT by integrating appearance information using a deep
    learning model, improving tracking through occlusions and reducing ID switches.
    It combines motion (Kalman filter) and appearance cues for data association.

    Args:
        reid_model (ReIDModel): An instance of a `ReIDModel` to extract
            appearance features.
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
        reid_model: ReIDModel,
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
        self.reid_model = reid_model
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

    def _get_combined_distance_matrix(
        self,
        iou_matrix: np.ndarray,
        appearance_dist_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Combine IOU and appearance distances into a single distance matrix.

        Args:
            iou_matrix (np.ndarray): IOU matrix between tracks and detections.
            appearance_dist_matrix (np.ndarray): Appearance distance matrix.

        Returns:
            np.ndarray: Combined distance matrix.
        """
        iou_distance: np.ndarray = 1 - iou_matrix
        combined_dist = (
            1 - self.appearance_weight
        ) * iou_distance + self.appearance_weight * appearance_dist_matrix

        # Set high distance for IOU below threshold
        mask = iou_matrix < self.minimum_iou_threshold
        combined_dist[mask] = 1.0

        # Set high distance for appearance above threshold
        mask = appearance_dist_matrix > self.appearance_threshold
        combined_dist[mask] = 1.0

        return combined_dist

    def _get_associated_indices(
        self,
        iou_matrix: np.ndarray,
        detection_features: np.ndarray,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """
        Associate detections to trackers based on both IOU and appearance.

        Args:
            iou_matrix (np.ndarray): IOU matrix between tracks and detections.
            detection_features (np.ndarray): Features extracted from current detections.

        Returns:
            tuple[list[tuple[int, int]], set[int], set[int]]: Matched indices,
                unmatched trackers, unmatched detections.
        """
        appearance_dist_matrix = self._get_appearance_distance_matrix(
            detection_features
        )
        combined_dist = self._get_combined_distance_matrix(
            iou_matrix, appearance_dist_matrix
        )
        matched_indices = []
        unmatched_trackers = set(range(len(self.trackers)))
        unmatched_detections = set(range(len(detection_features)))

        if combined_dist.size > 0:
            row_indices, col_indices = np.where(combined_dist < 1.0)
            sorted_pairs = sorted(
                zip(map(int, row_indices), map(int, col_indices)),
                key=lambda x: combined_dist[x[0], x[1]],
            )

            used_rows = set()
            used_cols = set()
            for row, col in sorted_pairs:
                if (row not in used_rows) and (col not in used_cols):
                    used_rows.add(row)
                    used_cols.add(col)
                    matched_indices.append((row, col))

            unmatched_trackers = unmatched_trackers - {int(row) for row in used_rows}
            unmatched_detections = unmatched_detections - {
                int(col) for col in used_cols
            }

        return matched_indices, unmatched_trackers, unmatched_detections

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
        detection_features = self.reid_model.extract_features(detections, frame)

        # Predict new locations for existing trackers
        for tracker in self.trackers:
            tracker.predict()

        # Build IOU cost matrix between detections and predicted bounding boxes
        iou_matrix = get_iou_matrix(
            trackers=self.trackers, detection_boxes=detection_boxes
        )

        # Associate detections to trackers based on IOU
        matched_indices, _, unmatched_detections = self._get_associated_indices(
            iou_matrix, detection_features
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

        # Remove dead trackers
        self.trackers = get_alive_trackers(
            trackers=self.trackers,
            maximum_frames_without_update=self.maximum_frames_without_update,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
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
