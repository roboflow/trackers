from typing import Optional, Union

import numpy as np
import supervision as sv
import torch
from scipy.spatial.distance import cdist

from trackers.core.base import BaseTrackerWithFeatures
from trackers.core.deepsort.feature_extractor import DeepSORTFeatureExtractor
from trackers.core.deepsort.kalman_box_tracker import DeepSORTKalmanBoxTracker
from trackers.utils.sort_utils import (
    get_alive_trackers,
    get_iou_matrix,
    update_detections_with_track_ids,
)


class DeepSORTTracker(BaseTrackerWithFeatures):
    """
    DeepSORT implementation that extends SORTTracker with appearance features.
    The DeepSORT algorithm incorporates both motion (IOU + Kalman filter) and
    appearance features extracted by a pre-trained feature extraction model for
    object tracking.

    ??? example
        ```python
        import numpy as np
        import supervision as sv
        from rfdetr import RFDETRBase
        from rfdetr.util.coco_classes import COCO_CLASSES

        from trackers.core.deepsort.tracker import DeepSORTTracker

        model = RFDETRBase(device="mps")
        tracker = DeepSORTTracker()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()


        def callback(frame: np.ndarray, _: int):
            detections = model.predict(frame, threshold=0.5)
            detections = tracker.update(detections, frame)
            labels = [
                f"#{tracker_id} {COCO_CLASSES[class_id]} {confidence:.2f}"
                for tracker_id, class_id, confidence in zip(
                    detections.tracker_id, detections.class_id, detections.confidence
                )
            ]
            annotated_image = frame.copy()
            annotated_image = box_annotator.annotate(annotated_image, detections)
            annotated_image = label_annotator.annotate(
                annotated_image, detections, labels
            )

            return annotated_image


        sv.process_video(
            source_path="data/people.mp4",
            target_path="data/out.mp4",
            callback=callback,
            max_frames=100,
        )
        ```

    !!! example "Using custom weights for the Feature Extractor"
        ```python
        import numpy as np
        import supervision as sv
        from rfdetr import RFDETRBase
        from rfdetr.util.coco_classes import COCO_CLASSES

        from trackers import DeepSORTTracker, DeepSORTFeatureExtractor

        model = RFDETRBase(device="mps")
        tracker = DeepSORTTracker(
            feature_extractor=DeepSORTFeatureExtractor(
                model_or_checkpoint_path="deepsort_feature_extractor_weights.pth"
            )
        )
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()


        def callback(frame: np.ndarray, _: int):
            detections = model.predict(frame, threshold=0.5)
            detections = tracker.update(detections, frame)
            labels = [
                f"#{tracker_id} {COCO_CLASSES[class_id]} {confidence:.2f}"
                for tracker_id, class_id, confidence in zip(
                    detections.tracker_id, detections.class_id, detections.confidence
                )
            ]
            annotated_image = frame.copy()
            annotated_image = box_annotator.annotate(annotated_image, detections)
            annotated_image = label_annotator.annotate(
                annotated_image, detections, labels
            )

            return annotated_image


        sv.process_video(
            source_path="data/traffic_video.mp4",
            target_path="data/out.mp4",
            callback=callback,
            max_frames=100,
        )
        ```

    Attributes:
        feature_extractor (DeepSORTFeatureExtractor): Model to extract appearance
            features.
        appearance_threshold (float): Cosine distance threshold for appearance
            matching.
        appearance_weight (float): Weight for appearance distance in the
            combined distance.

    Args:
        feature_extractor (Optional[Union[DeepSORTFeatureExtractor, torch.nn.Module, str]]):
            A feature extractor model checkpoint URL, model checkpoint path, or model
            instance or an instance of `DeepSORTFeatureExtractor` to extract
            appearance features. By default, the a default model checkpoint is downloaded
            and loaded.
        device (Optional[str]): Device to run the model on.
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
        minimum_iou_threshold (float): IOU threshold for associating detections to
            existing tracks.
        appearance_threshold (float): Cosine distance threshold for appearance matching.
        appearance_weight (float): Weight for appearance distance (0-1).
        distance_metric (str): Distance metric to use for matching. Can be one of
            'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
            'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
            'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'yule'.
    """  # noqa: E501

    def __init__(
        self,
        feature_extractor: Optional[
            Union[DeepSORTFeatureExtractor, torch.nn.Module, str]
        ] = None,
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
        if feature_extractor is None:
            self.feature_extractor = DeepSORTFeatureExtractor(device=device)
        elif isinstance(feature_extractor, str):
            self.feature_extractor = DeepSORTFeatureExtractor(
                model_or_checkpoint_path=feature_extractor,
                device=device,
            )
        elif isinstance(feature_extractor, torch.nn.Module):
            self.feature_extractor = DeepSORTFeatureExtractor(
                model_or_checkpoint_path=feature_extractor,
                device=device,
            )
        else:
            self.feature_extractor = feature_extractor

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
        iou_distance = 1 - iou_matrix
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
                zip(row_indices, col_indices), key=lambda x: combined_dist[x[0], x[1]]
            )

            used_rows = set()
            used_cols = set()
            for row, col in sorted_pairs:
                if (row not in used_rows) and (col not in used_cols):
                    used_rows.add(row)
                    used_cols.add(col)
                    matched_indices.append((row, col))

            unmatched_trackers = unmatched_trackers - used_rows
            unmatched_detections = unmatched_detections - used_cols

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

        self.trackers = get_alive_trackers(
            trackers=self.trackers,
            maximum_frames_without_update=self.maximum_frames_without_update,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """
        Args:
            detections (sv.Detections): The latest set of object detections.
            frame (np.ndarray): The current video frame.

        Returns:
            sv.Detections: A copy of the detections with `tracker_id` set
                for each detection that is tracked.
        """
        if len(self.trackers) == 0 and len(detections) == 0:
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

        # Update detections with tracker IDs
        updated_detections = update_detections_with_track_ids(
            trackers=self.trackers,
            detections=detections,
            detection_boxes=detection_boxes,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            minimum_iou_threshold=self.minimum_iou_threshold,
        )

        return updated_detections
