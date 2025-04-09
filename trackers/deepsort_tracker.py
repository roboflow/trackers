import numpy as np
import supervision as sv
from scipy.spatial.distance import cdist

from trackers.models.deepsort_feature_extractor import DeepSORTFeatureExtractor
from trackers.sort_tracker import KalmanBoxTracker, SORTTracker


class DeepSORTKalmanBoxTracker(KalmanBoxTracker):
    def __init__(self, bbox, feature=None, max_features=100):
        super().__init__(bbox)
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self.max_features = max_features

    def update_feature(self, feature):
        self.features.append(feature)
        if len(self.features) > self.max_features:
            self.features.pop(0)

    def get_feature(self):
        """
        Get the mean feature vector for this tracker.

        Returns:
            np.ndarray: Mean feature vector.
        """
        if len(self.features) > 0:
            return np.mean(self.features, axis=0)
        return None


class DeepSORTTracker(SORTTracker):
    """
    DeepSORT implementation that extends SORTTracker with appearance features.

    This implementation follows the DeepSORT algorithm by incorporating both
    motion (IOU + Kalman filter) and appearance features for object tracking.

    ??? example
        ```python
        import numpy as np
        import supervision as sv
        from rfdetr import RFDETRBase
        from rfdetr.util.coco_classes import COCO_CLASSES

        from trackers import DeepSORTFeatureExtractor, DeepSORTTracker

        model = RFDETRBase(device="mps")
        feature_extractor = DeepSORTFeatureExtractor(model_path="model_state_dict.pth")
        tracker = DeepSORTTracker(feature_extractor=feature_extractor)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()


        def callback(frame: np.ndarray, _: int):
            detections = model.predict(frame, threshold=0.5)
            detections = tracker.update(frame, detections)
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

    Attributes:
        feature_extractor (FeatureExtractor): Model to extract appearance features.
        appearance_threshold (float): Cosine distance threshold for appearance matching.
        appearance_weight (float): Weight for appearance distance in the
            combined distance.

    Args:
        wt_path (str): Path to the feature extractor model checkpoint.
        lost_track_buffer (int): Number of frames to buffer when a track is lost.
        frame_rate (float): Frame rate of the video.
        track_activation_threshold (float): Detection confidence threshold for
            track activation.
        minimum_consecutive_frames (int): Frames needed for a valid track.
        minimum_iou_threshold (float): IOU threshold for association.
        appearance_threshold (float): Cosine distance threshold for appearance matching.
        appearance_weight (float): Weight for appearance distance (0-1).
        device (str, optional): Device to run the feature extractor on.
    """

    def __init__(
        self,
        feature_extractor: DeepSORTFeatureExtractor,
        lost_track_buffer=30,
        frame_rate=30.0,
        track_activation_threshold=0.25,
        minimum_consecutive_frames=3,
        minimum_iou_threshold=0.3,
        appearance_threshold=0.7,
        appearance_weight=0.5,
        device="mps",
    ):
        super().__init__(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
        )

        self.feature_extractor = feature_extractor
        self.appearance_threshold = appearance_threshold
        self.appearance_weight = appearance_weight

        self.trackers = []

    def _get_appearance_distance_matrix(self, detection_features):
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

        for i, feature in enumerate(track_features):
            if feature is None:
                # Create a dummy feature with high distance to all detections
                track_features[i] = np.zeros_like(detection_features[0])

        distance_matrix = cdist(track_features, detection_features, metric="cosine")
        distance_matrix = np.clip(distance_matrix, 0, 1)

        return distance_matrix

    def _get_combined_distance_matrix(self, iou_matrix, appearance_dist_matrix):
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
        self, iou_matrix, detection_features
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
        self, detections, detection_boxes, detection_features, unmatched_detections
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
                    detection_boxes[detection_idx], feature
                )
                self.trackers.append(new_tracker)

        self.trackers = self._get_alive_trackers()

    def update(self, frame: np.ndarray, detections: sv.Detections) -> sv.Detections:
        """
        Args:
            frame (np.ndarray): The current video frame.
            detections (sv.Detections): The latest set of object detections.

        Returns:
            sv.Detections: A copy of the detections with `tracker_id` set
                for each detection that is tracked.
        """
        if len(self.trackers) == 0 and len(detections) == 0:
            return detections

        detection_boxes = (
            detections.xyxy if len(detections) > 0 else np.array([]).reshape(0, 4)
        )

        detection_features = np.array([])
        detection_features = self.feature_extractor.extract_features(frame, detections)

        for tracker in self.trackers:
            tracker.predict()

        iou_matrix = self._get_iou_matrix(detection_boxes)
        matched_indices, _, unmatched_detections = self._get_associated_indices(
            iou_matrix, detection_features
        )

        for row, col in matched_indices:
            self.trackers[row].update(detection_boxes[col])
            if detection_features is not None and len(detection_features) > col:
                self.trackers[row].update_feature(detection_features[col])

        self._spawn_new_trackers(
            detections, detection_boxes, detection_features, unmatched_detections
        )

        updated_detections = self._update_detections_with_track_ids(
            detections, detection_boxes
        )

        return updated_detections
