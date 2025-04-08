import numpy as np
import supervision as sv
import torch
import torchvision.transforms as transforms
from scipy.spatial.distance import cdist
from torch.serialization import add_safe_globals

try:
    from siamese_net import SiameseNetwork

    add_safe_globals([SiameseNetwork])
except ImportError:
    pass

from trackers.sort_tracker import KalmanBoxTracker, SORTTracker


class FeatureExtractor:
    """
    Feature extractor for DeepSORT that loads a PyTorch model and
    extracts appearance features from detection crops.
    """

    def __init__(self, model_path, device="mps"):
        """
        Initialize the feature extractor with a PyTorch model.

        Args:
            model_path (str): Path to the PyTorch model checkpoint.
            device (str, optional): Device to run the model on.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((128, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_model(self, model_path):
        """
        Load the PyTorch model from the given path.

        Note: this function was written by Claude, it works but I don't understand it.
        Need to find a better way to load the model.

        Args:
            model_path (str): Path to the model checkpoint.

        Returns:
            torch.nn.Module: Loaded PyTorch model.
        """
        try:
            # First try with weights_only=False to allow custom classes
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            return model
        except Exception as e:
            print(f"Error loading model with weights_only=False: {e}")

            try:
                # If that fails, try to dynamically import the SiameseNetwork class
                import importlib.util

                # Assume the module might be in the same directory as the checkpoint
                import os
                import sys

                potential_module_dir = os.path.dirname(os.path.abspath(model_path))

                # Look for siamese_net.py in the directory
                potential_module_path = os.path.join(
                    potential_module_dir, "siamese_net.py"
                )

                if os.path.exists(potential_module_path):
                    # If we found the module, load it
                    spec = importlib.util.spec_from_file_location(
                        "siamese_net", potential_module_path
                    )
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["siamese_net"] = module
                    spec.loader.exec_module(module)

                    # Add the SiameseNetwork class to safe globals
                    if hasattr(module, "SiameseNetwork"):
                        add_safe_globals([module.SiameseNetwork])
                        # Try loading again
                        model = torch.load(
                            model_path, map_location=self.device, weights_only=False
                        )
                        return model
            except Exception as nested_e:
                print(f"Error in dynamic import attempt: {nested_e}")

            # If all else fails, raise the original error
            raise RuntimeError(f"Failed to load model: {e}")

    def extract_features(self, frame, detections):
        """
        Extract features from detection crops in the frame.

        Args:
            frame (np.ndarray): The input frame.
            detections (sv.Detections): Detections from which to extract features.

        Returns:
            np.ndarray: Extracted features for each detection.
        """
        if len(detections) == 0:
            return np.array([])

        features = []
        with torch.no_grad():
            for box in detections.xyxy:
                # Get crop coordinates (ensure they are within frame boundaries)
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # Extract crop
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    # Assuming that the feature dimension is 512,
                    # this needs to be configurable
                    features.append(np.zeros(512))
                    continue

                try:
                    tensor = self.transform(crop).unsqueeze(0).to(self.device)
                    feature = self.model(tensor).cpu().numpy().flatten()
                    features.append(feature)
                except Exception as e:
                    print(f"Error extracting features: {e}")
                    # Fallback to zero features
                    features.append(np.zeros(512))

        return np.array(features)


class DeepSORTKalmanBoxTracker(KalmanBoxTracker):
    def __init__(self, bbox, feature=None, max_features=100):
        super().__init__(bbox)
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self.max_features = max_features

    def update_feature(self, feature):
        """
        Update the feature list for this tracker.

        Args:
            feature (np.ndarray): New feature to add.
        """
        if feature is not None:
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
        wt_path,
        lost_track_buffer=30,
        frame_rate=30.0,
        track_activation_threshold=0.25,
        minimum_consecutive_frames=3,
        minimum_iou_threshold=0.3,
        appearance_threshold=0.7,
        appearance_weight=0.5,
        device=None,
    ):
        super().__init__(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
        )

        self.feature_extractor = FeatureExtractor(wt_path, device)
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

        # Get track features (mean feature for each track)
        track_features = np.array([t.get_feature() for t in self.trackers])

        # For any tracker without features, use a fallback (high distance)
        for i, feature in enumerate(track_features):
            if feature is None:
                # Create a dummy feature with high distance to all detections
                track_features[i] = np.zeros_like(detection_features[0])

        # Calculate cosine distance
        distance_matrix = cdist(track_features, detection_features, metric="cosine")

        # Clip distances to [0, 1]
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

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the state of tracked objects with the newly received detections
        and returns the updated `sv.Detections` (including tracking IDs).

        Args:
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
        if (
            len(detections) > 0
            and hasattr(detections, "data")
            and "frame" in detections.data
        ):
            frame = detections.data["frame"]
            detection_features = self.feature_extractor.extract_features(
                frame, detections
            )
        else:
            # Dummy features
            detection_features = np.zeros((len(detection_boxes), 512))

        for tracker in self.trackers:
            tracker.predict()

        iou_matrix = self._get_iou_matrix(detection_boxes)
        matched_indices, unmatched_trackers, unmatched_detections = (
            self._get_associated_indices(iou_matrix, detection_features)
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
