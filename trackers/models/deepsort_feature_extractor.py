from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms

from trackers.models.deepsort_backbone import DeepSORTBackbone
from trackers.models.siamese_network import SiameseNetworkModel


class DeepSORTFeatureExtractor:
    """
    Feature extractor for DeepSORT that loads a PyTorch model and
    extracts appearance features from detection crops.
    """

    def __init__(
        self, model_path, device="mps", input_size: Tuple[int, int] = (128, 128)
    ):
        """
        Initialize the feature extractor with a PyTorch model.

        Args:
            model_path (str): Path to the PyTorch model checkpoint.
            device (str): Device to run the model on.
        """
        self.device = torch.device(device)
        self.input_size = input_size

        self._load_model(model_path)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Default feature dimension - used for fallback
        self.feature_dim = 512

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
        backbone_model = DeepSORTBackbone()
        backbone_model.load_state_dict(torch.load(model_path))
        self.model = SiameseNetworkModel(backbone_model=backbone_model)
        self.model.to(self.device)
        self.model.eval()

    def _ensure_valid_crop(self, crop):
        """
        Ensures the crop is valid and meets minimum size requirements.

        Args:
            crop (np.ndarray): Image crop to validate and fix if needed.

        Returns:
            np.ndarray: Valid crop with minimum dimensions.
        """
        # Check if crop is empty
        if crop.size == 0:
            # Return a blank image of the minimum size
            return np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)

        # This needs to be configurable
        min_height, min_width = 32, 32
        h, w = crop.shape[:2]

        # If smaller than minimum, resize with padding
        if h < min_height or w < min_width:
            # Create a canvas with the minimum size
            canvas = np.zeros(
                (max(min_height, h), max(min_width, w), 3), dtype=np.uint8
            )
            # Place the crop in the center
            y_offset = (canvas.shape[0] - h) // 2
            x_offset = (canvas.shape[1] - w) // 2
            if h > 0 and w > 0:  # Only try to copy if crop has non-zero dimensions
                canvas[y_offset : y_offset + h, x_offset : x_offset + w] = crop
            return canvas

        return crop

    def extract_features(self, frame, detections):
        """
        Extract features from detection crops in the frame.

        Note: this function was written by Claude, need to improve it.

        Args:
            frame (np.ndarray): The input frame.
            detections (sv.Detections): Detections from which to extract features.

        Returns:
            np.ndarray: Extracted features for each detection.
        """
        if len(detections) == 0:
            return np.array([])

        try:
            # Create a dummy input to determine output size
            dummy_input = torch.zeros(
                (1, 3, self.input_size[0], self.input_size[1])
            ).to(self.device)
            with torch.no_grad():
                dummy_output = self.model.forward_on_single_input(dummy_input)
            self.feature_dim = dummy_output.numel()
        except Exception as e:
            print(f"Error determining feature dimension: {e}")

        features = []
        with torch.no_grad():
            for box in detections.xyxy:
                # Get crop coordinates (ensure they are within frame boundaries)
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # Extract crop and ensure it meets minimum size requirements
                crop = frame[y1:y2, x1:x2]
                crop = self._ensure_valid_crop(crop)

                try:
                    tensor = self.transform(crop).unsqueeze(0).to(self.device)
                    feature = (
                        self.model.forward_on_single_input(tensor)
                        .cpu()
                        .numpy()
                        .flatten()
                    )
                    features.append(feature)
                except Exception as e:
                    print(f"Error extracting features: {e}")
                    # Fallback to zero features
                    features.append(np.zeros(self.feature_dim))

        return np.array(features)
