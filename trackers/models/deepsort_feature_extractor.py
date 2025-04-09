from typing import Tuple

import numpy as np
import supervision as sv
import torch
import torchvision.transforms as transforms

from trackers.models.deepsort_backbone import DeepSORTBackbone
from trackers.models.siamese_network import SiameseNetworkModel


class DeepSORTFeatureExtractor:
    """
    Feature extractor for DeepSORT that loads a PyTorch model and
    extracts appearance features from detection crops.

    Args:
        model_path (str): Path to the PyTorch model checkpoint.
        device (str): Device to run the model on.
        input_size (Tuple[int, int]): Size to which the input
            images are resized.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        input_size: Tuple[int, int] = (128, 128),
    ):
        self.device = torch.device(device)
        self.input_size = input_size

        self._load_model(model_path)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
            ]
        )

    def _load_model(self, model_path):
        """
        Load the PyTorch model from the given path.

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

    def extract_features(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
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
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                tensor = self.transform(crop).unsqueeze(0).to(self.device)
                feature = self.model(tensor).cpu().numpy().flatten()
                features.append(feature)

        return np.array(features)
