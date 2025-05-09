from typing import Optional

import numpy as np
import supervision as sv
import timm
import torch
import torch.nn as nn

from trackers.utils.torch_utils import parse_device_spec


class ReIDModel:
    def __init__(self, backbone_model: nn.Module, device: Optional[str] = "auto"):
        self.backbone_model = backbone_model
        self.device = parse_device_spec(device or "auto")
        self.backbone_model.to(self.device)
        self.backbone_model.eval()

    @classmethod
    def from_timm(
        cls,
        model_name: str,
        device: Optional[str] = "auto",
        pretrained: bool = True,
        get_pooled_features: bool = True,
        **kwargs,
    ):
        """
        Create a feature extractor from a
        [timm](https://huggingface.co/docs/timm) model.

        Args:
            model_name (str): Name of the timm model to use.
            device (str): Device to run the model on.
            pretrained (bool): Whether to use pretrained weights from timm or not.
            get_pooled_features (bool): Whether to get the pooled features from the
                model or not.
            **kwargs: Additional keyword arguments to pass to
                [`timm.create_model`](https://huggingface.co/docs/timm/en/reference/models#timm.create_model).

        Returns:
            DeepSORTFeatureExtractor: A new instance of DeepSORTFeatureExtractor.
        """
        if model_name not in timm.list_models(filter=model_name, pretrained=pretrained):
            raise ValueError(
                f"Model {model_name} not found in timm. "
                + "Please check the model name and try again."
            )
        if not get_pooled_features:
            kwargs["global_pool"] = ""
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, **kwargs
        )
        return cls(model, device)

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
                crop = sv.crop_image(image=frame, xyxy=[*box.astype(int)])
                tensor = self.transform(crop).unsqueeze(0).to(self.device)
                feature = torch.squeeze(self.model(tensor)).cpu().numpy().flatten()
                features.append(feature)

        return np.array(features)
