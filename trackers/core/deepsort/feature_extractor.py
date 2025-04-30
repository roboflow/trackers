from typing import Optional, Tuple, Union

import numpy as np
import supervision as sv
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from firerequests import FireRequests

from trackers.utils.torch_utils import parse_device_spec


class FeatureExtractionBackbone(nn.Module):
    """
    A simple backbone model for feature extraction.

    Args:
        backbone_model (nn.Module): The backbone model to use for feature extraction.
    """

    def __init__(self, backbone_model: nn.Module):
        super(FeatureExtractionBackbone, self).__init__()
        self.backbone_model = backbone_model

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on a single input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor.
        """
        output = self.backbone_model(input_tensor)
        output = torch.squeeze(output)
        return output


class DeepSORTFeatureExtractor:
    """
    Feature extractor for DeepSORT that loads a PyTorch model and
    extracts appearance features from detection crops.

    Args:
        model_or_checkpoint_path (Union[str, torch.nn.Module]): Path/URL to the PyTorch
            model checkpoint or the model itself.
        device (str): Device to run the model on.
        input_size (Tuple[int, int]): Size to which the input images are resized.
    """

    def __init__(
        self,
        model_or_checkpoint_path: Union[str, torch.nn.Module],
        device: Optional[str] = "auto",
        input_size: Tuple[int, int] = (128, 128),
    ):
        self.device = parse_device_spec(device or "auto")
        self.input_size = input_size

        self._initialize_model(model_or_checkpoint_path)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
            ]
        )

    @classmethod
    def from_timm(
        cls,
        model_name: str,
        device: Optional[str] = "auto",
        input_size: Tuple[int, int] = (128, 128),
        pretrained: bool = True,
        get_pooled_features: bool = True,
        **kwargs,
    ):
        """
        Create a feature extractor from a timm model.

        Args:
            model_name (str): Name of the timm model to use.
            device (str): Device to run the model on.
            input_size (Tuple[int, int]): Size to which the input images are resized.
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
        model = timm.create_model(model_name,  pretrained=pretrained, num_classes=0,**kwargs)
        backbone_model = FeatureExtractionBackbone(model)
        return cls(backbone_model, device, input_size)

    def _initialize_model(
        self, model_or_checkpoint_path: Union[str, torch.nn.Module, None]
    ):
        if isinstance(model_or_checkpoint_path, str):
            import validators

            if validators.url(model_or_checkpoint_path):
                checkpoint_path = FireRequests().download(model_or_checkpoint_path)[0]
                self._load_model_from_path(checkpoint_path)
            else:
                self._load_model_from_path(model_or_checkpoint_path)
        elif isinstance(model_or_checkpoint_path, torch.nn.Module):
            self.model = model_or_checkpoint_path
            self.model.to(self.device)
            self.model.eval()
        else:
            raise TypeError(
                "model_or_checkpoint_path must be a string (path/URL) "
                "or a torch.nn.Module instance."
            )

    def _load_model_from_path(self, model_path):
        """
        Load the PyTorch model from the given path.

        Args:
            model_path (str): Path to the model checkpoint.

        Returns:
            torch.nn.Module: Loaded PyTorch model.
        """
        self.model = FeatureExtractionBackbone(torch.load(model_path))
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
                crop = sv.crop_image(image=frame, xyxy=[*box.astype(int)])
                tensor = self.transform(crop).unsqueeze(0).to(self.device)
                feature = self.model(tensor).cpu().numpy().flatten()
                features.append(feature)

        return np.array(features)
