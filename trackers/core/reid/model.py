from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import PIL
import supervision as sv
import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.transforms import Compose, ToPILImage

from trackers.log import get_logger
from trackers.utils.torch_utils import parse_device_spec

logger = get_logger(__name__)


def _initialize_reid_model_from_timm(
    cls,
    model_name: str,
    device: Optional[str] = "auto",
    get_pooled_features: bool = True,
    **kwargs,
):
    if not get_pooled_features:
        kwargs["global_pool"] = ""
    model = timm.create_model(model_name, pretrained=True, num_classes=0, **kwargs)
    config = resolve_data_config(model.pretrained_cfg)
    transforms = create_transform(**config)
    return cls(model, device, transforms)


class ReIDModel:
    def __init__(
        self,
        backbone: nn.Module,
        device: Optional[str] = "auto",
        transforms: Optional[Union[Callable, list[Callable]]] = None,
    ):
        self.device = parse_device_spec(device or "auto")
        self.backbone = backbone.to(self.device)
        self._initialize_transforms(transforms)

    def _initialize_transforms(
        self, transforms: Optional[Union[Callable, list[Callable]]]
    ) -> None:
        if isinstance(transforms, list):
            self.train_transforms = Compose(transforms)
            self.inference_transforms = Compose([ToPILImage(), *transforms])
        else:
            self.train_transforms = Compose([transforms])
            self.inference_transforms = Compose([ToPILImage(), transforms])

    @classmethod
    def from_timm(
        cls,
        model_name_or_checkpoint_path: str,
        device: Optional[str] = "auto",
        get_pooled_features: bool = True,
        **kwargs,
    ) -> ReIDModel:
        return _initialize_reid_model_from_timm(
            cls,
            model_name_or_checkpoint_path,
            device,
            get_pooled_features,
            **kwargs,
        )

    def extract_features(
        self, detections: sv.Detections, frame: Union[np.ndarray, PIL.Image.Image]
    ) -> np.ndarray:
        """
        Extract features from detection crops in the frame.

        Args:
            detections (sv.Detections): Detections from which to extract features.
            frame (np.ndarray or PIL.Image.Image): The input frame.

        Returns:
            np.ndarray: Extracted features for each detection.
        """
        if len(detections) == 0:
            return np.array([])

        if isinstance(frame, PIL.Image.Image):
            frame = np.array(frame)

        features = []
        with torch.inference_mode():
            for box in detections.xyxy:
                crop = sv.crop_image(image=frame, xyxy=[*box.astype(int)])
                tensor = self.inference_transforms(crop).unsqueeze(0).to(self.device)
                feature = torch.squeeze(self.backbone(tensor)).cpu().numpy().flatten()
                features.append(feature)

        return np.array(features)
