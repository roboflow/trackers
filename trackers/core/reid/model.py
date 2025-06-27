from __future__ import annotations

import os
from typing import Any, Callable, Optional, Union

import numpy as np
import PIL
import supervision as sv
import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage

from trackers.core.reid.dataset.base import IdentityDataset
from trackers.core.reid.trainer.cross_entropy_trainer import CrossEntropyTrainer
from trackers.log import get_logger
from trackers.utils.torch_utils import load_safetensors_checkpoint, parse_device_spec

logger = get_logger(__name__)


class FeatureExtractorModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.fc: Optional[nn.Linear] = None

    def add_classification_head(self, num_classes: int) -> None:
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        pooled_features = self.backbone.global_pool(features)
        return pooled_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled_features = self.forward_features(x)
        output = self.fc(pooled_features) if self.fc is not None else pooled_features
        return output


class ReIDModel:
    def __init__(
        self,
        backbone: nn.Module,
        device: Optional[str] = "auto",
        transforms: Optional[Union[Callable, list[Callable], Compose]] = None,
        model_metadata: dict[str, Any] = {},
    ):
        self.device = parse_device_spec(device or "auto")
        self.feature_extractor = (
            FeatureExtractorModel(backbone)
            if not isinstance(backbone, FeatureExtractorModel)
            else backbone
        ).to(self.device)
        self._initialize_transforms(transforms)
        self.model_metadata = model_metadata

    def _initialize_transforms(
        self, transforms: Optional[Union[Callable, list[Callable], Compose]]
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
        **kwargs,
    ) -> ReIDModel:
        if not os.path.exists(model_name_or_checkpoint_path):
            model = timm.create_model(
                model_name_or_checkpoint_path, pretrained=True, num_classes=0, **kwargs
            )
            config = resolve_data_config(model.pretrained_cfg)
            transforms = create_transform(**config)
            return cls(model, device, transforms)
        else:
            state_dict, config = load_safetensors_checkpoint(
                model_name_or_checkpoint_path
            )
            model = timm.create_model(
                model_name_or_checkpoint_path, pretrained=True, num_classes=0, **kwargs
            )
            config = resolve_data_config(model.pretrained_cfg)
            transforms = create_transform(**config)
            reid_model_instance = cls(model, device, transforms)
            if config["num_classes"]:
                reid_model_instance.add_classification_head(
                    num_classes=config["num_classes"]
                )
            for k, _ in state_dict.items():
                state_dict[k].to(reid_model_instance.device)
            reid_model_instance.feature_extractor.load_state_dict(state_dict)
            return reid_model_instance

    def add_classification_head(
        self, num_classes: int, freeze_backbone: bool = False
    ) -> None:
        if freeze_backbone:
            for param in self.feature_extractor.backbone.parameters():
                param.requires_grad = False
        self.feature_extractor.add_classification_head(num_classes)

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

        features_list = []
        with torch.inference_mode():
            for box in detections.xyxy:
                crop = sv.crop_image(image=frame, xyxy=[*box.astype(int)])
                crop_tensor = (
                    self.inference_transforms(crop).unsqueeze(0).to(self.device)
                )
                pooled_features = self.feature_extractor.forward_features(crop_tensor)
                pooled_features = torch.squeeze(pooled_features).cpu().numpy().flatten()
                features_list.append(pooled_features)

        return np.array(features_list)

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        num_classes: int,
        validation_loader: Optional[DataLoader] = None,
        freeze_backbone: bool = False,
        label_smoothing: float = 1e-2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        random_state: Optional[Union[int, float, str, bytes, bytearray]] = None,
        checkpoint_interval: Optional[int] = None,
        log_dir: str = "logs",
        log_to_matplotlib: bool = False,
        log_to_tensorboard: bool = False,
        log_to_wandb: bool = False,
    ):
        if isinstance(train_loader.dataset, IdentityDataset):
            if validation_loader is not None:
                assert isinstance(validation_loader.dataset, IdentityDataset)
            self.add_classification_head(num_classes, freeze_backbone=freeze_backbone)
            trainer = CrossEntropyTrainer(
                model=self.feature_extractor,
                device=self.device,
                transforms=self.train_transforms,
                epochs=epochs,
                label_smoothing=label_smoothing,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                random_state=random_state,
                log_dir=log_dir,
                log_to_matplotlib=log_to_matplotlib,
                log_to_tensorboard=log_to_tensorboard,
                log_to_wandb=log_to_wandb,
            )
            trainer.train(
                train_loader=train_loader,
                validation_loader=validation_loader,
                checkpoint_interval=checkpoint_interval,
            )
            self.feature_extractor = trainer.model
