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
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage

from trackers.core.reid.dataset.base import IdentityDataset
from trackers.core.reid.trainer.cross_entropy_trainer import CrossEntropyTrainer
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
        transforms: Optional[Union[Callable, list[Callable], Compose]] = None,
    ):
        self.device = parse_device_spec(device or "auto")
        self.backbone = backbone.to(self.device)
        self._initialize_transforms(transforms)

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

    def add_classification_head(
        self, num_classes: int, freeze_backbone: bool = True
    ) -> None:
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Linear(self.backbone.num_features, num_classes)
        self.backbone.to(self.device)

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
                model=self.backbone,
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
            self.backbone = trainer.model
