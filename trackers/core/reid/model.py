import os
from enum import Enum
from typing import Callable, Optional

import numpy as np
import supervision as sv
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_file
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage
from tqdm.auto import tqdm

from trackers.core.reid.callbacks import TensorboardCallback
from trackers.utils.torch_utils import parse_device_spec


class ReIDOptimizer(str, Enum):
    ADAM = "adam"
    SGD = "sgd"


class ReIDModel:
    def __init__(
        self,
        backbone_model: nn.Module,
        device: Optional[str] = "auto",
        transforms: Optional[Callable] = None,
    ):
        self.backbone_model = backbone_model
        self.device = parse_device_spec(device or "auto")
        self.backbone_model.to(self.device)
        self.backbone_model.eval()
        self.inference_transforms = Compose([ToPILImage(), transforms])

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
        config = resolve_data_config(model.pretrained_cfg)
        transforms = create_transform(**config)
        return cls(model, device, transforms)

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
                tensor = self.inference_transforms(crop).unsqueeze(0).to(self.device)
                feature = (
                    torch.squeeze(self.backbone_model(tensor)).cpu().numpy().flatten()
                )
                features.append(feature)

        return np.array(features)

    def compile_for_training(
        self,
        optimizer: ReIDOptimizer = ReIDOptimizer.ADAM,
        learning_rate: float = 1e-4,
        **kwargs,
    ) -> None:
        if optimizer == ReIDOptimizer.ADAM:
            self.optimizer = optim.Adam(
                self.backbone_model.parameters(), lr=learning_rate, **kwargs
            )
        elif optimizer == ReIDOptimizer.SGD:
            self.optimizer = optim.SGD(
                self.backbone_model.parameters(), lr=learning_rate, **kwargs
            )
        self.criterion = nn.TripletMarginLoss(margin=1.0)

    def train_step(
        self,
        anchor_image: torch.Tensor,
        positive_image: torch.Tensor,
        negative_image: torch.Tensor,
    ) -> torch.Tensor:
        self.optimizer.zero_grad()

        anchor_image_features = self.backbone_model(anchor_image.to(self.device))
        positive_image_features = self.backbone_model(positive_image.to(self.device))
        negative_image_features = self.backbone_model(negative_image.to(self.device))

        loss = self.criterion(
            anchor_image_features,
            positive_image_features,
            negative_image_features,
        )
        loss.backward()
        self.optimizer.step()

        return {"train/loss": loss.item()}

    def validation_step(
        self,
        anchor_image: torch.Tensor,
        positive_image: torch.Tensor,
        negative_image: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            anchor_image_features = self.backbone_model(anchor_image.to(self.device))
            positive_image_features = self.backbone_model(
                positive_image.to(self.device)
            )
            negative_image_features = self.backbone_model(
                negative_image.to(self.device)
            )

        loss = self.criterion(
            anchor_image_features,
            positive_image_features,
            negative_image_features,
        )

        return {"validation/loss": loss.item()}

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        validation_loader: Optional[DataLoader] = None,
        checkpoint_interval: Optional[int] = None,
        checkpoint_dir: str = "checkpoints",
        log_to_tensorboard: bool = False,
    ) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)

        metric_logger_callback = []
        if log_to_tensorboard:
            metric_logger_callback.append(TensorboardCallback())

        for epoch in tqdm(range(epochs), desc="Training"):
            for idx, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Training Epoch {epoch + 1}/{epochs}",
                leave=False,
            ):
                anchor_image, positive_image, negative_image = data
                train_logs = self.train_step(
                    anchor_image, positive_image, negative_image
                )

                if metric_logger_callback:
                    for callback in metric_logger_callback:
                        callback.on_train_end(
                            train_logs, epoch * len(train_loader) + idx
                        )

            if validation_loader is not None:
                for idx, data in tqdm(
                    enumerate(validation_loader),
                    total=len(validation_loader),
                    desc=f"Validation Epoch {epoch + 1}/{epochs}",
                    leave=False,
                ):
                    anchor_image, positive_image, negative_image = data
                    validation_logs = self.validation_step(
                        anchor_image, positive_image, negative_image
                    )

                    if metric_logger_callback:
                        for callback in metric_logger_callback:
                            callback.on_validation_end(
                                validation_logs, epoch * len(train_loader) + idx
                            )

            if (
                checkpoint_interval is not None
                and (epoch + 1) % checkpoint_interval == 0
            ):
                state_dict = self.backbone_model.state_dict()
                save_file(
                    state_dict,
                    os.path.join(checkpoint_dir, f"reid_model_{epoch + 1}.safetensors"),
                )

        if metric_logger_callback:
            for callback in metric_logger_callback:
                callback.on_end()
