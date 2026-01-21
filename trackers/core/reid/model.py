from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional, Union

import numpy as np
import PIL
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

from trackers.core.reid.callbacks import BaseCallback
from trackers.core.reid.metrics import (
    TripletAccuracyMetric,
    TripletMetric,
)
from trackers.log import get_logger
from trackers.utils.torch_utils import load_safetensors_checkpoint, parse_device_spec

logger = get_logger(__name__)


def _initialize_reid_model_from_timm(
    cls,
    model_name_or_checkpoint_path: str,
    device: Optional[str] = "auto",
    get_pooled_features: bool = True,
    **kwargs,
):
    if model_name_or_checkpoint_path not in timm.list_models(
        filter=model_name_or_checkpoint_path, pretrained=True
    ):
        probable_model_name_list = timm.list_models(
            f"*{model_name_or_checkpoint_path}*", pretrained=True
        )
        if len(probable_model_name_list) == 0:
            raise ValueError(
                f"Model {model_name_or_checkpoint_path} not found in timm. "
                + "Please check the model name and try again."
            )
        logger.warning(
            f"Model {model_name_or_checkpoint_path} not found in timm. "
            + f"Using {probable_model_name_list[0]} instead."
        )
        model_name_or_checkpoint_path = probable_model_name_list[0]
    if not get_pooled_features:
        kwargs["global_pool"] = ""
    model = timm.create_model(
        model_name_or_checkpoint_path, pretrained=True, num_classes=0, **kwargs
    )
    config = resolve_data_config(model.pretrained_cfg)
    transforms = create_transform(**config)
    model_metadata = {
        "model_name_or_checkpoint_path": model_name_or_checkpoint_path,
        "get_pooled_features": get_pooled_features,
        "kwargs": kwargs,
    }
    return cls(model, device, transforms, model_metadata)


def _initialize_reid_model_from_checkpoint(cls, checkpoint_path: str, config_path:str):
    state_dict, config = load_safetensors_checkpoint(checkpoint_path, config_path)
    model_name = config.get("architecture")
    init_kwargs = {}
    init_kwargs["pretrained"] = False
    reid_model_instance = _initialize_reid_model_from_timm(
        cls, 
        model_name_or_checkpoint_path=model_name, 
        **init_kwargs
    )
    if config.get("projection_dimension"):
        reid_model_instance._add_projection_layer(
            projection_dimension=config.get("projection_dimension")
        )
    for k, v in state_dict.items():
        state_dict[k] = v.to(reid_model_instance.device)
    reid_model_instance.backbone_model.load_state_dict(state_dict,strict=False)
    return reid_model_instance


class ReIDModel:
    """
    A ReID model that is used to extract features from detection crops for trackers
    that utilize appearance features.

    Args:
        backbone_model (nn.Module): The torch model to use as the backbone.
        device (Optional[str]): The device to run the model on.
        transforms (Optional[Union[Callable, list[Callable]]]): The transforms to
            apply to the input images.
        model_metadata (dict[str, Any]): Metadata about the model architecture.
    """

    def __init__(
        self,
        backbone_model: nn.Module,
        device: Optional[str] = "auto",
        transforms: Optional[Union[Callable, list[Callable]]] = None,
        model_metadata: dict[str, Any] = {},
    ):
        self.backbone_model = backbone_model
        self.device = parse_device_spec(device or "auto")
        self.backbone_model.to(self.device)
        self.backbone_model.eval()
        self.train_transforms = (
            (Compose(*transforms) if isinstance(transforms, list) else transforms)
            if transforms is not None
            else None
        )
        self.inference_transforms = Compose(
            [ToPILImage(), *transforms]
            if isinstance(transforms, list)
            else [ToPILImage(), transforms]
        )
        self.model_metadata = model_metadata

    @classmethod
    def from_timm(
        cls,
        model_name_or_checkpoint_path: str,
        config_path:Optional[str] = None,
        device: Optional[str] = "auto",
        get_pooled_features: bool = True,
        **kwargs,
    ) -> ReIDModel:
        """
        Create a `ReIDModel` with a [timm](https://huggingface.co/docs/timm)
        model as the backbone.

        Args:
            model_name_or_checkpoint_path (str): Name of the timm model to use or
                path to a safetensors checkpoint. If the exact model name is not
                found, the closest match from `timm.list_models` will be used.
            config_path (str): Path to the config file for the local safetensors checkpoint.
            device (str): Device to run the model on.
            get_pooled_features (bool): Whether to get the pooled features from the
                model or not.
            **kwargs: Additional keyword arguments to pass to
                [`timm.create_model`](https://huggingface.co/docs/timm/en/reference/models#timm.create_model).

        Returns:
            ReIDModel: A new instance of `ReIDModel`.
        """
        if os.path.exists(model_name_or_checkpoint_path) and os.path.exists(config_path):
            return _initialize_reid_model_from_checkpoint(
                cls, 
                model_name_or_checkpoint_path, 
                config_path
            )
        else:
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
                feature = (
                    torch.squeeze(self.backbone_model(tensor)).cpu().numpy().flatten()
                )
                features.append(feature)

        return np.array(features)

    def _add_projection_layer(
        self, projection_dimension: Optional[int] = None, freeze_backbone: bool = False
    ):
        """
        Perform model surgery to add a projection layer to the model and freeze the
        backbone if specified. The backbone is only frozen if `projection_dimension`
        is specified.

        Args:
            projection_dimension (Optional[int]): The dimension of the projection layer.
            freeze_backbone (bool): Whether to freeze the backbone of the model during
                training.
        """
        if projection_dimension is not None:
            # Freeze backbone only if specified and projection_dimension is mentioned
            if freeze_backbone:
                for param in self.backbone_model.parameters():
                    param.requires_grad = False

            # Add projection layer if projection_dimension is specified
            self.backbone_model = nn.Sequential(
                self.backbone_model,
                nn.Linear(self.backbone_model.num_features, projection_dimension),
            )
            self.backbone_model.to(self.device)

    def _train_step(
        self,
        anchor_image: torch.Tensor,
        positive_image: torch.Tensor,
        negative_image: torch.Tensor,
        metrics_list: list[TripletMetric],
    ) -> dict[str, float]:
        """
        Perform a single training step.

        Args:
            anchor_image (torch.Tensor): The anchor image.
            positive_image (torch.Tensor): The positive image.
            negative_image (torch.Tensor): The negative image.
            metrics_list (list[Metric]): The list of metrics to update.
        """
        self.optimizer.zero_grad()
        anchor_image_features = self.backbone_model(anchor_image)
        positive_image_features = self.backbone_model(positive_image)
        negative_image_features = self.backbone_model(negative_image)

        loss = self.criterion(
            anchor_image_features,
            positive_image_features,
            negative_image_features,
        )
        loss.backward()
        self.optimizer.step()

        # Update metrics
        for metric in metrics_list:
            metric.update(
                anchor_image_features.detach(),
                positive_image_features.detach(),
                negative_image_features.detach(),
            )

        train_logs = {"train/loss": loss.item()}
        for metric in metrics_list:
            train_logs[f"train/{metric!s}"] = metric.compute()

        return train_logs

    def _validation_step(
        self,
        anchor_image: torch.Tensor,
        positive_image: torch.Tensor,
        negative_image: torch.Tensor,
        metrics_list: list[TripletMetric],
    ) -> dict[str, float]:
        """
        Perform a single validation step.

        Args:
            anchor_image (torch.Tensor): The anchor image.
            positive_image (torch.Tensor): The positive image.
            negative_image (torch.Tensor): The negative image.
            metrics_list (list[Metric]): The list of metrics to update.
        """
        with torch.inference_mode():
            anchor_image_features = self.backbone_model(anchor_image)
            positive_image_features = self.backbone_model(positive_image)
            negative_image_features = self.backbone_model(negative_image)

            loss = self.criterion(
                anchor_image_features,
                positive_image_features,
                negative_image_features,
            )

            # Update metrics
            for metric in metrics_list:
                metric.update(
                    anchor_image_features.detach(),
                    positive_image_features.detach(),
                    negative_image_features.detach(),
                )

        validation_logs = {"validation/loss": loss.item()}
        for metric in metrics_list:
            validation_logs[f"validation/{metric!s}"] = metric.compute()

        return validation_logs

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        validation_loader: Optional[DataLoader] = None,
        projection_dimension: Optional[int] = None,
        freeze_backbone: bool = False,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        triplet_margin: float = 1.0,
        random_state: Optional[Union[int, float, str, bytes, bytearray]] = None,
        checkpoint_interval: Optional[int] = None,
        log_dir: str = "logs",
        log_to_matplotlib: bool = False,
        log_to_tensorboard: bool = False,
        log_to_wandb: bool = False,
    ) -> None:
        """
        Train/fine-tune the ReID model.

        Args:
            train_loader (DataLoader): The training data loader.
            epochs (int): The number of epochs to train the model.
            validation_loader (Optional[DataLoader]): The validation data loader.
            projection_dimension (Optional[int]): The dimension of the projection layer.
            freeze_backbone (bool): Whether to freeze the backbone of the model. The
                backbone is only frozen if `projection_dimension` is specified.
            learning_rate (float): The learning rate to use for the optimizer.
            weight_decay (float): The weight decay to use for the optimizer.
            triplet_margin (float): The margin to use for the triplet loss.
            random_state (Optional[Union[int, float, str, bytes, bytearray]]): The
                random state to use for the training.
            checkpoint_interval (Optional[int]): The interval to save checkpoints.
            log_dir (str): The directory to save logs.
            log_to_matplotlib (bool): Whether to log to matplotlib.
            log_to_tensorboard (bool): Whether to log to tensorboard.
            log_to_wandb (bool): Whether to log to wandb. If `checkpoint_interval` is
                specified, the model will be logged to wandb as well.
                Project and entity name should be set using the environment variables
                `WANDB_PROJECT` and `WANDB_ENTITY`. For more details, refer to
                [wandb environment variables](https://docs.wandb.ai/guides/track/environment-variables).
        """
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "tensorboard_logs"), exist_ok=True)

        if random_state is not None:
            torch.manual_seed(random_state)

        self._add_projection_layer(projection_dimension, freeze_backbone)

        # Initialize optimizer, criterion and metrics
        self.optimizer = optim.Adam(
            self.backbone_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.TripletMarginLoss(margin=triplet_margin)
        metrics_list: list[TripletMetric] = [TripletAccuracyMetric()]

        config = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "random_state": random_state,
            "projection_dimension": projection_dimension,
            "freeze_backbone": freeze_backbone,
            "triplet_margin": triplet_margin,
            "model_metadata": self.model_metadata,
        }

        # Initialize callbacks
        callbacks: list[BaseCallback] = []
        if log_to_matplotlib:
            try:
                from trackers.core.reid.callbacks import MatplotlibCallback

                callbacks.append(MatplotlibCallback(log_dir=log_dir))
            except (ImportError, AttributeError) as e:
                logger.error(
                    "Metric logging dependencies are not installed. "
                    "Please install it using `pip install trackers[metrics]`.",
                )
                raise e
        if log_to_tensorboard:
            try:
                from trackers.core.reid.callbacks import TensorboardCallback

                callbacks.append(
                    TensorboardCallback(
                        log_dir=os.path.join(log_dir, "tensorboard_logs")
                    )
                )
            except (ImportError, AttributeError) as e:
                logger.error(
                    "Metric logging dependencies are not installed. "
                    "Please install it using `pip install trackers[metrics]`."
                )
                raise e

        if log_to_wandb:
            try:
                from trackers.core.reid.callbacks import WandbCallback

                callbacks.append(WandbCallback(config=config))
            except (ImportError, AttributeError) as e:
                logger.error(
                    "Metric logging dependencies are not installed. "
                    "Please install it using `pip install trackers[metrics]`."
                )
                raise e

        # Training loop over epochs
        for epoch in tqdm(range(epochs), desc="Training"):
            # Reset metrics at the start of each epoch
            for metric in metrics_list:
                metric.reset()

            # Training loop over batches
            accumulated_train_logs: dict[str, Union[float, int]] = {}
            for idx, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Training Epoch {epoch + 1}/{epochs}",
                leave=False,
            ):
                anchor_image, positive_image, negative_image = data
                if self.train_transforms is not None:
                    anchor_image = self.train_transforms(anchor_image)
                    positive_image = self.train_transforms(positive_image)
                    negative_image = self.train_transforms(negative_image)

                anchor_image = anchor_image.to(self.device)
                positive_image = positive_image.to(self.device)
                negative_image = negative_image.to(self.device)

                if callbacks:
                    for callback in callbacks:
                        callback.on_train_batch_start(
                            {}, epoch * len(train_loader) + idx
                        )

                train_logs = self._train_step(
                    anchor_image, positive_image, negative_image, metrics_list
                )

                for key, value in train_logs.items():
                    accumulated_train_logs[key] = (
                        accumulated_train_logs.get(key, 0) + value
                    )

                if callbacks:
                    for callback in callbacks:
                        for key, value in train_logs.items():
                            callback.on_train_batch_end(
                                {f"batch/{key}": value}, epoch * len(train_loader) + idx
                            )

            for key, value in accumulated_train_logs.items():
                accumulated_train_logs[key] = value / len(train_loader)

            # Compute and add training metrics to logs
            for metric in metrics_list:
                accumulated_train_logs[f"train/{metric!s}"] = metric.compute()
            # Metrics are reset at the start of the next epoch or before validation

            if callbacks:
                for callback in callbacks:
                    callback.on_train_epoch_end(accumulated_train_logs, epoch)

            # Validation loop over batches
            accumulated_validation_logs: dict[str, Union[float, int]] = {}
            if validation_loader is not None:
                # Reset metrics for validation
                for metric in metrics_list:
                    metric.reset()
                for idx, data in tqdm(
                    enumerate(validation_loader),
                    total=len(validation_loader),
                    desc=f"Validation Epoch {epoch + 1}/{epochs}",
                    leave=False,
                ):
                    if callbacks:
                        for callback in callbacks:
                            callback.on_validation_batch_start(
                                {}, epoch * len(train_loader) + idx
                            )

                    anchor_image, positive_image, negative_image = data
                    if self.train_transforms is not None:
                        anchor_image = self.train_transforms(anchor_image)
                        positive_image = self.train_transforms(positive_image)
                        negative_image = self.train_transforms(negative_image)

                    anchor_image = anchor_image.to(self.device)
                    positive_image = positive_image.to(self.device)
                    negative_image = negative_image.to(self.device)

                    validation_logs = self._validation_step(
                        anchor_image, positive_image, negative_image, metrics_list
                    )

                    for key, value in validation_logs.items():
                        accumulated_validation_logs[key] = (
                            accumulated_validation_logs.get(key, 0) + value
                        )

                    if callbacks:
                        for callback in callbacks:
                            for key, value in validation_logs.items():
                                callback.on_validation_batch_end(
                                    {f"batch/{key}": value},
                                    epoch * len(train_loader) + idx,
                                )

                for key, value in accumulated_validation_logs.items():
                    accumulated_validation_logs[key] = value / len(validation_loader)

                # Compute and add validation metrics to logs
                for metric in metrics_list:
                    accumulated_validation_logs[f"validation/{metric!s}"] = (
                        metric.compute()
                    )
                # Metrics will be reset at the start of the next training epoch loop

            if callbacks:
                for callback in callbacks:
                    callback.on_validation_epoch_end(accumulated_validation_logs, epoch)

            # Save checkpoint
            if (
                checkpoint_interval is not None
                and (epoch + 1) % checkpoint_interval == 0
            ):
                state_dict = self.backbone_model.state_dict()
                checkpoint_path = os.path.join(
                    log_dir, "checkpoints", f"reid_model_{epoch + 1}.safetensors"
                )
                save_file(
                    state_dict,
                    checkpoint_path,
                    metadata={"config": json.dumps(config), "format": "pt"},
                )
                if callbacks:
                    for callback in callbacks:
                        callback.on_checkpoint_save(checkpoint_path, epoch + 1)

        if callbacks:
            for callback in callbacks:
                callback.on_end()
