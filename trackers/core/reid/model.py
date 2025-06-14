from __future__ import annotations

import os
from typing import Any, Callable, Optional, Union

import numpy as np
import PIL
import supervision as sv
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage

from trackers.core.reid.dataset.base import TripletsDataset
from trackers.core.reid.trainers.callbacks import BaseCallback
from trackers.core.reid.trainers.metrics import (
    TripletAccuracyMetric,
    TripletMetric,
)
from trackers.core.reid.trainers.triplets_trainer import TripletsTrainer
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


def _initialize_reid_model_from_checkpoint(cls, checkpoint_path: str):
    state_dict, config = load_safetensors_checkpoint(checkpoint_path)
    reid_model_instance = _initialize_reid_model_from_timm(
        cls, **config["model_metadata"]
    )
    if config["projection_dimension"]:
        reid_model_instance._add_projection_layer(
            projection_dimension=config["projection_dimension"]
        )
    for k, v in state_dict.items():
        state_dict[k].to(reid_model_instance.device)
    reid_model_instance.backbone_model.load_state_dict(state_dict)
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
            device (str): Device to run the model on.
            get_pooled_features (bool): Whether to get the pooled features from the
                model or not.
            **kwargs: Additional keyword arguments to pass to
                [`timm.create_model`](https://huggingface.co/docs/timm/en/reference/models#timm.create_model).

        Returns:
            ReIDModel: A new instance of `ReIDModel`.
        """
        if os.path.exists(model_name_or_checkpoint_path):
            return _initialize_reid_model_from_checkpoint(
                cls, model_name_or_checkpoint_path
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

    def _initialize_callbacks(
        self,
        log_to_matplotlib: bool,
        log_to_tensorboard: bool,
        log_to_wandb: bool,
        log_dir: str,
        config: dict[str, Any],
    ) -> list[BaseCallback]:
        callbacks: list[BaseCallback] = []
        if log_to_matplotlib:
            try:
                from trackers.core.reid.trainers.callbacks import MatplotlibCallback

                callbacks.append(MatplotlibCallback(log_dir=log_dir))
            except (ImportError, AttributeError) as e:
                logger.error(
                    "Metric logging dependencies are not installed. "
                    "Please install it using `pip install trackers[metrics]`.",
                )
                raise e
        if log_to_tensorboard:
            try:
                from trackers.core.reid.trainers.callbacks import TensorboardCallback

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
                from trackers.core.reid.trainers.callbacks import WandbCallback

                callbacks.append(WandbCallback(config=config))
            except (ImportError, AttributeError) as e:
                logger.error(
                    "Metric logging dependencies are not installed. "
                    "Please install it using `pip install trackers[metrics]`."
                )
                raise e

        return callbacks

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
                This is only used if the dataset is a `TripletsDataset`.
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
        metrics_list: list[TripletMetric] = [TripletAccuracyMetric()]

        config = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "random_state": random_state,
            "projection_dimension": projection_dimension,
            "freeze_backbone": freeze_backbone,
            "model_metadata": self.model_metadata,
        }

        # Initialize callbacks
        callbacks = self._initialize_callbacks(
            log_to_matplotlib=log_to_matplotlib,
            log_to_tensorboard=log_to_tensorboard,
            log_to_wandb=log_to_wandb,
            log_dir=log_dir,
            config=config,
        )

        if isinstance(train_loader.dataset, TripletsDataset):
            config["triplet_margin"] = triplet_margin
            trainer = TripletsTrainer(
                model=self.backbone_model,
                optimizer=self.optimizer,
                criterion=nn.TripletMarginLoss(margin=triplet_margin),
                train_transforms=self.train_transforms,
                device=self.device,
            )
            trainer.train(
                train_loader=train_loader,
                epochs=epochs,
                validation_loader=validation_loader,
                metrics_list=metrics_list,
                callbacks=callbacks,
                checkpoint_interval=checkpoint_interval,
                log_dir=log_dir,
                config=config,
            )

        if callbacks:
            for callback in callbacks:
                callback.on_end()
