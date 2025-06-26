import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm.auto import tqdm

from trackers.core.reid.trainer.callbacks import BaseCallback
from trackers.log import get_logger

logger = get_logger(__name__)


class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        transforms: Compose,
        epochs: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        random_state: Optional[Union[int, float, str, bytes, bytearray]] = None,
        log_dir: str = "logs",
        log_to_matplotlib: bool = False,
        log_to_tensorboard: bool = False,
        log_to_wandb: bool = False,
        config: Optional[dict[str, Any]] = {},
    ):
        self.device = device
        self.model = model.to(device)
        self.transforms = transforms
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.log_dir = log_dir
        self.config = config

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "tensorboard_logs"), exist_ok=True)

        if random_state is not None:
            torch.manual_seed(random_state)

        self.initialize_callbacks(log_to_matplotlib, log_to_tensorboard, log_to_wandb)

    def initialize_callbacks(
        self,
        log_to_matplotlib: bool = False,
        log_to_tensorboard: bool = False,
        log_to_wandb: bool = False,
    ):
        self.callbacks: list[BaseCallback] = []
        if log_to_matplotlib:
            try:
                from trackers.core.reid.trainer.callbacks import MatplotlibCallback

                self.callbacks.append(MatplotlibCallback(log_dir=self.log_dir))
            except (ImportError, AttributeError) as e:
                logger.error(
                    "Metric logging dependencies are not installed. "
                    "Please install it using `pip install trackers[metrics]`.",
                )
                raise e
        if log_to_tensorboard:
            try:
                from trackers.core.reid.trainer.callbacks import TensorboardCallback

                self.callbacks.append(
                    TensorboardCallback(
                        log_dir=os.path.join(self.log_dir, "tensorboard_logs")
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
                from trackers.core.reid.trainer.callbacks import WandbCallback

                self.callbacks.append(WandbCallback(config=self.config))
            except (ImportError, AttributeError) as e:
                logger.error(
                    "Metric logging dependencies are not installed. "
                    "Please install it using `pip install trackers[metrics]`."
                )
                raise e

    @abstractmethod
    def train_step(self, data: dict[str, torch.Tensor]):
        raise NotImplementedError(
            "Subclasses of `BaseTrainer` must implement `train_step`"
        )

    @abstractmethod
    def validation_step(self, data: dict[str, torch.Tensor]):
        raise NotImplementedError(
            "Subclasses of `BaseTrainer` must implement `validation_step`"
        )

    def execute_train_batch_loop(self, train_loader: DataLoader, epoch: int):
        accumulated_train_logs: dict[str, Union[float, int]] = {}
        for idx, data in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Training Epoch {epoch + 1}/{self.epochs}",
            leave=False,
        ):
            train_logs = self.train_step(data)
            for key, value in train_logs.items():
                accumulated_train_logs[key] = accumulated_train_logs.get(key, 0) + value
            for callback in self.callbacks:
                for key, value in train_logs.items():
                    callback.on_train_batch_end(
                        {f"batch/{key}": value}, epoch * len(train_loader) + idx
                    )
        for key, value in accumulated_train_logs.items():
            accumulated_train_logs[key] = value / len(train_loader)

        for callback in self.callbacks:
            callback.on_train_epoch_end(accumulated_train_logs, epoch)

    def execute_validation_batch_loop(self, validation_loader: DataLoader, epoch: int):
        accumulated_validation_logs: dict[str, Union[float, int]] = {}
        for idx, data in tqdm(
            enumerate(validation_loader),
            total=len(validation_loader),
            desc=f"Validation Epoch {epoch + 1}/{self.epochs}",
            leave=False,
        ):
            validation_logs = self.validation_step(data)
            for key, value in validation_logs.items():
                accumulated_validation_logs[key] = (
                    accumulated_validation_logs.get(key, 0) + value
                )
            for callback in self.callbacks:
                for key, value in validation_logs.items():
                    callback.on_validation_batch_end(
                        {f"batch/{key}": value}, epoch * len(validation_loader) + idx
                    )
        for key, value in accumulated_validation_logs.items():
            accumulated_validation_logs[key] = value / len(validation_loader)

        for callback in self.callbacks:
            callback.on_validation_epoch_end(accumulated_validation_logs, epoch)

    def train(
        self, train_loader: DataLoader, validation_loader: Optional[DataLoader] = None
    ):
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Training"):
            self.execute_train_batch_loop(train_loader, epoch)
            if validation_loader is not None:
                self.execute_validation_batch_loop(validation_loader, epoch)
