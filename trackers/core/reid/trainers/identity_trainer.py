from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from trackers.core.reid.trainers.callbacks import BaseCallback
from trackers.core.reid.trainers.commons import save_checkpoint
from trackers.utils.torch_utils import parse_device_spec


class IdentityTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_transforms: Optional[Callable] = None,
        device: Optional[Union[str, torch.device]] = "auto",
        label_smoothing: Optional[float] = 0.1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = parse_device_spec(device) if isinstance(device, str) else device
        self.train_transforms = train_transforms
        self.label_smoothing = label_smoothing

    def train_step(self, image_batch: torch.Tensor, entity_id_batch: torch.Tensor):
        outputs = self.model(image_batch)
        logits = F.log_softmax(outputs, dim=1)
        loss = F.cross_entropy(
            logits, entity_id_batch, label_smoothing=self.label_smoothing
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_logs = {"train/loss": loss.item()}
        return train_logs

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        validation_loader: Optional[DataLoader] = None,
        callbacks: list[BaseCallback] = [],
        checkpoint_interval: Optional[int] = None,
        log_dir: str = "logs",
        config: dict[str, Any] = {},
    ):
        for epoch in tqdm(range(epochs), desc="Training"):
            accumulated_train_logs: dict[str, Union[float, int]] = {}
            for idx, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Training Epoch {epoch + 1}/{epochs}",
                leave=False,
            ):
                image_batch = data["image"]
                entity_id_batch = data["entity_id"]
                if self.train_transforms is not None:
                    image_batch = self.train_transforms(image_batch)

                image_batch = image_batch.to(self.device)
                entity_id_batch = entity_id_batch.to(self.device)

                if callbacks:
                    for callback in callbacks:
                        callback.on_train_batch_start(
                            {}, epoch * len(train_loader) + idx
                        )

                train_logs = self.train_step(image_batch, entity_id_batch)

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

            if callbacks:
                for callback in callbacks:
                    callback.on_train_epoch_end(accumulated_train_logs, epoch)

            save_checkpoint(
                model=self.model,
                epoch=epoch,
                checkpoint_interval=checkpoint_interval,
                log_dir=log_dir,
                config=config,
                callbacks=callbacks,
            )
