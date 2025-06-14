from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from trackers.core.reid.trainers.callbacks import BaseCallback
from trackers.core.reid.trainers.commons import save_checkpoint
from trackers.core.reid.trainers.metrics import TripletMetric
from trackers.utils.torch_utils import parse_device_spec


class TripletsTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_transforms: Optional[Callable] = None,
        device: Optional[Union[str, torch.device]] = "auto",
        triplet_margin: Optional[float] = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = parse_device_spec(device) if isinstance(device, str) else device
        self.train_transforms = train_transforms
        self.triplet_margin = triplet_margin

    def train_step(
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
        anchor_image_features = self.model(anchor_image)
        positive_image_features = self.model(positive_image)
        negative_image_features = self.model(negative_image)

        loss = F.triplet_margin_loss(
            anchor_image_features,
            positive_image_features,
            negative_image_features,
            margin=self.triplet_margin,
        )
        self.optimizer.zero_grad()
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

    def validation_step(
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
            anchor_image_features = self.model(anchor_image)
            positive_image_features = self.model(positive_image)
            negative_image_features = self.model(negative_image)

            loss = F.triplet_margin_loss(
                anchor_image_features,
                positive_image_features,
                negative_image_features,
                margin=self.triplet_margin,
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

    def batch_train_loop(
        self,
        train_loader: DataLoader,
        epoch: int,
        epochs: int,
        callbacks: list[BaseCallback],
        metrics_list: Optional[list[TripletMetric]] = None,
    ):
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
                    callback.on_train_batch_start({}, epoch * len(train_loader) + idx)

            train_logs = self.train_step(
                anchor_image, positive_image, negative_image, metrics_list or []
            )

            for key, value in train_logs.items():
                accumulated_train_logs[key] = accumulated_train_logs.get(key, 0) + value

            if callbacks:
                for callback in callbacks:
                    for key, value in train_logs.items():
                        callback.on_train_batch_end(
                            {f"batch/{key}": value}, epoch * len(train_loader) + idx
                        )

        for key, value in accumulated_train_logs.items():
            accumulated_train_logs[key] = value / len(train_loader)

        # Compute and add training metrics to logs
        if metrics_list is not None:
            for metric in metrics_list:
                accumulated_train_logs[f"train/{metric!s}"] = metric.compute()
        # Metrics are reset at the start of the next epoch or before validation

        if callbacks:
            for callback in callbacks:
                callback.on_train_epoch_end(accumulated_train_logs, epoch)

    def batch_validation_loop(
        self,
        validation_loader: DataLoader,
        epoch: int,
        epochs: int,
        callbacks: list[BaseCallback],
        metrics_list: Optional[list[TripletMetric]] = None,
    ):
        accumulated_validation_logs: dict[str, Union[float, int]] = {}
        if validation_loader is not None:
            # Reset metrics for validation
            if metrics_list is not None:
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
                            {}, epoch * len(validation_loader) + idx
                        )

                anchor_image, positive_image, negative_image = data
                if self.train_transforms is not None:
                    anchor_image = self.train_transforms(anchor_image)
                    positive_image = self.train_transforms(positive_image)
                    negative_image = self.train_transforms(negative_image)

                anchor_image = anchor_image.to(self.device)
                positive_image = positive_image.to(self.device)
                negative_image = negative_image.to(self.device)

                validation_logs = self.validation_step(
                    anchor_image, positive_image, negative_image, metrics_list or []
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
                                epoch * len(validation_loader) + idx,
                            )

            for key, value in accumulated_validation_logs.items():
                accumulated_validation_logs[key] = value / len(validation_loader)

            # Compute and add validation metrics to logs
            if metrics_list is not None:
                for metric in metrics_list:
                    accumulated_validation_logs[f"validation/{metric!s}"] = (
                        metric.compute()
                    )
            # Metrics will be reset at the start of the next training epoch loop

        if callbacks:
            for callback in callbacks:
                callback.on_validation_epoch_end(accumulated_validation_logs, epoch)

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        validation_loader: Optional[DataLoader] = None,
        metrics_list: Optional[list[TripletMetric]] = None,
        callbacks: list[BaseCallback] = [],
        checkpoint_interval: Optional[int] = None,
        log_dir: str = "logs",
        config: dict[str, Any] = {},
    ):
        for epoch in tqdm(range(epochs), desc="Training"):
            # Reset metrics at the start of each epoch
            if metrics_list is not None:
                for metric in metrics_list:
                    metric.reset()

            # Training loop over batches
            self.batch_train_loop(
                train_loader=train_loader,
                epoch=epoch,
                epochs=epochs,
                callbacks=callbacks,
                metrics_list=metrics_list,
            )

            # Validation loop over batches
            self.batch_validation_loop(
                validation_loader=validation_loader,
                epoch=epoch,
                epochs=epochs,
                callbacks=callbacks,
                metrics_list=metrics_list,
            )

            save_checkpoint(
                model=self.model,
                epoch=epoch,
                checkpoint_interval=checkpoint_interval,
                log_dir=log_dir,
                config=config,
                callbacks=callbacks,
            )
