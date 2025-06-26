from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from trackers.core.reid.trainer.base import BaseTrainer
from trackers.log import get_logger

logger = get_logger(__name__)


def top_k_accuracy(
    logits: torch.Tensor, true_labels: torch.Tensor, top_k: int = 1
) -> torch.Tensor:
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    top_k_predicted_indices = logits.topk(top_k, 1, True, True)[1].t()
    correct_matches = top_k_predicted_indices.eq(
        true_labels.view(1, -1).expand_as(top_k_predicted_indices)
    )
    num_correct_in_top_k = correct_matches[:top_k].view(-1).float().sum(0, keepdim=True)
    return num_correct_in_top_k.mul_(100.0 / true_labels.size(0))


class CrossEntropyTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        transforms: Compose,
        epochs: int,
        label_smoothing: float = 1e-2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        random_state: Optional[Union[int, float, str, bytes, bytearray]] = None,
        log_dir: str = "logs",
        log_to_matplotlib: bool = False,
        log_to_tensorboard: bool = False,
        log_to_wandb: bool = False,
    ):
        config = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "random_state": random_state,
            "label_smoothing": label_smoothing,
        }
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        super().__init__(
            model=model,
            device=device,
            transforms=transforms,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            random_state=random_state,
            log_dir=log_dir,
            log_to_matplotlib=log_to_matplotlib,
            log_to_tensorboard=log_to_tensorboard,
            log_to_wandb=log_to_wandb,
            config=config,
        )

    def train_step(self, data: dict[str, torch.Tensor]):
        images = self.transforms(data["image"]).to(self.device)
        identities = data["identity"].to(self.device)
        outputs = self.model(images)
        loss = self.criterion(F.log_softmax(outputs, dim=1), identities)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "train/loss": loss.item(),
            "train/accuracy": top_k_accuracy(outputs, identities).item(),
        }

    def validation_step(self, data: dict[str, torch.Tensor]):
        images = self.transforms(data["image"]).to(self.device)
        identities = data["identity"].to(self.device)
        with torch.inference_mode():
            outputs = self.model(images)
            loss = self.criterion(F.log_softmax(outputs, dim=1), identities)
        return {
            "validation/loss": loss.item(),
            "validation/accuracy": top_k_accuracy(outputs, identities).item(),
        }
