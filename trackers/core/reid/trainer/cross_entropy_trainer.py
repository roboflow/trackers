from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm.auto import tqdm


class CrossEntropyTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        transforms: Optional[Union[Callable, list[Callable], Compose]] = None,
        label_smoothing: float = 1e-2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        self.device = device
        self.model = model.to(device)
        self.transforms = transforms
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def train_step(self, data: dict[str, Union[torch.Tensor, str, int]]):
        images = self.transforms(data["image"]).to(self.device)
        identities = data["identity"].to(self.device)
        outputs = self.model(images)
        loss = self.criterion(F.log_softmax(outputs, dim=1), identities)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "Train/loss": loss.item(),
        }

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
    ):
        for epoch in tqdm(range(epochs), desc="Training"):
            for idx, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Training Epoch {epoch + 1}/{epochs}",
                leave=False,
            ):
                _ = self.train_step(data)
