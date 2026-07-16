# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID fine-tuning losses (label-smoothed CE, batch-hard triplet, optional center)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(self, num_classes: int, epsilon: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= epsilon < 1.0:
            raise ValueError(f"epsilon must be in [0, 1), got {epsilon}.")
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute smoothed cross-entropy."""
        log_probs = F.log_softmax(logits, dim=1)
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        return (-targets * log_probs).sum(dim=1).mean()


def _pairwise_euclidean(embeddings: torch.Tensor) -> torch.Tensor:
    """Return the ``(B, B)`` pairwise Euclidean distance matrix."""
    distances = torch.cdist(embeddings, embeddings, p=2)
    return distances.clamp(min=1e-12)


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss (hardest positive/negative per anchor).

    For each anchor, the hardest positive is the farthest same-identity sample
    and the hardest negative is the nearest different-identity sample.
    """

    def __init__(self, margin: float = 0.3, normalize: bool = True) -> None:
        super().__init__()
        self.margin = margin
        self.normalize = normalize

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute batch-hard triplet loss."""
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        distances = _pairwise_euclidean(embeddings)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        different = ~same
        positive_mask = same.clone()
        positive_mask.fill_diagonal_(False)

        hardest_positive = (distances * positive_mask).max(dim=1).values
        masked = distances.clone()
        masked[~different] = float("inf")
        hardest_negative = masked.min(dim=1).values

        return F.relu(self.margin + hardest_positive - hardest_negative).mean()


class CenterLoss(nn.Module):
    """Center loss: pull embeddings toward learnable class centers."""

    def __init__(self, num_classes: int, feat_dim: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute center loss."""
        centers_batch = self.centers.index_select(0, labels)
        return (embeddings - centers_batch).pow(2).sum(dim=1).clamp(min=1e-12).mean() / 2.0


class ReIDLoss(nn.Module):
    """Combined CE + triplet (+ optional center) ReID objective."""

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        *,
        margin: float = 0.3,
        label_smoothing: float = 0.1,
        triplet_weight: float = 1.0,
        center_weight: float = 0.0005,
        use_center: bool = False,
        normalize_triplet: bool = True,
    ) -> None:
        super().__init__()
        self.cross_entropy = CrossEntropyLabelSmooth(num_classes, label_smoothing)
        self.triplet = BatchHardTripletLoss(margin=margin, normalize=normalize_triplet)
        self.triplet_weight = triplet_weight
        self.use_center = use_center
        self.center_weight = center_weight if use_center else 0.0
        self.center = CenterLoss(num_classes, feat_dim) if use_center else None

    def forward(
        self,
        embeddings: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(total_loss, component_dict)``."""
        ce = self.cross_entropy(logits, labels)
        triplet = self.triplet(embeddings, labels)
        total = ce + self.triplet_weight * triplet
        components = {"ce": ce.item(), "triplet": triplet.item()}

        if self.center is not None:
            center = self.center(embeddings, labels)
            total = total + self.center_weight * center
            components["center"] = center.item()

        components["total"] = total.item()
        return total, components
