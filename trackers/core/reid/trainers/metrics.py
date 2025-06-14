from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class TripletMetric(ABC):
    @abstractmethod
    def update(
        self,
        anchor_embed: torch.Tensor,
        positive_embed: torch.Tensor,
        negative_embed: torch.Tensor,
    ) -> None:
        pass

    @abstractmethod
    def compute(self) -> float:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class TripletAccuracyMetric(TripletMetric):
    """
    Calculates the triplet accuracy using pairwise distance.
    Accuracy is defined as the proportion of triplets where the distance
    between the anchor and positive embedding is less than the distance
    between the anchor and negative embedding.
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __str__(self):
        return "triplet_accuracy"

    def update(
        self,
        anchor_embed: torch.Tensor,
        positive_embed: torch.Tensor,
        negative_embed: torch.Tensor,
    ) -> None:
        """
        Update the metric with a batch of embeddings.

        Args:
            anchor_embed (torch.Tensor): Embeddings of the anchor samples.
            positive_embed (torch.Tensor): Embeddings of the positive samples.
            negative_embed (torch.Tensor): Embeddings of the negative samples.
        """
        dist_ap = F.pairwise_distance(anchor_embed, positive_embed, p=2)
        dist_an = F.pairwise_distance(anchor_embed, negative_embed, p=2)
        self.correct += torch.sum(dist_ap < dist_an).item()
        self.total += anchor_embed.size(0)

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = 0
        self.total = 0
