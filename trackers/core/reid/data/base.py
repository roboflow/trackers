from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


class TripletsDataset(Dataset, ABC):
    """
    A base class for datasets that contains triplets of images.

    Args:
        data_dir (str): The directory containing the dataset.
        transforms (Optional[Compose]): The transforms to apply to the images.
    """

    def __init__(self, data_dir: str, transforms: Optional[Compose] = None):
        self.data_dir = data_dir
        self.transforms = transforms or ToTensor()
        self.triplet_classes = self.get_triplet_classes()

    @abstractmethod
    def get_triplet_classes(self):
        pass

    @abstractmethod
    def get_anchor_image_file(self, triplet_class: str) -> str:
        pass

    @abstractmethod
    def get_positive_image_file(self, triplet_class: str, anchor_image_file) -> str:
        pass

    @abstractmethod
    def get_negative_image_file(self, triplet_class: str) -> str:
        pass

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        triplet_class = self.triplet_classes[index]
        anchor_image_file = self.get_anchor_image_file(triplet_class)
        positive_image_file = self.get_positive_image_file(
            triplet_class, anchor_image_file
        )
        negative_image_file = self.get_negative_image_file(triplet_class)

        anchor_image = Image.open(anchor_image_file).convert("RGB")
        positive_image = Image.open(positive_image_file).convert("RGB")
        negative_image = Image.open(negative_image_file).convert("RGB")

        if self.transforms:
            anchor_image = self.transforms(anchor_image)
            positive_image = self.transforms(positive_image)
            negative_image = self.transforms(negative_image)
        return anchor_image, positive_image, negative_image
