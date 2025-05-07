import random
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from trackers.utils.data_utils import validate_tracker_id_to_images


class TripletsDataset(Dataset):
    """A dataset that provides triplets of images for training ReID models.

    This dataset is designed for training models with triplet loss, where each sample
    consists of an anchor image, a positive image (same identity as anchor),
    and a negative image (different identity from anchor).

    Args:
        tracker_id_to_images (dict[str, list[str]]): Dictionary mapping tracker IDs
            to lists of image paths
        transforms (Optional[Compose]): Optional image transformations to apply

    Attributes:
        tracker_id_to_images (dict[str, list[str]]): Dictionary mapping tracker IDs
            to lists of image paths
        transforms (Optional[Compose]): Optional image transformations to apply
        tracker_ids (list[str]): List of all unique tracker IDs in the dataset
    """

    def __init__(
        self,
        tracker_id_to_images: dict[str, list[str]],
        transforms: Optional[Compose] = None,
    ):
        self.tracker_id_to_images = validate_tracker_id_to_images(tracker_id_to_images)
        self.transforms = transforms or ToTensor()
        self.tracker_ids = list(tracker_id_to_images.keys())

    def __len__(self) -> int:
        return len(self.tracker_ids)

    def _load_and_transform_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image

    def _get_triplet_image_paths(self, tracker_id: str) -> Tuple[str, str, str]:
        tracker_id_image_paths = self.tracker_id_to_images[tracker_id]

        anchor_image_path, positive_image_path = random.sample(  # nosec B311
            tracker_id_image_paths, 2
        )

        negative_candidates = [tid for tid in self.tracker_ids if tid != tracker_id]
        negative_tracker_id = random.choice(negative_candidates)  # nosec B311

        negative_image_path = random.choice(  # nosec B311
            self.tracker_id_to_images[negative_tracker_id]
        )

        return anchor_image_path, positive_image_path, negative_image_path

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tracker_id = self.tracker_ids[index]

        anchor_image_path, positive_image_path, negative_image_path = (
            self._get_triplet_image_paths(tracker_id)
        )

        anchor_image = self._load_and_transform_image(anchor_image_path)
        positive_image = self._load_and_transform_image(positive_image_path)
        negative_image = self._load_and_transform_image(negative_image_path)

        return anchor_image, positive_image, negative_image
