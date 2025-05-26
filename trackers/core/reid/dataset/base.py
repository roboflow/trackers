from __future__ import annotations

import random
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from PIL import Image
from supervision.dataset.utils import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from trackers.core.reid.dataset.utils import validate_tracker_id_to_images


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
        transforms (Union[Compose, ToTensor]): Image transformations to apply
        tracker_ids (list[str]): List of all unique tracker IDs in the dataset
    """

    def __init__(
        self,
        tracker_id_to_images: dict[str, list[str]],
        transforms: Optional[Compose] = None,
    ):
        self.tracker_id_to_images = validate_tracker_id_to_images(tracker_id_to_images)
        self.transforms: Union[Compose, ToTensor] = transforms or ToTensor()
        self.tracker_ids = list(self.tracker_id_to_images.keys())

    @classmethod
    def from_image_directories(
        cls,
        root_directory: str,
        transforms: Optional[Compose] = None,
        image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> TripletsDataset:
        """
        Create TripletsDataset from a directory structured by tracker IDs.

        Args:
            root_directory (str): Root directory with tracker folders.
            transforms (Optional[Compose]): Optional image transformations.
            image_extensions (Tuple[str, ...]): Valid image extensions to load.

        Returns:
            TripletsDataset: An initialized dataset.
        """
        root_path = Path(root_directory)
        tracker_id_to_images = {}

        for tracker_path in sorted(root_path.iterdir()):
            if not tracker_path.is_dir():
                continue

            image_paths = sorted(
                [
                    str(image_path)
                    for image_path in tracker_path.glob("*")
                    if image_path.suffix.lower() in image_extensions
                    and image_path.is_file()
                ]
            )

            if image_paths:
                tracker_id_to_images[tracker_path.name] = image_paths

        return cls(
            tracker_id_to_images=tracker_id_to_images,
            transforms=transforms,
        )

    def __len__(self) -> int:
        """
        Return the number of unique tracker IDs (identities) in the dataset.

        Returns:
            int: The total number of unique identities (tracker IDs) available for
                sampling triplets.
        """
        return len(self.tracker_ids)

    def _load_and_transform_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        transformed_image: torch.Tensor = self.transforms(image)
        return transformed_image

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
        """
        Retrieve a random triplet (anchor, positive, negative) of images for a given
        identity.

        For the tracker ID at the given index, samples two different images as the
        anchor and positive (same identity), and one image from a different tracker ID
        as the negative (different identity).

        Args:
            index (int): Index of the tracker ID (identity) to sample the triplet from.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                A tuple containing the anchor, positive, and negative image tensors.
        """
        tracker_id = self.tracker_ids[index]

        anchor_image_path, positive_image_path, negative_image_path = (
            self._get_triplet_image_paths(tracker_id)
        )

        anchor_image = self._load_and_transform_image(anchor_image_path)
        positive_image = self._load_and_transform_image(positive_image_path)
        negative_image = self._load_and_transform_image(negative_image_path)

        return anchor_image, positive_image, negative_image

    def split(
        self,
        split_ratio: float = 0.8,
        random_state: Optional[int] = None,
        shuffle: bool = True,
    ) -> Tuple[TripletsDataset, TripletsDataset]:
        train_tracker_id_to_images, validation_tracker_id_to_images = train_test_split(
            list(self.tracker_id_to_images.keys()),
            train_ratio=split_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )
        train_data: Dict[str, List[str]] = {
            tracker_id: self.tracker_id_to_images[tracker_id]
            for tracker_id in train_tracker_id_to_images
        }
        val_data: Dict[str, List[str]] = {
            tracker_id: self.tracker_id_to_images[tracker_id]
            for tracker_id in validation_tracker_id_to_images
        }
        return (
            TripletsDataset(train_data, self.transforms),
            TripletsDataset(val_data, self.transforms),
        )
