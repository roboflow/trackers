import secrets
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


class TripletsDataset(Dataset):
    def __init__(
        self,
        tracker_id_to_images: Dict[str, List[str]],
        transforms: Optional[Compose] = None,
    ):
        self.tracker_id_to_images = tracker_id_to_images
        self.transforms = transforms or ToTensor()
        self.tracker_ids = list(tracker_id_to_images.keys())

    def __len__(self) -> int:
        return len(self.tracker_ids)

    def _load_and_transform_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image

    def _secure_sample(self, population, k=1):
        """Securely sample k elements from the population."""
        if k == 1:
            return [secrets.choice(population)]

        # For multiple samples, shuffle and take first k
        result = list(population)
        for i in range(len(result) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            result[i], result[j] = result[j], result[i]
        return result[:k]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tracker_id = self.tracker_ids[index]
        tracker_id_image_paths = self.tracker_id_to_images[tracker_id]

        # Use secure sampling for anchor and positive images
        anchor_image_path, positive_image_path = self._secure_sample(
            tracker_id_image_paths, 2
        )

        # Use secrets for negative tracker ID selection
        negative_candidates = [tid for tid in self.tracker_ids if tid != tracker_id]
        negative_tracker_id = secrets.choice(negative_candidates)

        # Use secrets for negative image selection
        negative_image_path = secrets.choice(
            self.tracker_id_to_images[negative_tracker_id]
        )

        anchor_image = self._load_and_transform_image(anchor_image_path)
        positive_image = self._load_and_transform_image(positive_image_path)
        negative_image = self._load_and_transform_image(negative_image_path)

        return anchor_image, positive_image, negative_image
