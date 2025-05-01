import os
import random
from glob import glob
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


class Market1501SiameseDataset(Dataset):
    """
    The Market-1501 siamese dataset for person re-identification.

    Args:
        data_dir (str): Path to the dataset directory for Market-1501.
        transforms (Optional[Compose]): Optional transforms to apply to the images.
            Default is `torchvision.transforms.ToTensor()`.
    """

    def __init__(self, data_dir: str, transforms: Optional[Compose] = None):
        self.data_dir = data_dir
        self.train_data_dir = os.path.join(data_dir, "bounding_box_train")
        self.triplet_classes = self.get_triplet_classes()
        self.transforms = transforms or ToTensor()

    def get_triplet_classes(self):
        image_files = glob(os.path.join(self.train_data_dir, "*.jpg"))
        classes = set(
            [file_path.split("/")[-1].split("_")[0] for file_path in image_files]
        )
        return list(classes)

    def __len__(self):
        return len(self.triplet_classes)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        triplet_class = self.triplet_classes[index]

        # get anchor image file
        class_image_files = glob(
            os.path.join(self.train_data_dir, f"{triplet_class}*.jpg")
        )
        anchor_image_file = random.choice(class_image_files)

        # get positive image file
        positive_image_file = random.choice(
            [
                file if file != anchor_image_file else random.choice(class_image_files)
                for file in class_image_files
            ]
        )

        # get negative image file
        negative_classes = [cls for cls in self.triplet_classes if cls != triplet_class]
        negative_class_image_files = glob(
            os.path.join(self.train_data_dir, f"{random.choice(negative_classes)}*.jpg")
        )
        negative_image_file = random.choice(negative_class_image_files)

        anchor_image = Image.open(anchor_image_file).convert("RGB")
        positive_image = Image.open(positive_image_file).convert("RGB")
        negative_image = Image.open(negative_image_file).convert("RGB")

        if self.transforms:
            anchor_image = self.transforms(anchor_image)
            positive_image = self.transforms(positive_image)
            negative_image = self.transforms(negative_image)
        return anchor_image, positive_image, negative_image
