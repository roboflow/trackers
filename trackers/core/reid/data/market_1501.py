import os
from glob import glob
from secrets import SystemRandom
from typing import Optional

from torchvision.transforms import Compose

from trackers.core.reid.data.base import TripletsDataset


class Market1501Dataset(TripletsDataset):
    """
    The Market-1501 siamese dataset for person re-identification.

    Args:
        data_dir (str): Path to the dataset directory for Market-1501.
        transforms (Optional[Compose]): Optional transforms to apply to the images.
            Default is `torchvision.transforms.ToTensor()`.
    """

    def __init__(self, data_dir: str, transforms: Optional[Compose] = None):
        super().__init__(data_dir=data_dir, transforms=transforms)
        self.train_data_dir = os.path.join(data_dir, "bounding_box_train")
        self.secure_random = SystemRandom()

    def get_triplet_classes(self):
        train_data_dir = os.path.join(self.data_dir, "bounding_box_train")
        image_files = glob(os.path.join(train_data_dir, "*.jpg"))
        classes = set(
            [file_path.split("/")[-1].split("_")[0] for file_path in image_files]
        )
        return list(classes)

    def __len__(self):
        return len(self.triplet_classes)

    def get_anchor_image_file(self, triplet_class: str) -> str:
        class_image_files = glob(
            os.path.join(self.train_data_dir, f"{triplet_class}*.jpg")
        )
        return self.secure_random.choice(class_image_files)

    def get_positive_image_file(self, triplet_class: str, anchor_image_file) -> str:
        class_image_files = glob(
            os.path.join(self.train_data_dir, f"{triplet_class}*.jpg")
        )
        return self.secure_random.choice(
            [
                file
                if file != anchor_image_file
                else self.secure_random.choice(class_image_files)
                for file in class_image_files
            ]
        )

    def get_negative_image_file(self, triplet_class: str) -> str:
        negative_classes = [cls for cls in self.triplet_classes if cls != triplet_class]
        negative_class_image_files = glob(
            os.path.join(
                self.train_data_dir,
                f"{self.secure_random.choice(negative_classes)}*.jpg",
            )
        )
        return self.secure_random.choice(negative_class_image_files)
