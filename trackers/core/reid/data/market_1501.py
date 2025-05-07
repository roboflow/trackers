import os
from glob import glob
from typing import Dict, List, Optional

from torchvision.transforms import Compose

from trackers.core.reid.data.base import TripletsDataset


def parse_market1501_dataset(data_dir: str) -> Dict[str, List[str]]:
    """Parse the [Market1501 dataset](https://paperswithcode.com/dataset/market-1501)
    to create a dictionary mapping tracker IDs to lists of image paths.

    Args:
        data_dir (str): The path to the Market1501 dataset.

    Returns:
        Dict[str, List[str]]: A dictionary mapping tracker IDs to lists of image paths.
    """
    train_data_dir = os.path.join(data_dir, "bounding_box_train")
    image_files = glob(os.path.join(train_data_dir, "*.jpg"))
    unique_ids = set(
        os.path.basename(image_file).split("_")[0] for image_file in image_files
    )
    tracker_id_to_images: Dict[str, List[str]] = {
        tracker_id: [] for tracker_id in unique_ids
    }
    for image_file in image_files:
        tracker_id = os.path.basename(image_file).split("_")[0]
        tracker_id_to_images[tracker_id].append(image_file)
    return tracker_id_to_images


class Market1501Dataset(TripletsDataset):
    """[Market1501 dataset](https://paperswithcode.com/dataset/market-1501)
    that provides triplets of images for training ReID models.

    Args:
        data_dir (str): The path to the Market1501 dataset.
        transforms (Optional[Compose]): Optional image transformations to apply.
    """

    def __init__(self, data_dir: str, transforms: Optional[Compose] = None):
        tracker_id_to_images = parse_market1501_dataset(data_dir)
        super().__init__(tracker_id_to_images, transforms)
