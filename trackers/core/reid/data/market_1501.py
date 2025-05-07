import os
import random
from collections import defaultdict
from glob import glob
from typing import Dict, List, Optional, Union

from torchvision.transforms import Compose

from trackers.core.reid.data.base import TripletsDataset


def parse_market1501_dataset(data_dir: str, split: str) -> Dict[str, List[str]]:
    """Parse the [Market1501 dataset](https://paperswithcode.com/dataset/market-1501)
    to create a dictionary mapping tracker IDs to lists of image paths.

    Args:
        data_dir (str): The path to the Market1501 dataset.
        split (str): The mode to use. Must be one of "train" or "test".

    Returns:
        Dict[str, List[str]]: A dictionary mapping tracker IDs to lists of image paths.
    """
    data_dir = os.path.join(data_dir, f"bounding_box_{split}")
    image_files = glob(os.path.join(data_dir, "*.jpg"))
    tracker_id_to_images = defaultdict(list)
    for image_file in image_files:
        tracker_id = os.path.basename(image_file).split("_")[0]
        tracker_id_to_images[tracker_id].append(image_file)
    return dict(tracker_id_to_images)


def get_market1501_dataset(
    data_dir: str,
    validation_split_fraction: float = 0.2,
    transforms: Optional[Compose] = None,
    seed: Optional[Union[int, float, str, bytes, bytearray]] = None,
) -> dict[str, TripletsDataset]:
    """Get the [Market1501 dataset](https://paperswithcode.com/dataset/market-1501).

    Args:
        data_dir (str): The path to the
            [Market1501 dataset](https://paperswithcode.com/dataset/market-1501).
        validation_split_fraction (float): The fraction of the dataset to use
            for validation.
        transforms (Compose): The transforms to apply to the images.
        seed (Optional[Union[int, float, str, bytes, bytearray]]): The seed to use
            for the random number generator. If None, the random number generator will
            not be seeded.

    Returns:
        dict[str, TripletsDataset]: A dictionary mapping dataset splits to
            `TripletsDataset` objects.
    """
    random.seed(seed)
    if validation_split_fraction < 0 or validation_split_fraction > 1:
        raise ValueError("Validation split fraction must be between 0 and 1")
    tracker_id_to_images = parse_market1501_dataset(data_dir, "train")
    tracker_ids = list(tracker_id_to_images.keys())
    random.shuffle(tracker_ids)  # nosec B311
    num_train_samples = len(tracker_ids) * (1 - validation_split_fraction)
    train_tracker_ids = tracker_ids[: int(num_train_samples)]
    validation_tracker_ids = tracker_ids[int(num_train_samples) :]
    train_tracker_id_to_images = {
        tracker_id: tracker_id_to_images[tracker_id] for tracker_id in train_tracker_ids
    }
    validation_tracker_id_to_images = {
        tracker_id: tracker_id_to_images[tracker_id]
        for tracker_id in validation_tracker_ids
    }
    test_tracker_id_to_images = parse_market1501_dataset(data_dir, "test")
    return {
        "train": TripletsDataset(train_tracker_id_to_images, transforms),
        "validation": TripletsDataset(validation_tracker_id_to_images, transforms),
        "test": TripletsDataset(test_tracker_id_to_images, transforms),
    }
