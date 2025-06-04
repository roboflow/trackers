import glob
import os
import re
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from torchvision.transforms import Compose

from trackers.core.reid.dataset.base import TripletsDataset


class DatasetType(Enum):
    TRIPLET = "triplet"
    IDENTITY = "identity"


def parse_market1501_triplet_mapping(data_dir: str) -> Dict[str, List[str]]:
    """Parse the [Market1501 dataset](https://paperswithcode.com/dataset/market-1501)
    to create a dictionary mapping tracker IDs to lists of image paths.

    Args:
        data_dir (str): The path to the Market1501 dataset.

    Returns:
        Dict[str, List[str]]: A dictionary mapping tracker IDs to lists of image paths.
    """
    image_files = glob.glob(os.path.join(data_dir, "*.jpg"))
    tracker_id_to_images = defaultdict(list)
    for image_file in image_files:
        tracker_id = os.path.basename(image_file).split("_")[0]
        tracker_id_to_images[tracker_id].append(image_file)
    return dict(tracker_id_to_images)


def parse_market1501_identity_mapping(
    data_dir: str, relabel: bool = True
) -> List[Dict[str, Any]]:
    """Parse the [Market1501 dataset](https://paperswithcode.com/dataset/market-1501)
    to create a list of dictionaries mapping image paths to entity IDs and camera IDs.

    Args:
        data_dir (str): The path to the Market1501 dataset.
        relabel (bool): Whether to relabel the entity IDs.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries mapping image paths to
            entity IDs and camera IDs.
    """
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    pattern = re.compile(r"([-\d]+)_c(\d)")
    entity_ids = set()
    for image_path in image_paths:
        match = pattern.search(image_path)
        if match is None:
            continue
        entity_id, _ = map(int, match.groups())
        if entity_id != -1:
            entity_ids.add(entity_id)
    entity_id_to_label = {
        entity_id: label for label, entity_id in enumerate(entity_ids)
    }
    identity_mappings = []
    for image_path in image_paths:
        match = pattern.search(image_path)
        if match is None:
            continue
        entity_id, camera_id = map(int, match.groups())
        if entity_id != -1:
            camera_id -= 1
            if relabel:
                entity_id = entity_id_to_label[entity_id]
            identity_mappings.append(
                {
                    "image_path": image_path,
                    "entity_id": entity_id,
                    "camera_id": camera_id,
                }
            )
    return identity_mappings


def get_market1501_triplets_dataset(
    data_dir: str,
    dataset_type: DatasetType = DatasetType.TRIPLET,
    split_ratio: Optional[float] = None,
    random_state: Optional[Union[int, float, str, bytes, bytearray]] = None,
    shuffle: bool = True,
    transforms: Optional[Compose] = None,
) -> Union[TripletsDataset, Tuple[TripletsDataset, TripletsDataset]]:
    """Get the [Market1501 dataset](https://paperswithcode.com/dataset/market-1501).

    Args:
        data_dir (str): The path to the bounding box train/test directory of the
            [Market1501 dataset](https://paperswithcode.com/dataset/market-1501).
        dataset_type (DatasetType): The type of the dataset to return.
        split_ratio (Optional[float]): The ratio of the dataset to split into training
            and validation sets. If `None`, the dataset is returned as a single
            `TripletsDataset` object, otherwise the dataset is split into a tuple of
            training and validation `TripletsDataset` objects.
        random_state (Optional[Union[int, float, str, bytes, bytearray]]): The random
            state to use for the split.
        shuffle (bool): Whether to shuffle the dataset.
        transforms (Optional[Compose]): The transforms to apply to the dataset.

    Returns:
        Union[TripletsDataset, Tuple[TripletsDataset, TripletsDataset]]: A single
            `TripletsDataset` object or a tuple of training and validation
            `TripletsDataset` objects.
    """
    if dataset_type == DatasetType.TRIPLET:
        tracker_id_to_images = parse_market1501_triplet_mapping(data_dir)
        dataset = TripletsDataset(tracker_id_to_images, transforms)
        if split_ratio is not None:
            train_dataset, validation_dataset = dataset.split(
                split_ratio=split_ratio, random_state=random_state, shuffle=shuffle
            )
            return train_dataset, validation_dataset
        return dataset
    else:
        # Handle other dataset types if needed in the future
        raise NotImplementedError(f"Dataset type {dataset_type} is not implemented")
