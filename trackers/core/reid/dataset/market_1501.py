import glob
import os
import re
from typing import Callable, Optional, Tuple, Union

from torchvision.transforms import Compose

from trackers.core.reid.dataset.base import IdentityDataset


def parse_market1501_dataset(
    data_dir: str, relabel: bool = False
) -> list[Tuple[str, int, int]]:
    """Parse the [Market1501 dataset](https://paperswithcode.com/dataset/market-1501)
    to create a list of tuples, each containing an image path, an identity label,
    and a camera label.

    Args:
        data_dir (str): The path to the Market1501 dataset.
        relabel (bool): Whether to relabel the identities to a compact range of starting
            from 0.

    Returns:
        list[Tuple[str, int, int]]: A list of tuples, each containing an image path,
            an identity label, and a camera label.
    """
    image_files = glob.glob(os.path.join(data_dir, "*.jpg"))
    file_pattern = re.compile(r"([-\d]+)_c(\d)")
    id_container = set()
    data = []
    for img_path in image_files:
        match = file_pattern.search(img_path)
        if match is None:
            continue
        identity, camera_id = map(int, match.groups())
        if identity != -1:
            id_container.add(identity)
            data.append((img_path, identity, camera_id - 1))

    if relabel:
        id_to_label = {identity: label for label, identity in enumerate(id_container)}
        data = [
            (img_path, id_to_label[identity], camera_id)
            for img_path, identity, camera_id in data
        ]

    return data


def get_market1501_dataset(
    data_dir: str,
    split_ratio: Optional[float] = None,
    random_state: Optional[Union[int, float, str, bytes, bytearray]] = None,
    shuffle: bool = True,
    transforms: Optional[Union[Callable, Compose]] = None,
) -> Union[IdentityDataset, Tuple[IdentityDataset, IdentityDataset]]:
    """Get the [Market1501 dataset](https://paperswithcode.com/dataset/market-1501).

    Args:
        data_dir (str): The path to the bounding box train/test directory of the
            [Market1501 dataset](https://paperswithcode.com/dataset/market-1501).
        split_ratio (Optional[float]): The ratio of the dataset to split into training
            and validation sets. If `None`, the dataset is returned as a single
            `IdentityDataset` object, otherwise the dataset is split into a tuple of
            training and validation `IdentityDataset` objects.
        random_state (Optional[Union[int, float, str, bytes, bytearray]]): The random
            state to use for the split.
        shuffle (bool): Whether to shuffle the dataset.
        transforms (Optional[Union[Callable, Compose]]): The transforms to apply to
            the dataset.

    Returns:
        Union[IdentityDataset, Tuple[IdentityDataset, IdentityDataset]]: A single
            `IdentityDataset` object if `split_ratio` is `None`, otherwise a tuple of
            training and validation `IdentityDataset` objects.
    """
    dataset = IdentityDataset(parse_market1501_dataset(data_dir), transforms=transforms)
    if split_ratio is not None:
        train_dataset, validation_dataset = dataset.split(
            split_ratio=split_ratio, random_state=random_state, shuffle=shuffle
        )
        return train_dataset, validation_dataset
    return dataset
