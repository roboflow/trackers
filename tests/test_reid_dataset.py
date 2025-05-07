import os
import shutil

import pytest
from firerequests import FireRequests

from trackers.core.reid import get_market1501_dataset
from trackers.core.reid.data.market_1501 import parse_market1501_dataset
from trackers.utils.data_utils import unzip_file

DATASET_URL = "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/market_1501.zip"


@pytest.fixture
def market_dataset():
    os.makedirs("test_data", exist_ok=True)
    dataset_path = os.path.join("test_data", "Market-1501-v15.09.15")
    zip_path = os.path.join("test_data", "market_1501.zip")
    if not os.path.exists(dataset_path):
        if not os.path.exists(zip_path):
            FireRequests().download(DATASET_URL)
            shutil.move("market_1501.zip", str(zip_path))
        unzip_file(str(zip_path), "test_data")
    yield dataset_path


def validate_dataset_triplet_paths(dataset):
    for tracker_id in dataset.tracker_ids:
        anchor_image_path, positive_image_path, negative_image_path = (
            dataset._get_triplet_image_paths(tracker_id)
        )
        anchor_image_id = os.path.basename(anchor_image_path).split("_")[0]
        positive_image_id = os.path.basename(positive_image_path).split("_")[0]
        negative_image_id = os.path.basename(negative_image_path).split("_")[0]
        if anchor_image_id != positive_image_id:
            pytest.fail(
                "Anchor and positive image IDs mismatch. "
                f"Expected {anchor_image_id} == {positive_image_id}"
            )
        if anchor_image_id == negative_image_id:
            pytest.fail(
                "Anchor and negative image IDs mismatch. "
                f"Expected {anchor_image_id} != {negative_image_id}"
            )


@pytest.mark.parametrize("split", ["train", "test"])
def test_market1501_dataset_triplet_paths(market_dataset, split):
    dataset = get_market1501_dataset(
        os.path.join(market_dataset, f"bounding_box_{split}")
    )
    if split == "train":
        if not len(dataset) == 751:  # nosec B101
            pytest.fail(f"Dataset length mismatch. Expected 751, got {len(dataset)}")
    elif split == "test":
        if not len(dataset) == 752:  # nosec B101
            pytest.fail(f"Dataset length mismatch. Expected 752, got {len(dataset)}")
    else:
        pytest.fail(f"Invalid split. Expected 'train' or 'test', got {split}")
    validate_dataset_triplet_paths(dataset)
    


@pytest.mark.parametrize("split_ratio", [None, 0.8])
def test_market1501_dataset_split_ratio(market_dataset, split_ratio):
    dataset = get_market1501_dataset(
        os.path.join(market_dataset, "bounding_box_train"), split_ratio=split_ratio
    )
    if split_ratio is None:
        if not len(dataset) == 751:  # nosec B101
            pytest.fail(f"Dataset length mismatch. Expected 751, got {len(dataset)}")
        validate_dataset_triplet_paths(dataset)
    else:
        train_dataset, val_dataset = dataset
        if not len(train_dataset) == 600:  # nosec B101
            pytest.fail(f"Dataset length mismatch. Expected 751, got {len(train_dataset)}")
        if not len(val_dataset) == 151:  # nosec B101
            pytest.fail(f"Dataset length mismatch. Expected 151, got {len(val_dataset)}")
        validate_dataset_triplet_paths(train_dataset)
        validate_dataset_triplet_paths(val_dataset)
    


@pytest.mark.parametrize("split", ["train", "test"])
def test_parse_market1501_dataset(market_dataset, split):
    dataset_path = os.path.join(market_dataset, f"bounding_box_{split}")
    tracker_id_to_images = parse_market1501_dataset(dataset_path)
    if split == "train":
        if not len(tracker_id_to_images) == 751:  # nosec B101
            pytest.fail(
                "Dataset length mismatch. "
                f"Expected 751, got {len(tracker_id_to_images)}"
            )
    elif split == "test":
        if not len(tracker_id_to_images) == 752:  # nosec B101
            pytest.fail(
                "Dataset length mismatch. "
                f"Expected 752, got {len(tracker_id_to_images)}"
            )
    else:
        pytest.fail(f"Invalid split. Expected 'train' or 'test', got {split}")
