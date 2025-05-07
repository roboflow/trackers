import os
import shutil

import pytest
from firerequests import FireRequests

from trackers.core.reid import get_market1501_dataset
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


def test_market1501_dataset_train_val_split(market_dataset):
    train_dataset, val_dataset = get_market1501_dataset(
        os.path.join(market_dataset, "bounding_box_train"), split_ratio=0.8
    )
    if not len(train_dataset) == 600:  # nosec B101
        pytest.fail(f"Dataset length mismatch. Expected 751, got {len(train_dataset)}")
    if not len(val_dataset) == 151:  # nosec B101
        pytest.fail(f"Dataset length mismatch. Expected 151, got {len(val_dataset)}")

    for tracker_id in train_dataset.tracker_ids:
        anchor_image_path, positive_image_path, negative_image_path = (
            train_dataset._get_triplet_image_paths(tracker_id)
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

    for tracker_id in val_dataset.tracker_ids:
        anchor_image_path, positive_image_path, negative_image_path = (
            val_dataset._get_triplet_image_paths(tracker_id)
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


def test_market1501_dataset_test_split(market_dataset):
    test_dataset = get_market1501_dataset(
        os.path.join(market_dataset, "bounding_box_test"), split_ratio=None
    )
    if not len(test_dataset) == 752:  # nosec B101
        pytest.fail(f"Dataset length mismatch. Expected 752, got {len(test_dataset)}")
    for tracker_id in test_dataset.tracker_ids:
        anchor_image_path, positive_image_path, negative_image_path = (
            test_dataset._get_triplet_image_paths(tracker_id)
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
