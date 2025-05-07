import os
import shutil

import pytest
from firerequests import FireRequests
from tqdm import tqdm

from trackers.core.reid import get_market1501_dataset
from trackers.utils.data_utils import unzip_file

MARKER_1501_DATASET_URL = "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/market_1501.zip"


def test_market_1501_dataset():
    FireRequests().download(MARKER_1501_DATASET_URL)
    os.makedirs("test_data", exist_ok=True)
    shutil.move("market_1501.zip", "test_data/market_1501.zip")
    unzip_file("test_data/market_1501.zip", "test_data")
    os.remove("test_data/market_1501.zip")
    dataset_dict = get_market1501_dataset("./test_data/Market-1501-v15.09.15")

    train_dataset = dataset_dict["train"]
    validation_dataset = dataset_dict["validation"]
    test_dataset = dataset_dict["test"]

    # test length of dataset
    if not len(train_dataset) == 600:  # nosec B101
        pytest.fail(f"Dataset length mismatch. Expected 751, got {len(train_dataset)}")
    if not len(validation_dataset) == 151:  # nosec B101
        pytest.fail(
            f"Dataset length mismatch. Expected 388, got {len(validation_dataset)}"
        )
    if not len(test_dataset) == 752:  # nosec B101
        pytest.fail(f"Dataset length mismatch. Expected 752, got {len(test_dataset)}")

    # test triplet image paths
    for idx in tqdm(train_dataset.tracker_ids):
        (
            anchor_image,
            positive_image,
            negative_image,
        ) = train_dataset._get_triplet_image_paths(idx)

        anchor_image_id = os.path.basename(anchor_image).split("_")[0]
        positive_image_id = os.path.basename(positive_image).split("_")[0]
        negative_image_id = os.path.basename(negative_image).split("_")[0]

        if anchor_image_id != positive_image_id:
            pytest.fail(
                f"Anchor image ID {anchor_image_id} does not match "
                f"positive image ID {positive_image_id}"
            )
        if anchor_image_id == negative_image_id:
            pytest.fail(
                f"Anchor image ID {anchor_image_id} matches "
                f"negative image ID {negative_image_id}"
            )

    for idx in tqdm(validation_dataset.tracker_ids):
        (
            anchor_image,
            positive_image,
            negative_image,
        ) = validation_dataset._get_triplet_image_paths(idx)

        anchor_image_id = os.path.basename(anchor_image).split("_")[0]
        positive_image_id = os.path.basename(positive_image).split("_")[0]
        negative_image_id = os.path.basename(negative_image).split("_")[0]

        if anchor_image_id != positive_image_id:
            pytest.fail(
                f"Anchor image ID {anchor_image_id} does not match "
                f"positive image ID {positive_image_id}"
            )
        if anchor_image_id == negative_image_id:
            pytest.fail(
                f"Anchor image ID {anchor_image_id} matches "
                f"negative image ID {negative_image_id}"
            )

    for idx in tqdm(test_dataset.tracker_ids):
        (
            anchor_image,
            positive_image,
            negative_image,
        ) = test_dataset._get_triplet_image_paths(idx)

        anchor_image_id = os.path.basename(anchor_image).split("_")[0]
        positive_image_id = os.path.basename(positive_image).split("_")[0]
        negative_image_id = os.path.basename(negative_image).split("_")[0]

        if anchor_image_id != positive_image_id:
            pytest.fail(
                f"Anchor image ID {anchor_image_id} does not match "
                f"positive image ID {positive_image_id}"
            )
        if anchor_image_id == negative_image_id:
            pytest.fail(
                f"Anchor image ID {anchor_image_id} matches "
                f"negative image ID {negative_image_id}"
            )

    shutil.rmtree("test_data")
