import os
import shutil

import pytest
from firerequests import FireRequests

from trackers.core.reid import Market1501Dataset
from trackers.utils.data_utils import unzip_file

DATASET_URL = "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/market_1501.zip"


def test_reid_dataset():
    FireRequests().download(DATASET_URL)
    os.makedirs("test_data", exist_ok=True)
    shutil.move("market_1501.zip", "test_data/market_1501.zip")
    unzip_file("test_data/market_1501.zip", "test_data")
    os.remove("test_data/market_1501.zip")
    dataset = Market1501Dataset("./test_data/Market-1501-v15.09.15")

    # test length of dataset
    if not len(dataset) == 751:  # nosec B101
        pytest.fail(f"Dataset length mismatch. Expected 751, got {len(dataset)}")
    if not len(dataset.tracker_id_to_images["0002"]) == 46:  # nosec B101
        pytest.fail(
            "Tracker ID 0002 length mismatch. Expected 46,"
            f"got {len(dataset.tracker_id_to_images['0002'])}"
        )

    for idx in range(len(dataset)):
        (
            anchor_image,
            positive_image,
            negative_image,
        ) = dataset._get_triplet_image_paths(idx)

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
