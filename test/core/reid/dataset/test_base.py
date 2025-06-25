import os
import shutil

import pytest

from trackers.core.reid.dataset.base import IdentityDataset
from trackers.core.reid.dataset.market_1501 import parse_market1501_dataset
from trackers.core.reid.model import ReIDModel
from trackers.utils.data_utils import unzip_file
from trackers.utils.downloader import download_file

DATASET_URL = "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/market_1501.zip"


@pytest.fixture
def market_1501_dataset():
    os.makedirs("test_data", exist_ok=True)
    dataset_path = os.path.join("test_data", "Market-1501-v15.09.15")
    zip_path = os.path.join("test_data", "market_1501.zip")
    if not os.path.exists(dataset_path):
        if not os.path.exists(zip_path):
            download_file(DATASET_URL)
            shutil.move("market_1501.zip", str(zip_path))
        unzip_file(str(zip_path), "test_data")
    yield dataset_path


@pytest.mark.parametrize(
    "dataset_split", ["bounding_box_train, bounding_box_test", "query"]
)
def test_identity_dataset(market_1501_dataset, dataset_split):
    dataset_path = os.path.join(market_1501_dataset, dataset_split)
    dataset = IdentityDataset(parse_market1501_dataset(data_dir=dataset_path))
    if dataset_split == "bounding_box_train":
        assert len(dataset) == 12936
        assert dataset.get_num_identities() == 751
    elif dataset_split == "bounding_box_test":
        assert len(dataset) == 15913
        assert dataset.get_num_identities() == 751
    elif dataset_split == "query":
        assert len(dataset) == 3368
        assert dataset.get_num_identities() == 750


@pytest.mark.parametrize(
    "dataset_split", ["bounding_box_train, bounding_box_test", "query"]
)
def test_reid_model_classification_head(market_1501_dataset, dataset_split):
    dataset_path = os.path.join(market_1501_dataset, dataset_split)
    dataset = IdentityDataset(parse_market1501_dataset(data_dir=dataset_path))
    model = ReIDModel.from_timm("resnet50")
    model.add_classification_head(
        num_classes=dataset.get_num_identities(), freeze_backbone=True
    )
    assert model.backbone[-1].out_features == dataset.get_num_identities()
