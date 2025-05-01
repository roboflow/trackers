import os
import shutil

from firerequests import FireRequests

from trackers.core.reid import Market1501SiameseDataset
from trackers.utils.data_utils import unzip_file

DATASET_URL = "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/market_1501.zip"


def test_reid_dataset():
    FireRequests().download(DATASET_URL)
    os.makedirs("test_data", exist_ok=True)
    shutil.move("market_1501.zip", "test_data/market_1501.zip")
    unzip_file("test_data/market_1501.zip", "test_data")
    os.remove("test_data/market_1501.zip")
    dataset = Market1501SiameseDataset("./test_data/Market-1501-v15.09.15")
    assert len(dataset) == 751
    shutil.rmtree("test_data")
