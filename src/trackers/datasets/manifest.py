# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from enum import Enum
from typing import Any


class Dataset(str, Enum):
    """Supported benchmark tracking datasets.

    Attributes:
        MOT17: Pedestrian tracking with crowded scenes and frequent
            occlusions. Strongly tests re-identification and identity
            stability.
        SPORTSMOT: Sports broadcast tracking with fast motion, camera
            pans, and similar-looking targets. Tests association under
            speed and appearance ambiguity.
    """

    MOT17 = "mot17"
    SPORTSMOT = "sportsmot"


class DatasetSplit(str, Enum):
    """Available dataset splits.

    Attributes:
        TRAIN: Training split.
        VAL: Validation split.
        TEST: Test split.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetAsset(str, Enum):
    """Downloadable asset types within a dataset split.

    Attributes:
        FRAMES: Raw video frames as individual image files.
        ANNOTATIONS: Ground-truth bounding box and identity labels.
        DETECTIONS: Pre-computed object detection results.
    """

    FRAMES = "frames"
    ANNOTATIONS = "annotations"
    DETECTIONS = "detections"


_BASE_MOT17_URL = (
    "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/mot17-v2"
)

_BASE_SPORTSMOT_URL = (
    "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/"
    "sportsmot-v1"
)

_DATASETS: dict[Dataset, dict[str, Any]] = {
    Dataset.MOT17: {
        "description": (
            "Pedestrian tracking with crowded scenes and frequent"
            " occlusions. Strongly tests re-identification and"
            " identity stability."
        ),
        "splits": {
            "train": {
                "frames": {
                    "url": f"{_BASE_MOT17_URL}/mot17-train-frames.zip",
                    "md5": "65987312a51e9934c03679bba421b897",
                },
                "annotations": {
                    "url": f"{_BASE_MOT17_URL}/mot17-train-annotations.zip",
                    "md5": "1db34490e8a66fa09516aebbdcee48f4",
                },
                "detections": {
                    "url": f"{_BASE_MOT17_URL}/mot17-train-public-detections.zip",
                    "md5": "c3ebc4df29d23602f17729ef9b0eb933",
                },
            },
            "val": {
                "frames": {
                    "url": f"{_BASE_MOT17_URL}/mot17-val-frames.zip",
                    "md5": "e431859e5c5afdbd04acaba572056255",
                },
                "annotations": {
                    "url": f"{_BASE_MOT17_URL}/mot17-val-annotations.zip",
                    "md5": "24c2389850d47e5ad7af96425c31e4b5",
                },
                "detections": {
                    "url": f"{_BASE_MOT17_URL}/mot17-val-public-detections.zip",
                    "md5": "6421f23608a3394583ce79d9cb35283c",
                },
            },
            "test": {
                "frames": {
                    "url": f"{_BASE_MOT17_URL}/mot17-test-frames.zip",
                    "md5": "2b81a90fd834f38ce432d214381c5baf",
                },
                "detections": {
                    "url": f"{_BASE_MOT17_URL}/mot17-test-public-detections.zip",
                    "md5": "6f7bd92e162a6cecc752441d50b47a32",
                },
            },
        },
    },
    Dataset.SPORTSMOT: {
        "description": (
            "Sports broadcast tracking with fast motion, camera pans,"
            " and similar-looking targets. Tests association under"
            " speed and appearance ambiguity."
        ),
        "splits": {
            "train": {
                "frames": {
                    "url": f"{_BASE_SPORTSMOT_URL}/sportsmot-train-frames.zip",
                    "md5": "d92b648464d14e9c22587876b7ac3fbc",
                },
                "annotations": {
                    "url": f"{_BASE_SPORTSMOT_URL}/sportsmot-train-annotations.zip",
                    "md5": "4afae3c3e380b7b80008025a697bce45",
                },
            },
            "val": {
                "frames": {
                    "url": f"{_BASE_SPORTSMOT_URL}/sportsmot-val-frames.zip",
                    "md5": "850ca19cef57d4bf6ec5062dd30af725",
                },
                "annotations": {
                    "url": f"{_BASE_SPORTSMOT_URL}/sportsmot-val-annotations.zip",
                    "md5": "514fefc618cc71c40816fb2adf72f131",
                },
            },
            "test": {
                "frames": {
                    "url": f"{_BASE_SPORTSMOT_URL}/sportsmot-test-frames.zip",
                    "md5": "293dc3622792d89d1d4879fb391be1ff",
                },
            },
        },
    },
}
