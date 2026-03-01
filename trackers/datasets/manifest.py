#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


# ------------------------------------------------------------------------
# MOT-17
# ------------------------------------------------------------------------

BASE_MOT17 = (
    "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/mot17-v2"
)

DATASETS = {
    "mot17": {
        "description": (
            "MOT17 benchmark dataset with official TrackEval-compatible "
            "train/val split and renumbered validation frames."
        ),
        "splits": {
            "train": {
                "frames": {
                    "url": f"{BASE_MOT17}/mot17-train-frames.zip",
                    "md5": "65987312a51e9934c03679bba421b897",
                },
                "annotations": {
                    "url": f"{BASE_MOT17}/mot17-train-annotations.zip",
                    "md5": "1db34490e8a66fa09516aebbdcee48f4",
                },
                "detections": {
                    "url": f"{BASE_MOT17}/mot17-train-public-detections.zip",
                    "md5": "c3ebc4df29d23602f17729ef9b0eb933",
                },
            },
            "val": {
                "frames": {
                    "url": f"{BASE_MOT17}/mot17-val-frames.zip",
                    "md5": "e431859e5c5afdbd04acaba572056255",
                },
                "annotations": {
                    "url": f"{BASE_MOT17}/mot17-val-annotations.zip",
                    "md5": "24c2389850d47e5ad7af96425c31e4b5",
                },
                "detections": {
                    "url": f"{BASE_MOT17}/mot17-val-public-detections.zip",
                    "md5": "6421f23608a3394583ce79d9cb35283c",
                },
            },
            "test": {
                "frames": {
                    "url": f"{BASE_MOT17}/mot17-test-frames.zip",
                    "md5": "2b81a90fd834f38ce432d214381c5baf",
                },
                "detections": {
                    "url": f"{BASE_MOT17}/mot17-test-public-detections.zip",
                    "md5": "6f7bd92e162a6cecc752441d50b47a32",
                },
            },
        },
    }
}

# ------------------------------------------------------------------------
# SportsMOT (Need to update MD5 checksums once final versions are uploaded)
# ------------------------------------------------------------------------

BASE_SPORTSMOT = (
    "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/"
    "sportsmot-zips-v1"
)

DATASETS["sportsmot"] = {
    "description": "SportsMOT multi-object tracking benchmark.",
    "splits": {
        "train": {
            "frames": {
                "url": f"{BASE_SPORTSMOT}/sportsmot-train-frames.zip",
                "md5": "d92b648464d14e9c22587876b7ac3fbc",
            },
            "annotations": {
                "url": f"{BASE_SPORTSMOT}/sportsmot-train-annotations.zip",
                "md5": "4afae3c3e380b7b80008025a697bce45",
            },
        },
        "val": {
            "frames": {
                "url": f"{BASE_SPORTSMOT}/sportsmot-val-frames.zip",
                "md5": "850ca19cef57d4bf6ec5062dd30af725",
            },
            "annotations": {
                "url": f"{BASE_SPORTSMOT}/sportsmot-val-annotations.zip",
                "md5": "514fefc618cc71c40816fb2adf72f131",
            },
        },
        "test": {
            "frames": {
                "url": f"{BASE_SPORTSMOT}/sportsmot-test-frames.zip",
                "md5": "293dc3622792d89d1d4879fb391be1ff",
            },
        },
    },
}
