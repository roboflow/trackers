# trackers/datasets/manifest.py

BASE_MOT17 = (
    "https://storage.googleapis.com/com-roboflow-marketing/trackers/datasets/mot17-v1"
)

DATASETS = {
    "mot17": {
        "description": "MOT17 benchmark Dataset.",
        "splits": {
            "train": {
                "frames": f"{BASE_MOT17}/mot17-train-frames.zip",
                "annotations": f"{BASE_MOT17}/mot17-train-annotations.zip",
                "detections": f"{BASE_MOT17}/mot17-train-public-detections.zip",
            },
            "val": {
                "frames": f"{BASE_MOT17}/mot17-val-frames.zip",
                "annotations": f"{BASE_MOT17}/mot17-val-annotations.zip",
                "detections": f"{BASE_MOT17}/mot17-val-public-detections.zip",
            },
            "test": {
                "frames": f"{BASE_MOT17}/mot17-test-frames.zip",
                "detections": f"{BASE_MOT17}/mot17-test-public-detections.zip",
            },
        },
    }
}
