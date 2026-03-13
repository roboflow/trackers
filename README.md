<div align="center">
    <img width="200" src="https://raw.githubusercontent.com/roboflow/trackers/refs/heads/release/stable/docs/assets/logo-trackers-violet.svg" alt="trackers logo">
    <h1>trackers</h1>
    <p>Plug-and-play multi-object tracking for any detection model.</p>

[![version](https://badge.fury.io/py/trackers.svg)](https://badge.fury.io/py/trackers)
[![downloads](https://img.shields.io/pypi/dm/trackers)](https://pypistats.org/packages/trackers)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/trackers/blob/release/stable/LICENSE.md)
[![python-version](https://img.shields.io/pypi/pyversions/trackers)](https://badge.fury.io/py/trackers)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-bytetrack-tracker.ipynb)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

</div>

https://github.com/user-attachments/assets/eef9b00a-cfe4-40f7-a495-954550e3ef1f

## Install

```bash
pip install trackers
```

<details>
<summary>Install from source</summary>

```bash
pip install git+https://github.com/roboflow/trackers.git
```

</details>

## Track from CLI

Point at a video, webcam, RTSP stream, or image directory. Get tracked output.

```bash
trackers track \
    --source video.mp4 \
    --output output.mp4 \
    --model rfdetr-medium \
    --tracker bytetrack \
    --show-labels \
    --show-trajectories
```

Use our [interactive command builder](https://trackers.roboflow.com/develop/learn/track) to configure your tracking pipeline.

## Track from Python

Plug trackers into your existing detection pipeline. Works with any detector.

```python
import cv2
import supervision as sv
from inference import get_model
from trackers import ByteTrackTracker

model = get_model(model_id="rfdetr-medium")
tracker = ByteTrackTracker()

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = model.infer(frame)[0]
    detections = sv.Detections.from_inference(result)
    tracked = tracker.update(detections)
```

## Evaluate

Benchmark your tracker against ground truth with standard MOT metrics.

```bash
trackers eval \
    --gt-dir ./data/mot17/val \
    --tracker-dir results \
    --metrics CLEAR HOTA Identity \
    --columns MOTA HOTA IDF1
```

```
Sequence                        MOTA    HOTA    IDF1
----------------------------------------------------
MOT17-02-FRCNN                30.192  35.475  38.515
MOT17-04-FRCNN                48.912  55.096  61.854
MOT17-05-FRCNN                52.755  45.515  55.705
----------------------------------------------------
COMBINED                      47.406  50.355  56.600
```

## Download Datasets

Pull benchmark datasets for evaluation with a single command.

```bash
trackers download mot17 --split val --asset annotations,detections --output ./data
```

Supported datasets: MOT17, SportsMOT, SoccerNet, DanceTrack.

## Detection Quality

The detector is the most impactful part of the tracking pipeline. Swapping the detector alone can produce larger gains than any tracker-level optimization. See the [detection quality guide](https://trackers.roboflow.com/develop/learn/detection-quality/) for a full comparison.

## Algorithms

Clean, modular implementations of leading trackers. See the [tracker comparison](https://trackers.roboflow.com/develop/trackers/comparison/) for detailed benchmarks.

|                   Algorithm                   |  MOT17   | SportsMOT | SoccerNet |
| :-------------------------------------------: | :------: | :-------: | :-------: |
|   [SORT](https://arxiv.org/abs/1602.00763)    |   58.4   |   70.9    |   81.6    |
| [ByteTrack](https://arxiv.org/abs/2110.06864) |   60.1   | **73.0**  | **84.0**  |
|  [OC-SORT](https://arxiv.org/abs/2203.14360)  | **61.9** |   71.5    |   78.6    |

## Try It

Try trackers in your browser with our [Hugging Face Playground](https://huggingface.co/spaces/roboflow/trackers).

## Documentation

Full guides, API reference, and tutorials: [trackers.roboflow.com](https://trackers.roboflow.com)

## Contributing

We welcome contributions. Read our [contributor guidelines](https://github.com/roboflow/trackers/blob/release/stable/CONTRIBUTING.md) to get started.

## License

The code is released under the [Apache 2.0 license](https://github.com/roboflow/trackers/blob/release/stable/LICENSE).
