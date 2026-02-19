<div align="center">
    <img width="200" src="https://raw.githubusercontent.com/roboflow/trackers/refs/heads/release/stable/docs/assets/logo-trackers-violet.svg" alt="trackers logo">
    <h1>trackers</h1>
    <p>Plug-and-play multi-object tracking for any detection model.</p>

[![version](https://badge.fury.io/py/trackers.svg)](https://badge.fury.io/py/trackers)
[![downloads](https://img.shields.io/pypi/dm/trackers)](https://pypistats.org/packages/trackers)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/trackers/blob/release/stable/LICENSE.md)
[![python-version](https://img.shields.io/pypi/pyversions/trackers)](https://badge.fury.io/py/trackers)
[![hf space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Roboflow/Trackers)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-bytetrack-tracker.ipynb)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

</div>

## Try It

No install needed. Try trackers in your browser with our [Hugging Face Playground](https://huggingface.co/spaces/roboflow/trackers).

## Install

```bash
pip install trackers
```

<details>
<summary>install from source</summary>

```bash
pip install git+https://github.com/roboflow/trackers.git
```

</details>

https://github.com/user-attachments/assets/eef9b00a-cfe4-40f7-a495-954550e3ef1f

## Track from CLI

Point at a video, webcam, RTSP stream, or image directory. Get tracked output.

Use our [interactive command builder](https://trackers.roboflow.com/develop/learn/track) to configure your tracking pipeline.

```bash
trackers track \
    --source video.mp4 \
    --output output.mp4 \
    --model rfdetr-medium \
    --tracker bytetrack \
    --show-labels \
    --show-trajectories
```

## Track from Python

Plug trackers into your existing detection pipeline. Works with any detector.

```python
import cv2
import supervision as sv
from inference import get_model
from trackers import ByteTrackTracker

model = get_model(model_id="rfdetr-medium")
tracker = ByteTrackTracker()

label_annotator = sv.LabelAnnotator()
trajectory_annotator = sv.TrajectoryAnnotator()

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = model.infer(frame)[0]
    detections = sv.Detections.from_inference(result)
    tracked = tracker.update(detections)

    frame = label_annotator.annotate(frame, tracked)
    frame = trajectory_annotator.annotate(frame, tracked)
```

## Evaluate

Benchmark your tracker against ground truth with standard MOT metrics.

```bash
trackers eval \
    --gt-dir data/gt \
    --tracker-dir data/trackers \
    --metrics CLEAR HOTA Identity
```

```
Sequence                        MOTA    HOTA    IDF1  IDSW
----------------------------------------------------------
MOT17-02-FRCNN                75.600  62.300  72.100    42
MOT17-04-FRCNN                78.200  65.100  74.800    31
----------------------------------------------------------
COMBINED                      75.033  62.400  72.033    73
```

## Algorithms

Clean, modular implementations of leading trackers. See the [tracker comparison](https://trackers.roboflow.com/develop/trackers/comparison/) for detailed benchmarks.

|                   Algorithm                   |  MOT17   | SportsMOT | SoccerNet |
| :-------------------------------------------: | :------: | :-------: | :-------: |
|   [SORT](https://arxiv.org/abs/1602.00763)    |   58.4   |   70.9    |   81.6    |
| [ByteTrack](https://arxiv.org/abs/2110.06864) | **60.1** | **73.0**  | **84.0**  |
|  [OC-SORT](https://arxiv.org/abs/2203.14360)  |    —     |     —     |     —     |
| [BoT-SORT](https://arxiv.org/abs/2206.14651)  |    —     |     —     |     —     |
|  [McByte](https://arxiv.org/abs/2506.01373)   |    —     |     —     |     —     |

## Contributing

We welcome contributions. Read our [contributor guidelines](https://github.com/roboflow/trackers/blob/release/stable/CONTRIBUTING.md) to get started.

## License

The code is released under the [Apache 2.0 license](https://github.com/roboflow/trackers/blob/release/stable/LICENSE).
