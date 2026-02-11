[![version](https://badge.fury.io/py/trackers.svg)](https://badge.fury.io/py/trackers)
[![downloads](https://img.shields.io/pypi/dm/trackers)](https://pypistats.org/packages/trackers)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/trackers/blob/main/LICENSE.md)
[![python-version](https://img.shields.io/pypi/pyversions/trackers)](https://badge.fury.io/py/trackers)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

<div align="center">
    <h1 align="center">trackers</h1>
    <img width="200" src="https://raw.githubusercontent.com/roboflow/trackers/refs/heads/main/docs/assets/logo-trackers-violet.svg" alt="trackers logo">
</div>

`trackers` gives you clean, modular re-implementations of leading multi-object tracking algorithms released under the permissive Apache 2.0 license. You combine them with any detection model you already use.

https://github.com/user-attachments/assets/eef9b00a-cfe4-40f7-a495-954550e3ef1f

## Install

You can install and use `trackers` in a [**Python>=3.10**](https://www.python.org/) environment. For detailed installation instructions, including installing from source and setting up a local development environment, check out our [install](https://trackers.roboflow.com/develop/learn/install/) page.

```bash
pip install trackers
```

<details>
<summary>install from source</summary>

<br>

By installing `trackers` from source, you can explore the most recent features and enhancements that have not yet been officially released. Please note that these updates are still in development and may not be as stable as the latest published release.

```bash
pip install https://github.com/roboflow/trackers/archive/refs/heads/develop.zip
```

</details>

## Quickstart

Use the `trackers` CLI to quickly test how our tracking algorithms perform on your videos and streams. This feature is experimental; see the [CLI documentation](https://roboflow.github.io/trackers/learn/track/) for details.

```bash
trackers track --source source.mp4 --output output.mp4 --model rfdetr-nano --tracker bytetrack
```

## Tracking Algorithms

`trackers` gives you clean, modular re-implementations of leading multi-object tracking algorithms. The package currently supports [SORT](https://arxiv.org/abs/1602.00763) and [ByteTrack](https://arxiv.org/abs/2110.06864). [OC-SORT](https://arxiv.org/abs/2203.14360), [BoT-SORT](https://arxiv.org/abs/2206.14651), and [McByte](https://arxiv.org/abs/2506.01373) support is coming soon. For comparisons, see the [tracker comparison](https://trackers.roboflow.com/develop/trackers/comparison/) page.

| Algorithm | MOT17 HOTA | MOT17 IDF1 | MOT17 MOTA | SportsMOT HOTA | SoccerNet HOTA |
| :-------: | :--------: | :--------: | :--------: | :------------: | :------------: |
|   SORT    |    58.4    |    69.9    |    67.2    |      70.9      |      81.6      |
| ByteTrack |  **60.1**  |  **73.2**  |  **74.1**  |    **73.0**    |    **84.0**    |
|  OC-SORT  |     —      |     —      |     —      |       —        |       —        |
| BoT-SORT  |     —      |     —      |     —      |       —        |       —        |
|  McByte   |     —      |     —      |     —      |       —        |       —        |

## Integration

With a modular design, `trackers` lets you combine object detectors from different libraries with the tracker of your choice.

```python
import cv2
from rfdetr import RFDETRNano
from trackers import ByteTrackTracker

model = RFDETRNano()
tracker = ByteTrackTracker()

cap = cv2.VideoCapture("source.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = model.predict(frame_rgb)
    detections = tracker.update(detections)
```

<details>
<summary>run with Inference</summary>

```python
import cv2
import supervision as sv
from inference import get_model
from trackers import ByteTrackTracker

model = get_model(model_id="rfdetr-nano")
tracker = ByteTrackTracker()

cap = cv2.VideoCapture("source.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.infer(frame)[0]
    detections = sv.Detections.from_inference(result)
    detections = tracker.update(detections)
```

</details>

<details>
<summary>run with Ultralytics</summary>

```python
import cv2
import supervision as sv
from ultralytics import YOLO
from trackers import ByteTrackTracker

model = YOLO("yolo11n.pt")
tracker = ByteTrackTracker()

cap = cv2.VideoCapture("source.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update(detections)
```

</details>

## License

The code is released under the [Apache 2.0 license](https://github.com/roboflow/trackers/blob/main/LICENSE).

## Contribution

We welcome all contributions—whether it’s reporting issues, suggesting features, or submitting pull requests. Please read our [contributor guidelines](https://github.com/roboflow/trackers/blob/main/CONTRIBUTING.md) to learn about our processes and best practices.
