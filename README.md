[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VT_FYIe3kborhWrfKKBqqfR0EjQeQNiO?usp=sharing)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

<div align="center">
    <h1 align="center">trackers</h1>
    <img width="200" src="https://github.com/user-attachments/assets/3fce0d37-dc1a-4b1f-b9ec-ca6ccf3a33f1" alt="make sense logo">
    </br>
    <p>coming: when it's ready...</p>
</div>

## Hello

A unified library for object tracking featuring clean room re-implementations of leading multi-object tracking algorithms.

## Installation

Pip install the `trackers` package in a [**Python>=3.9**](https://www.python.org/) environment.

```bash
pip install trackers
```

<details>
<summary>Install from source</summary>

<br>

By installing `trackers` from source, you can explore the most recent features and enhancements that have not yet been officially released. Please note that these updates are still in development and may not be as stable as the latest published release.

```bash
pip install git+https://github.com/roboflow/trackers.git
```

</details>

## Quickstart

With a modular design, `trackers` lets you combine object detectors from different libraries (such as `ultralytics`, `transformers`, or `mmdetection`) with the tracker of your choice.

```python
import supervision as sv
from rfdetr import RFDETRBase
from trackers.sort_tracker import SORTTracker

model = RFDETRBase()
tracker = SORTTracker()
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

def callback(frame, _):
    detections = model.predict(frame)
    detections = tracker.update(detections)
    return annotator.annotate(frame, detections, detections.tracker_id)

sv.process_video(
    source_path=<SOURCE_VIDEO_PATH>,
    target_path=<TARGET_VIDEO_PATH>,
    callback=callback,
)
```

<details>
<summary>Ultralytics YOLO model </summary>

<br>

```python
import supervision as sv
from ultralytics import YOLO
from trackers.sort_tracker import SORTTracker

model = YOLO("yolo11n.pt")
tracker = SORTTracker()
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

def callback(frame, _):
    detections = model.predict(frame)[0]
    formatted_detections = sv.Detections.from_ultralytics(detections)
    detections = tracker.update(formatted_detections)
    return annotator.annotate(frame, detections, detections.tracker_id)

sv.process_video(
    source_path=<SOURCE_VIDEO_PATH>,
    target_path=<TARGET_VIDEO_PATH>,
    callback=callback,
)
```


</details>

https://github.com/user-attachments/assets/910490b3-32a0-4b7f-8b84-5b50aa83e004

## License

The code is released under the [Apache 2.0 license](https://github.com/roboflow/trackers/blob/main/LICENSE).

## Contribution

We welcome all contributions—whether it’s reporting issues, suggesting features, or submitting pull requests. Please read our [contributor guidelines](https://github.com/roboflow/trackers/blob/main/CONTRIBUTING.md) to learn about our processes and best practices.
