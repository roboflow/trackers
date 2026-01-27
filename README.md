<div align="center">
    <h1 align="center">trackers</h1>
    <img width="200" src="https://raw.githubusercontent.com/roboflow/trackers/refs/heads/main/docs/assets/logo-trackers-violet.svg" alt="trackers logo">

[![version](https://badge.fury.io/py/trackers.svg)](https://badge.fury.io/py/trackers)
[![downloads](https://img.shields.io/pypi/dm/trackers)](https://pypistats.org/packages/trackers)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/trackers/blob/main/LICENSE.md)
[![python-version](https://img.shields.io/pypi/pyversions/trackers)](https://badge.fury.io/py/trackers)

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VT_FYIe3kborhWrfKKBqqfR0EjQeQNiO?usp=sharing)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)
</div>

## Installation

Pip install the `trackers` package in a [**Python>=3.10**](https://www.python.org/) environment.

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

## Supported Trackers & Benchmarks

`trackers` gives you clean, modular re-implementations of leading MOT algorithms released under the permissive **Apache 2.0 license**.

All numbers below were measured using the official **TrackEval** library across three diverse MOT evaluation datasets (MOT17, SportsMOT, SoccerNet). Table below shows only a selection of the most representative metrics. For full tables (including all metrics on every dataset), detailed dataset descriptions, metric explanations and reproducibility instructions, check out our benchmarks page.

| Tracker       | MOT17 HOTA | MOT17 IDF1 | MOT17 MOTA | SportsMOT HOTA | SoccerNet HOTA |
|---------------|------------|------------|------------|----------------|----------------|
| SORT          | 58.4       | 69.9       | 67.2       | 70.9           | 81.6           |
| **ByteTrack** | **60.1**   | **73.2**   | **74.1**   | **73.0**       | **84.0**       |
| OC-SORT       | —          | —          | —          | —              | —              |

<details>
<summary>metric cheat-sheet</summary>

<br>

- **HOTA** – overall tracking quality (best single-number comparison across detection & association)
- **IDF1** – identity consistency over long periods (crucial for sports analytics, trajectory analysis, re-identification)
- **MOTA** – mainly detection coverage & basic frame-to-frame errors (less sensitive to ID switches)

</details>

## Quickstart

With a modular design, `trackers` lets you combine object detectors from different libraries with the tracker of your choice. Here's how you can use `SORTTracker` with various detectors:


```python
import supervision as sv
from trackers import SORTTracker
from rfdetr import RFDETRMedium

tracker = SORTTracker()
model = RFDETRMedium(device="cuda")
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

def callback(frame, _):
    detections = model.predict(frame, threshold=NMS_THRESHOLD)
    detections = tracker.update(detections)
    return annotator.annotate(frame, detections, labels=detections.tracker_id)

sv.process_video(
    source_path="<INPUT_VIDEO_PATH>",
    target_path="<OUTPUT_VIDEO_PATH>",
    callback=callback,
)
```

<details>
<summary>run with <code>inference</code></summary>

<br>

```python
import supervision as sv
from trackers import SORTTracker
from inference import get_model

tracker = SORTTracker()
model = get_model(model_id="rfdetr-medium")
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

def callback(frame, _):
    result = model.infer(frame)[0]
    detections = sv.Detections.from_inference(result)
    detections = tracker.update(detections)
    return annotator.annotate(frame, detections, labels=detections.tracker_id)

sv.process_video(
    source_path="<INPUT_VIDEO_PATH>",
    target_path="<OUTPUT_VIDEO_PATH>",
    callback=callback,
)
```
</details>

<details>
<summary>run with <code>ultralytics</code></summary>

<br>

```python
import supervision as sv
from trackers import SORTTracker
from ultralytics import YOLO

tracker = SORTTracker()
model = YOLO("yolo11m.pt")
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

def callback(frame, _):
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update(detections)
    return annotator.annotate(frame, detections, labels=detections.tracker_id)

sv.process_video(
    source_path="<INPUT_VIDEO_PATH>",
    target_path="<OUTPUT_VIDEO_PATH>",
    callback=callback,
)
```

</details>

<details>
<summary>run with <code>transformers</code></summary>

<br>

```python
import torch
import supervision as sv
from trackers import SORTTracker
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

tracker = SORTTracker()
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

def callback(frame, _):
    inputs = image_processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    h, w, _ = frame.shape
    results = image_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(h, w)]),
        threshold=0.5
    )[0]

    detections = sv.Detections.from_transformers(
        transformers_results=results,
        id2label=model.config.id2label
    )

    detections = tracker.update(detections)
    return annotator.annotate(frame, detections, labels=detections.tracker_id)

sv.process_video(
    source_path="<INPUT_VIDEO_PATH>",
    target_path="<OUTPUT_VIDEO_PATH>",
    callback=callback,
)
```

</details>

## License

The code is released under the [Apache 2.0 license](https://github.com/roboflow/trackers/blob/main/LICENSE).

## Contribution

We welcome all contributions—whether it’s reporting issues, suggesting features, or submitting pull requests. Please read our [contributor guidelines](https://github.com/roboflow/trackers/blob/main/CONTRIBUTING.md) to learn about our processes and best practices.
