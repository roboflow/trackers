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

## Hello

`trackers` is a unified library offering clean room re-implementations of leading multi-object tracking algorithms. Its modular design allows you to easily swap trackers and integrate them with object detectors from various libraries like `inference`, `ultralytics`, or `transformers`.

<div align="center">
  <table>
    <thead>
      <tr>
        <th>Tracker</th>
        <th>Paper</th>
        <th>MOTA</th>
        <th>Year</th>
        <th>Status</th>
        <th>Colab</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>SORT</td>
        <td><a href="https://arxiv.org/abs/1602.00763"><img src="https://img.shields.io/badge/arXiv-1602.00763-b31b1b.svg" alt="arXiv"></a></td>
        <td>74.6</td>
        <td>2016</td>
        <td>✅</td>
        <td><a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-sort-tracker.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a></td>
      </tr>
      <tr>
        <td>DeepSORT</td>
        <td><a href="https://arxiv.org/abs/1703.07402"><img src="https://img.shields.io/badge/arXiv-1703.07402-b31b1b.svg" alt="arXiv"></a></td>
        <td>75.4</td>
        <td>2017</td>
        <td>✅</td>
        <td><a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-deepsort-tracker.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a></td>
      </tr>
      <tr>
        <td>ByteTrack</td>
        <td><a href="https://arxiv.org/abs/2110.06864"><img src="https://img.shields.io/badge/arXiv-2110.06864-b31b1b.svg" alt="arXiv"></a></td>
        <td>77.8</td>
        <td>2021</td>
        <td>🚧</td>
        <td>🚧</td>
      </tr>
      <tr>
        <td>OC-SORT</td>
        <td><a href="https://arxiv.org/abs/2203.14360"><img src="https://img.shields.io/badge/arXiv-2203.14360-b31b1b.svg" alt="arXiv"></a></td>
        <td>75.9</td>
        <td>2022</td>
        <td>🚧</td>
        <td>🚧</td>
      </tr>
      <tr>
        <td>BoT-SORT</td>
        <td><a href="https://arxiv.org/abs/2206.14651"><img src="https://img.shields.io/badge/arXiv-2206.14651-b31b1b.svg" alt="arXiv"></a></td>
        <td>77.8</td>
        <td>2022</td>
        <td>🚧</td>
        <td>🚧</td>
      </tr>
    </tbody>
  </table>
</div>

https://github.com/user-attachments/assets/eef9b00a-cfe4-40f7-a495-954550e3ef1f

## Installation

Pip install the `trackers` package in a [**Python>=3.9**](https://www.python.org/) environment.

```bash
pip install trackers
```

<details>
<summary>install from source</summary>

<br>

By installing `trackers` from source, you can explore the most recent features and enhancements that have not yet been officially released. Please note that these updates are still in development and may not be as stable as the latest published release.

```bash
pip install git+https://github.com/roboflow/trackers.git
```

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
