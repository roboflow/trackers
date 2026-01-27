<div align="center">
    <h1 align="center">trackers</h1>
    <img width="200" src="https://raw.githubusercontent.com/roboflow/trackers/refs/heads/main/docs/assets/logo-trackers-violet.svg" alt="trackers logo">
</div>

Trackers gives you clean, modular re-implementations of leading multi-object tracking algorithms released under the permissive Apache 2.0 license. You combine them with any detection model you already use.

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

## Tracking Algorithms

Trackers gives you clean, modular re-implementations of leading multi-object tracking algorithms. The package currently supports [SORT](https://arxiv.org/abs/1602.00763) and [ByteTrack](https://arxiv.org/abs/2110.06864). [OC-SORT](https://arxiv.org/abs/2203.14360) support is coming soon. For full results, see the [benchmarks](https://trackers.roboflow.com/develop/learn/benchmarks/) page.

|   Tracker    | Trackers package class | MOT17<br><small>HOTA</small> | MOT17<br><small>IDF1</small> | MOT17<br><small>MOTA</small> | SportsMOT<br><small>HOTA</small> | SoccerNet<br><small>HOTA</small> |
|:------------:|:----------------------:|:----------------------------:|:----------------------------:|:----------------------------:|:--------------------------------:|:--------------------------------:|
|     SORT     |     `SORTTracker`      |            58.4              |            69.9              |            67.2              |              70.9                |              81.6                |
|  ByteTrack   |   `ByteTrackTracker`   |          **60.1**            |          **73.2**            |          **74.1**            |            **73.0**              |            **84.0**              |
|   OC-SORT    |    `OCSORTTracker`     |              —               |              —               |              —               |                —                 |                —                 |

## Quickstart

With a modular design, `trackers` lets you combine object detectors from different libraries with the tracker of your choice. Here's how you can use `SORTTracker` with various detectors:

```python
import supervision as sv
from trackers import ByteTrackTracker
from rfdetr import RFDETRMedium

tracker = ByteTrackTracker()
model = RFDETRMedium()

annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

def callback(frame, _):
    detections = model.predict(frame)
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
