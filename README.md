[![version](https://badge.fury.io/py/trackers.svg)](https://badge.fury.io/py/trackers)
[![downloads](https://img.shields.io/pypi/dm/trackers)](https://pypistats.org/packages/trackers)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/trackers/blob/main/LICENSE.md)
[![python-version](https://img.shields.io/pypi/pyversions/trackers)](https://badge.fury.io/py/trackers)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

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

|   Algorithm   |                              Trackers API                                       | MOT17 HOTA | MOT17 IDF1  | MOT17 MOTA | SportsMOT HOTA | SoccerNet HOTA |
|:-------------:|:-------------------------------------------------------------------------------:|:----------:|:-----------:|:----------:|:--------------:|:--------------:|
|     SORT      |      [`SORTTracker`](https://trackers.roboflow.com/develop/trackers/sort/)      |    58.4    |    69.9     |    67.2    |      70.9      |      81.6      |
|   ByteTrack   | [`ByteTrackTracker`](https://trackers.roboflow.com/develop/trackers/bytetrack/) |  **60.1**  |  **73.2**   |  **74.1**  |    **73.0**    |    **84.0**    |
|    OC-SORT    |                                 `OCSORTTracker`                                 |     —      |      —      |     —      |       —        |       —        |

## Quickstart

With a modular design, Trackers lets you combine object detectors from different libraries with the tracker of your choice. Here's how you can use ByteTrack with various detectors. These examples use OpenCV for decoding and display. Replace `<SOURCE_VIDEO_PATH>` with your input.

```python
import cv2
import supervision as sv
from rfdetr import RFDETRMedium
from trackers import ByteTrack

tracker = ByteTrack()
model = RFDETRMedium()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

video_capture = cv2.VideoCapture("<SOURCE_VIDEO_PATH>")
if not video_capture.isOpened():
    raise RuntimeError("Failed to open video source")

while True:
    success, frame_bgr = video_capture.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = model.predict(frame_rgb)
    detections = tracker.update(detections)

    annotated_frame = box_annotator.annotate(frame_bgr, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=detections.tracker_id)

    cv2.imshow("RF-DETR + ByteTrack", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
```

<details>
<summary>run with Inference</summary>

<br>

```python
import cv2
import supervision as sv
from inference import get_model
from trackers import ByteTrack

tracker = ByteTrack()
model = get_model(model_id="rfdetr-medium")

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

video_capture = cv2.VideoCapture("<SOURCE_VIDEO_PATH>")
if not video_capture.isOpened():
    raise RuntimeError("Failed to open video source")

while True:
    success, frame_bgr = video_capture.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = model.infer(frame_rgb)[0]
    detections = sv.Detections.from_inference(result)
    detections = tracker.update(detections)

    annotated_frame = box_annotator.annotate(frame_bgr, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=detections.tracker_id)

    cv2.imshow("Inference + ByteTrack", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
```
</details>

<details>
<summary>run with Ultralytics</summary>

<br>

```python
import cv2
import supervision as sv
from ultralytics import YOLO
from trackers import ByteTrack

tracker = ByteTrack()
model = YOLO("yolo26m.pt")

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

video_capture = cv2.VideoCapture("<SOURCE_VIDEO_PATH>")
if not video_capture.isOpened():
    raise RuntimeError("Failed to open video source")

while True:
    success, frame_bgr = video_capture.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = model(frame_rgb)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update(detections)

    annotated_frame = box_annotator.annotate(frame_bgr, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=detections.tracker_id)

    cv2.imshow("Ultralytics + ByteTrack", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
```

</details>

<details>
<summary>run with Transformers</summary>

<br>

```python
import torch
import cv2
import supervision as sv
from trackers import ByteTrack
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

tracker = ByteTrack()
processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

video_capture = cv2.VideoCapture("<SOURCE_VIDEO_PATH>")
if not video_capture.isOpened():
    raise RuntimeError("Failed to open video source")

while True:
    success, frame_bgr = video_capture.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    h, w = frame_bgr.shape[:2]
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[h, w]]),
        threshold=0.5
    )[0]

    detections = sv.Detections.from_transformers(
        transformers_results=results,
        id2label=model.config.id2label
    )
    detections = tracker.update(detections)

    annotated_frame = box_annotator.annotate(frame_bgr, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=detections.tracker_id)

    cv2.imshow("Transformers + ByteTrack", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
```

</details>

## License

The code is released under the [Apache 2.0 license](https://github.com/roboflow/trackers/blob/main/LICENSE).

## Contribution

We welcome all contributions—whether it’s reporting issues, suggesting features, or submitting pull requests. Please read our [contributor guidelines](https://github.com/roboflow/trackers/blob/main/CONTRIBUTING.md) to learn about our processes and best practices.
