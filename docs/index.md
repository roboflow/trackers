---
comments: true
---

<div align="center">

<img src="assets/logo-trackers-violet.svg" alt="Trackers Logo" width="200" height="200">

</div>

Trackers gives you clean, modular re-implementations of leading multi-object tracking algorithms released under the permissive Apache 2.0 license. You combine them with any detection model you already use.

## Install

You can install and use `trackers` in a [**Python>=3.10**](https://www.python.org/) environment. For detailed installation instructions, including installing from source and setting up a local development environment, check out our [install](learn/install.md) page.

!!! example "Installation"
    [![version](https://badge.fury.io/py/trackers.svg)](https://badge.fury.io/py/trackers)
    [![downloads](https://img.shields.io/pypi/dm/trackers)](https://pypistats.org/packages/trackers)
    [![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/trackers/blob/main/LICENSE.md)
    [![python-version](https://img.shields.io/pypi/pyversions/trackers)](https://badge.fury.io/py/trackers)

    === "pip"
        ```bash
        pip install trackers
        ```

    === "uv"
        ```bash
        uv pip install trackers
        ```

## Tutorials

<div class="grid cards" markdown>

-   **How to Track Objects with SORT**

    ---

    [![](https://storage.googleapis.com/com-roboflow-marketing/trackers/assets/sort-sample.png)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-sort-tracker.ipynb)

    End-to-end example showing how to run RF-DETR detection with the SORT tracker.

    [:simple-googlecolab: Run Google Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-sort-tracker.ipynb)

-   **How to Track Objects with ByteTrack**

    ---

    [![](https://storage.googleapis.com/com-roboflow-marketing/trackers/assets/bytetrack-sample.png)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-bytetrack-tracker.ipynb)

    End-to-end example showing how to run RF-DETR detection with the ByteTrack tracker.

    [:simple-googlecolab: Run Google Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-bytetrack-tracker.ipynb)

</div>


## Tracking Algorithms

Trackers gives you clean, modular re-implementations of leading multi-object tracking algorithms. The package currently supports [SORT](https://arxiv.org/abs/1602.00763) and [ByteTrack](https://arxiv.org/abs/2110.06864). [OC-SORT](https://arxiv.org/abs/2203.14360) support is coming soon. For comparisons, see the [tracker comparison](trackers/comparison.md) page.

|  Algorithm  |                Trackers API                 | MOT17 HOTA | MOT17 IDF1 | MOT17 MOTA | SportsMOT HOTA | SoccerNet HOTA |
|:-----------:|:-------------------------------------------:|:----------:|:----------:|:----------:|:--------------:|:--------------:|
|    SORT     |      [`SORTTracker`](trackers/sort.md)      |    58.4    |    69.9    |    67.2    |      70.9      |      81.6      |
|  ByteTrack  | [`ByteTrackTracker`](trackers/bytetrack.md) |  **60.1**  |  **73.2**  |  **74.1**  |    **73.0**    |    **84.0**    |
|   OC-SORT   |               `OCSORTTracker`               |     —      |     —      |     —      |       —        |       —        |

## Quickstart

With a modular design, Trackers lets you combine object detectors from different libraries with the tracker of your choice. Here's how you can use ByteTrack with various detectors. These examples use OpenCV for decoding and display. Replace `<SOURCE_VIDEO_PATH>` with your input.

=== "RF-DETR"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers import ByteTrackTracker

    tracker = ByteTrackTracker()
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

=== "Inference"

    ```python
    import cv2
    import supervision as sv
    from inference import get_model
    from trackers import ByteTrackTracker

    tracker = ByteTrackTracker()
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

=== "Ultralytics"

    ```python
    import cv2
    import supervision as sv
    from ultralytics import YOLO
    from trackers import ByteTrackTracker

    tracker = ByteTrackTracker()
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

=== "Transformers"

    ```python
    import torch
    import cv2
    import supervision as sv
    from trackers import ByteTrackTracker
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    tracker = ByteTrackTracker()
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
