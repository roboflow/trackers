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
    [![python-version](https://img.shields.io/pypi/pyversions/trackers)](https://badge.fury.io/py/trackers)
    [![downloads](https://img.shields.io/pypi/dm/trackers)](https://pypistats.org/packages/trackers)

    === "pip"
        ```bash
        pip install trackers
        ```

    === "uv"
        ```bash
        uv pip install trackers
        ```

## Trackers

Trackers gives you clean, modular re-implementations of leading multi-object tracking algorithms. The package currently supports [SORT](https://arxiv.org/abs/1602.00763) and [ByteTrack](https://arxiv.org/abs/2110.06864). [OC-SORT](https://arxiv.org/abs/2203.14360) support is coming soon. For full results, see the benchmarks page. For full results, see the [benchmarks](learn/benchmarks.md) page.

|   Tracker    | Trackers package class | MOT17<br>HOTA | MOT17<br>IDF1 | MOT17<br>MOTA | SportsMOT<br>HOTA | SoccerNet<br>HOTA |
|:------------:|:----------------------:|:-------------:|:-------------:|:-------------:|:-----------------:|:-----------------:|
|     SORT     |     `SORTTracker`      |     58.4      |     69.9      |     67.2      |       70.9        |       81.6        |
|  ByteTrack   |   `ByteTrackTracker`   |   **60.1**    |   **73.2**    |   **74.1**    |     **73.0**      |     **84.0**      |
|   OC-SORT    |    `OCSORTTracker`     |       —       |       —       |       —       |         —         |         —         |

## Quickstart

With a modular design, `trackers` lets you combine object detectors from different libraries with the tracker of your choice. Here's how you can use `SORTTracker` with various detectors:

=== "inference"

    ```python hl_lines="2 5 12"
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

=== "rf-detr"

    ```python hl_lines="2 5 11"
    import supervision as sv
    from trackers import SORTTracker
    from rfdetr import RFDETRMedium

    tracker = SORTTracker()
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

=== "ultralytics"

    ```python hl_lines="2 5 12"
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

=== "transformers"

    ```python hl_lines="3 6 28"
    import torch
    import supervision as sv
    from trackers import SORTTracker
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

    tracker = SORTTracker()
    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        inputs = processor(images=frame, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        h, w, _ = frame.shape
        results = processor.post_process_object_detection(
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
