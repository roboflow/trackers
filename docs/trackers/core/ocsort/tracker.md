---
comments: true
---

# OC-SORT

[![arXiv](https://img.shields.io/badge/arXiv-2203.14360-b31b1b.svg)](https://arxiv.org/abs/2203.14360)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-sort-tracker.ipynb)

## Overview

OC-SORT remains Simple, Online, and Real-Time ([SORT](../sort/tracker.md)) but improves robustness during occlusion and non-linear motion.
It recognizes limitations from SORT and the linear motion assumption of the Kalman filter, and adds three
mechanisms to enhance tracking:
    1. Observation-Centre Re-Update (ORU): runs a predict-update loop with a 'virtual trajectory'
        depending on the last observation and new observation when a track is re-activated after being lost.
    2. Observation-Centric Momentum (OCM): incorporate the direction consistency of tracks in the cost matrix for the association.
    3. Observation-centric Recovery (OCR): a second-stage association step between the last observation of unmatched tracks
        to the unmatched observations after the usual association. It attempts to recover tracks that were lost
        due to object stopping or short-term occlusion. Uses only IoU.

## Examples

=== "inference"

    ```python hl_lines="2 5 12"
    import supervision as sv
    from trackers import OCSORTTracker
    from inference import get_model

    tracker = OCSORTTracker()
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
    from trackers import OCSORTTracker
    from rfdetr import RFDETRMedium

    tracker = OCSORTTracker()
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
    from trackers import OCSORTTracker
    from ultralytics import YOLO

    tracker = OCSORTTracker()
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
    from trackers import OCSORTTracker
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

    tracker = OCSORTTracker()
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


## API

::: trackers.core.sort.tracker.OCSORTTracker
