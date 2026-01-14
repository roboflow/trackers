---
comments: true
---

# SORT

[![arXiv](https://img.shields.io/badge/arXiv-1602.00763-b31b1b.svg)](https://arxiv.org/abs/1602.00763)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-sort-tracker.ipynb)

## Overview

SORT (Simple Online and Realtime Tracking) is a lean, tracking-by-detection method that combines a Kalman filter for motion prediction with the Hungarian algorithm for data association. It uses object detections—commonly from a high-performing CNN-based detector—as its input, updating each tracked object’s bounding box based on linear velocity estimates. Because SORT relies on minimal appearance modeling (only bounding box geometry is used), it is extremely fast and can run comfortably at hundreds of frames per second. This speed and simplicity make it well suited for real-time applications in robotics or surveillance, where rapid, approximate solutions are essential. However, its reliance on frame-to-frame matching makes SORT susceptible to ID switches and less robust during long occlusions, since there is no built-in re-identification module.

## Examples

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
    from rfdetr import RFDETRBase

    tracker = SORTTracker()
    model = RFDETRBase()
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

## API

::: trackers.core.sort.tracker.SORTTracker
