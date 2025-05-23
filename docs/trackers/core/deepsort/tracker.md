---
comments: true
---

# DeepSORT

[![arXiv](https://img.shields.io/badge/arXiv-1703.07402-b31b1b.svg)](https://arxiv.org/abs/1703.07402)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-deepsort-tracker.ipynb)

## Overview

DeepSORT extends the original [SORT](../sort/tracker.md) algorithm by integrating appearance information through a deep association metric. While maintaining the core Kalman filtering and Hungarian algorithm components from SORT, DeepSORT adds a convolutional neural network (CNN) trained on large-scale person re-identification datasets to extract appearance features from detected objects. This integration allows the tracker to maintain object identities through longer periods of occlusion, effectively reducing identity switches compared to the original SORT. DeepSORT operates with a dual-metric approach, combining motion information (Mahalanobis distance) with appearance similarity (cosine distance in feature space) to improve data association decisions. It also introduces a matching cascade that prioritizes recently seen tracks, enhancing robustness during occlusions. Most of the computational complexity is offloaded to an offline pre-training stage, allowing the online tracking component to run efficiently at approximately 20Hz, making it suitable for real-time applications while achieving competitive tracking performance with significantly improved identity preservation.


## Examples

=== "inference"

    ```python hl_lines="2 5-6 13"
    import supervision as sv
    from trackers import DeepSORTTracker, ReIDModel
    from inference import get_model

    reid_model = ReIDModel.from_timm("resnetv2_50.a1h_in1k")
    tracker = DeepSORTTracker(reid_model=reid_model)
    model = get_model(model_id="yolov11m-640")
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections, frame)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="<INPUT_VIDEO_PATH>",
        target_path="<OUTPUT_VIDEO_PATH>",
        callback=callback,
    )
    ```

=== "rf-detr"

    ```python hl_lines="2 5-6 12"
    import supervision as sv
    from trackers import DeepSORTTracker, ReIDModel
    from rfdetr import RFDETRBase

    reid_model = ReIDModel.from_timm("resnetv2_50.a1h_in1k")
    tracker = DeepSORTTracker(reid_model=reid_model)
    model = RFDETRBase()
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        detections = model.predict(frame)
        detections = tracker.update(detections, frame)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="<INPUT_VIDEO_PATH>",
        target_path="<OUTPUT_VIDEO_PATH>",
        callback=callback,
    )
    ```

=== "ultralytics"

    ```python hl_lines="2 5-6 13"
    import supervision as sv
    from trackers import DeepSORTTracker, ReIDModel
    from ultralytics import YOLO

    reid_model = ReIDModel.from_timm("resnetv2_50.a1h_in1k")
    tracker = DeepSORTTracker(reid_model=reid_model)
    model = YOLO("yolo11m.pt")
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update(detections, frame)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="<INPUT_VIDEO_PATH>",
        target_path="<OUTPUT_VIDEO_PATH>",
        callback=callback,
    )
    ```

=== "transformers"

    ```python hl_lines="3 6-7 29"
    import torch
    import supervision as sv
    from trackers import DeepSORTTracker, ReIDModel
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

    reid_model = ReIDModel.from_timm("resnetv2_50.a1h_in1k")
    tracker = DeepSORTTracker(reid_model=reid_model)
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

        detections = tracker.update(detections, frame)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="<INPUT_VIDEO_PATH>",
        target_path="<OUTPUT_VIDEO_PATH>",
        callback=callback,
    )
    ```

## API

!!! example "Install DeepSORT"

    === "CPU"
        ```bash
        pip install "trackers[reid,cpu]"
        ```

    === "CUDA 11.8"
        ```bash
        pip install "trackers[reid,cu118]"
        ```

    === "CUDA 12.4"
        ```bash
        pip install "trackers[reid,cu124]"
        ```

    === "CUDA 12.6"
        ```bash
        pip install "trackers[reid,cu126]"
        ```

    === "ROCm 6.1"
        ```bash
        pip install "trackers[reid,rocm61]"
        ```

    === "ROCm 6.2.4"
        ```bash
        pip install "trackers[reid,rocm624]"
        ```

::: trackers.core.deepsort.tracker.DeepSORTTracker
