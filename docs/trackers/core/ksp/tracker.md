---
comments: true
---

# KSP

[![IEEE](https://img.shields.io/badge/IEEE-10.1109/TPAMI.2011.21-blue.svg)](https://doi.org/10.1109/TPAMI.2011.21)
[![PDF (Unofficial)](https://img.shields.io/badge/PDF-Stanford--Preprint-red.svg)](http://vision.stanford.edu/teaching/cs231b_spring1415/papers/Berclaz-tracking.pdf)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-sort-tracker.ipynb)

## Overview

**KSP Tracker** (K-Shortest Paths Tracker) is an offline, tracking-by-detection method that formulates multi-object tracking as a global optimization problem over a directed graph. Each object detection is represented as a node, and feasible transitions between detections are modeled as edges weighted by spatial and temporal consistency. By solving a K-shortest paths problem, the tracker extracts globally optimal trajectories that span the entire sequence.

Unlike online trackers, which make frame-by-frame decisions, KSP Tracker leverages the full temporal context of a video to achieve greater robustness against occlusions, missed detections, and fragmented tracks. This makes it especially suitable for applications where high tracking accuracy is required, such as surveillance review, sports analytics, or autonomous system evaluation. However, the reliance on global optimization introduces higher computational cost and requires access to the full sequence before tracking can be performed.

## Examples

=== "inference"

    ```python hl_lines="2 6 11 15"
    import supervision as sv
    from trackers import KSPTracker
    from inference import get_model
    import numpy as np

    tracker = KSPTracker()
    model = get_model(model_id="yolo11x")
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    def get_model_detections(frame: np.ndarray):
        result = model.infer(frame)[0]
        return sv.Detections.from_inference(result)

    tracked_dets = tracker.track(
        source_path="<INPUT_VIDEO_PATH>", 
        get_model_detections=get_model_detections
    )

    frame_idx_to_dets = {i: tracked_dets[i] for i in range(len(tracked_dets))}

    def annotate_frame(frame: np.ndarray, i: int) -> np.ndarray:
        detections = frame_idx_to_dets.get(i, sv.Detections.empty())
        detections.tracker_id = detections.tracker_id or np.zeros(len(detections), dtype=int)
        labels = [f"{tid}" for tid in detections.tracker_id]
        ann = box_annotator.annotate(frame.copy(), detections)
        return label_annotator.annotate(ann, detections, labels=labels)

    sv.process_video(
        source_path="<INPUT_VIDEO_PATH>",
        target_path="<OUTPUT_VIDEO_PATH>",
        callback=annotate_frame,
    )
    ```

=== "rf-detr"

    ```python hl_lines="2 6 11 14"
    import supervision as sv
    from trackers import KSPTracker
    from rfdetr import RFDETRBase
    import numpy as np

    tracker = KSPTracker()
    model = RFDETRBase()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    def get_model_detections(frame: np.ndarray):
        return model.predict(frame)

    tracked_dets = tracker.track(
        source_path="<INPUT_VIDEO_PATH>", 
        get_model_detections=get_model_detections
    )

    frame_idx_to_dets = {i: tracked_dets[i] for i in range(len(tracked_dets))}

    def annotate_frame(frame: np.ndarray, i: int) -> np.ndarray:
        detections = frame_idx_to_dets.get(i, sv.Detections.empty())
        detections.tracker_id = detections.tracker_id or np.zeros(len(detections), dtype=int)
        labels = [f"{tid}" for tid in detections.tracker_id]
        ann = box_annotator.annotate(frame.copy(), detections)
        return label_annotator.annotate(ann, detections, labels=labels)

    sv.process_video(
        source_path="<INPUT_VIDEO_PATH>",
        target_path="<OUTPUT_VIDEO_PATH>",
        callback=annotate_frame,
    )
    ```

=== "ultralytics"

    ```python hl_lines="2 6 11 16"
    import supervision as sv
    from trackers import KSPTracker
    from ultralytics import YOLO
    import numpy as np

    tracker = KSPTracker()
    model = YOLO("yolo11m.pt")
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    def get_model_detections(frame: np.ndarray):
        result = model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        return detections[detections.class_id == 0] if not detections.is_empty() else detections

    tracked_dets = tracker.track(
        source_path="<INPUT_VIDEO_PATH>", 
        get_model_detections=get_model_detections
    )

    frame_idx_to_dets = {i: tracked_dets[i] for i in range(len(tracked_dets))}

    def annotate_frame(frame: np.ndarray, i: int) -> np.ndarray:
        detections = frame_idx_to_dets.get(i, sv.Detections.empty())
        detections.tracker_id = detections.tracker_id or np.zeros(len(detections), dtype=int)
        labels = [f"{tid}" for tid in detections.tracker_id]
        ann = box_annotator.annotate(frame.copy(), detections)
        return label_annotator.annotate(ann, detections, labels=labels)

    sv.process_video(
        source_path="<INPUT_VIDEO_PATH>",
        target_path="<OUTPUT_VIDEO_PATH>",
        callback=annotate_frame,
    )
    ```

=== "transformers"

    ```python hl_lines="3 7 13 27"
    import torch
    import supervision as sv
    from trackers import KSPTracker
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
    import numpy as np

    tracker = KSPTracker()
    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    def get_model_detections(frame: np.ndarray):
        inputs = processor(images=frame, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        h, w, _ = frame.shape
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(h, w)]),
            threshold=0.5
        )[0]

        return sv.Detections.from_transformers(results, id2label=model.config.id2label)

    tracked_dets = tracker.track(
        "<INPUT_VIDEO_PATH>", 
        get_model_detections=get_model_detections
    )

    frame_idx_to_dets = {i: tracked_dets[i] for i in range(len(tracked_dets))}

    def annotate_frame(frame: np.ndarray, i: int) -> np.ndarray:
        detections = frame_idx_to_dets.get(i, sv.Detections.empty())
        detections.tracker_id = detections.tracker_id or np.zeros(len(detections), dtype=int)
        labels = [f"{tid}" for tid in detections.tracker_id]
        ann = box_annotator.annotate(frame.copy(), detections)
        return label_annotator.annotate(ann, detections, labels=labels)

    sv.process_video(
        source_path="<INPUT_VIDEO_PATH>",
        target_path="<OUTPUT_VIDEO_PATH>",
        callback=callback,
    )
    ```

## API

::: trackers.core.ksp.tracker.KSPTracker
