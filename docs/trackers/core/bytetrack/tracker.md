---
comments: true
---

# ByteTrack

[![arXiv](https://img.shields.io/badge/arXiv-2110.06864-b31b1b.svg)](https://arxiv.org/pdf/2110.06864)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-bytetrack-tracker.ipynb)

## Overview

ByteTrack presents a simple and generic association method which overcomes the limitation of only associating high confidence detections. Low score boxes are typically occluded objects, so skipping these objects from association would result in lost and fragmented tracks.
That's why the proposed method consists of two key steps. The first step will associate the high confidence detections (subject to a threshold) with the existing tracks using Intersection over Union (IoU). The second step will associate the low confidence detections with the tracks that didn't match in the previous step using IoU. In addition to this, we added parametrized thresholds for accepting the matches only if the similarity between the tracked box and the detection is higher than the corresponding threshold. Finally, it starts new tracks with the high confidence detections that didn't match in step 1.

Just like [SORT](../sort/tracker.md), this method combines the Kalman Filter as motion model and the Hungarian algorithm for calculating the associations between the predicted position of the track and the detection. This tracker also keeps the simplicity and efficiency of [SORT](../sort/tracker.md) while improving tracking capabilities for occluded objects, leveraging all detections to enhance multi-object tracking.


## Examples
=== "rf-detr"

    ```python hl_lines="3 7 13"
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers import ByteTrackTracker

    model = RFDETRMedium(device ="cuda")

    tracker = ByteTrackTracker()

    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        detections = model.predict(frame)
        detections = tracker.update(detections)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="input.mp4",
        target_path="output.mp4",
        callback=callback,
    )
    ```
=== "inference"

    ```python hl_lines="2 5 12"
    import supervision as sv
    from trackers import ByteTrackTracker
    from inference import get_model

    tracker = ByteTrackTracker()
    model = get_model(model_id="rfdetr-medium")
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="input.mp4",
        target_path="output.mp4",
        callback=callback,
    )
    ```

=== "ultralytics"

    ```python hl_lines="2 5 12"
    import supervision as sv
    from trackers import ByteTrackTracker
    from ultralytics import YOLO

    tracker = ByteTrackTracker()
    model = YOLO("yolo11m.pt")
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update(detections)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="input.mp4",
        target_path="output.mp4",
        callback=callback,
    )
    ```

=== "transformers"

    ```python hl_lines="3 6 28"
    import torch
    import supervision as sv
    from trackers import ByteTrackTracker
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

    tracker = ByteTrackTracker()
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
        source_path="input.mp4",
        target_path="output.mp4",
        callback=callback,
    )
    ```


## Usage

::: trackers.core.bytetrack.tracker.ByteTrackTracker
