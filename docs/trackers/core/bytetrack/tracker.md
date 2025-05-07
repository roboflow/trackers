---
comments: true
---

# ByteTrack

[![arXiv](https://img.shields.io/badge/arXiv-2110.06864-b31b1b.svg)](https://arxiv.org/pdf/2110.06864)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1C7XyJV8_V6HJwx20E838oQ03F5DLauuC?usp=sharing) <!-- Change the URL to custom Roboflow one before merging --> 

## Overview

ByteTrack presents a simple and generic association method which associates almost every detection box instead of only the high probability ones. Low score boxes are typically occluded object, so leaving out this objects of the tracking will result in a fatal failure, but because they aren't clearly seen in the image we cannot trust on it's appearance. 

That's why the proposed method consists in 2 key steps. The first step will associate the high score detections to the existing tracks using a chosen similarity metric that can be either IoU or based in appearance features. The second step will associate the low score detections to the trackers that didn't match in the previous step using IoU distance. In addition to this, we added parametrized thresholds for accepting the matches only if the similarity is higher to the corresponding threshold. Finally it starts new tracks with the high score detections that didn't match in step 1. Just like [SORT](../sort/tracker.md) and [DeepSORT](../deepsort/tracker.md) this method combines Kalman Filters for having a motion model in order to match low score boxes and the Hungarian algorithm for calculating the optimal associations.

While calculating the appearance features with a Convolutional Neural Network might be slower than only comparing IoU, it makes it possible to track unpredictable trajectories and objects that dissapear and reappear in the scene, while also being suitable for real time tracking.

BytTrack is independent on the object detector and feature extractor network so it can be used for any tracking task as long as the user provides the adequate model. 


## Examples

=== "rf-detr"

    ```python hl_lines="2 5-8 15"
    import supervision as sv
    from rfdetr import RFDETRBase
    from trackers import ByteTrackTracker
    from trackers.core.deepsort.feature_extractor import DeepSORTFeatureExtractor

    model = RFDETRBase(device="cuda")  # Load the Object Detector
    feature_extractor = DeepSORTFeatureExtractor.from_timm(
        model_name="mobilenetv4_conv_small.e1200_r224_in1k",
    )
    tracker = ByteTrackTracker(feature_extractor=feature_extractor)

    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        # Obtain bounding box predictions from RF-DETR
        detections = model.predict(frame, threshold=0.5)

        # Update tracker with new detections and retrieve updated IDs
        detections = tracker.update(detections, frame)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="input.mp4",
        target_path="output.mp4",
        callback=callback,
    )
    ```
=== "inference"

    ```python hl_lines="2 5-8 15"
    import supervision as sv
    from trackers import DeepSORTFeatureExtractor
    from trackers import ByteTrackTracker
    from inference import get_model

    feature_extractor = DeepSORTFeatureExtractor.from_timm(
        model_name="mobilenetv4_conv_small.e1200_r224_in1k"
    )
    tracker = ByteTrackTracker(feature_extractor=feature_extractor)
    model = get_model(model_id="yolov11m-640")
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections, frame)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="input.mp4",
        target_path="output.mp4",
        callback=callback,
    )
    ```


=== "ultralytics"

    ```python hl_lines="2 5-8 15"
    import supervision as sv
    from trackers import DeepSORTFeatureExtractor
    from trackers import ByteTrackTracker

    from ultralytics import YOLO

    feature_extractor = DeepSORTFeatureExtractor.from_timm(
        model_name="mobilenetv4_conv_small.e1200_r224_in1k"
    )
    tracker = ByteTrackTracker(feature_extractor=feature_extractor)
    model = YOLO("yolo11m.pt")
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update(detections, frame)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="input.mp4",
        target_path="output.mp4",
        callback=callback,
    )
    ```

=== "transformers"

    ```python hl_lines="3 6-9 31"
    import torch
    import supervision as sv
    from trackers import DeepSORTFeatureExtractor
    from trackers import ByteTrackTracker
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

    feature_extractor = DeepSORTFeatureExtractor.from_timm(
        model_name="mobilenetv4_conv_small.e1200_r224_in1k"
    )
    tracker = ByteTrackTracker(feature_extractor=feature_extractor)
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
        source_path="input.mp4",
        target_path="output.mp4",
        callback=callback,
    )
    ```

## Usage

::: trackers.core.bytetrack.tracker.ByteTrackTracker
