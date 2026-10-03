---
title: Hybrid-SORT — Weak-Cue Multi-Object Tracker
comments: true
description: Hybrid-SORT extends OC-SORT with weak cues — detection-confidence modeling and height-modulated IoU — for robust occlusion handling in crowded, uniform-appearance scenes, with no appearance model or training.
---

# Hybrid-SORT

## What is Hybrid-SORT?

Hybrid-SORT keeps [OC-SORT](ocsort.md)'s strong cues — box overlap and motion direction — and adds two weak ones that cost almost nothing to compute: how a track's detection confidence evolves over time, and whether candidate boxes agree in height. Strong cues become ambiguous exactly when objects overlap, and that is where the weak cues still separate them: an occluded object's detection confidence drops as it disappears and recovers as it reappears, and objects at different depths have different box heights. Like OC-SORT it needs no appearance model, no training, and no camera motion compensation, and it runs in real time on a CPU.

## How does Hybrid-SORT compare to other trackers?

Hybrid-SORT gives its largest gains in crowded scenes with frequent occlusion and uniform appearance, such as group dancing. The table below compares the motion-only trackers at their default parameters on the public validation splits, with identical YOLOX detections for every tracker. BoT-SORT runs without camera motion compensation here because no video frames are used.

!!! info "Validation-split results"

    Scores are measured on public validation splits with the detections released alongside the YOLOX-X checkpoints used in the original papers: DanceTrack val, SportsMOT val, and MOT17 half-val (second half of each training sequence, ByteTrack ablation detector). They are not the Codabench test-split numbers of the [tracker comparison](../evaluations/results.md), so compare rows within this table only.

| Tracker     | DanceTrack val HOTA | SportsMOT val HOTA | MOT17 half-val HOTA |
| :---------- | :-----------------: | :----------------: | :-----------------: |
| SORT        |        45.1         |        69.9        |        57.7         |
| ByteTrack   |        50.5         |        72.6        |        62.9         |
| OC-SORT     |        52.3         |        70.4        |        66.0         |
| BoT-SORT    |        54.0         |        72.4        |      **68.0**       |
| C-BIoU      |        53.7         |        73.1        |        67.5         |
| Hybrid-SORT |      **59.4**       |      **74.1**      |        66.7         |

On MOT17 half-val, BoT-SORT and C-BIoU score higher; Hybrid-SORT still improves on OC-SORT, the tracker it extends.

## How does Hybrid-SORT work?

Hybrid-SORT extends OC-SORT, so it keeps observation-centric re-update (ORU), which re-smooths the Kalman filter along a virtual trajectory after an occlusion, and observation-centric recovery (OCR), which matches unmatched tracks by their last observation. On top of that it adds:

**Tracklet Confidence Modeling (TCM).** Every track also runs a small Kalman filter over its detection confidence. In the first association stage, the difference between each high-confidence detection's score and the track's predicted confidence is subtracted from the matching score, weighted by `confidence_weight_first_assoc`. A detection whose confidence continues the track's recent trend is preferred over one that merely overlaps more.

**Height-Modulated IoU (HMIoU).** Association uses IoU multiplied by the overlap ratio of the two boxes' vertical extents. Two candidates with the same IoU are split by whether their heights agree, which separates people standing at different depths. HMIoU is available to every tracker as [`HMIoU`](../guides/iou.md#hmiou).

**Robust OCM.** OC-SORT's direction-consistency term compares a track's motion direction with the direction to a candidate detection using box centers. Hybrid-SORT makes the same comparison for all four box corners and averages the scores, which is less sensitive to a single noisy box edge. The `direction_consistency_weight` parameter scales this term.

**Low-confidence association.** After the first stage, detections between `0.1` and `high_conf_det_threshold` get a chance to match the tracks that are still unmatched, as in ByteTrack. This stage is penalized by the gap between the detection's score and the track's linearly extrapolated confidence, weighted by `confidence_weight_second_assoc`. Low-confidence detections keep existing tracks alive through partial occlusion but never start new tracks.

## Key Parameters

| Parameter                        | Purpose                                                                                                                                                                                                                                                                           | Tuning guidance                                                                                                              |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `lost_track_buffer`              | Frames to keep an unmatched track alive before deletion (specified in 30 FPS units, scaled proportionally by `frame_rate`).                                                                                                                                                       | Higher tolerates longer occlusions but risks false re-association. 10-30 for most scenes; up to 60 for very long occlusions. |
| `minimum_consecutive_frames`     | Consecutive detections required to confirm a new track.                                                                                                                                                                                                                           | 1 confirms immediately; 2-3 filters out single-frame false positives.                                                        |
| `minimum_iou_threshold`          | Minimum similarity (HMIoU by default) to accept a match, in every association stage.                                                                                                                                                                                              | HMIoU is never larger than IoU, so thresholds sit lower than for plain IoU. 0.1-0.25 typical.                                |
| `direction_consistency_weight`   | Strength of the four-corner direction-consistency term in the first association stage.                                                                                                                                                                                            | 0.1-0.3 typical. Higher enforces stricter directional consistency.                                                           |
| `high_conf_det_threshold`        | Detections at or above this confidence are matched first and can start new tracks. Detections between `0.1` and this threshold only recover existing tracks; detections at or below `0.1` are never associated. Every detection is returned, unmatched ones with `tracker_id=-1`. | 0.5-0.7 typical. Lower values let more detections start tracks.                                                              |
| `confidence_weight_first_assoc`  | Weight of the confidence-consistency penalty for high-confidence detections (Kalman-predicted track confidence).                                                                                                                                                                  | 0.5-1.5 typical; `0` disables it. Has no effect when every detection has the same confidence, such as ground-truth boxes.    |
| `confidence_weight_second_assoc` | Weight of the confidence-consistency penalty for low-confidence detections (linearly extrapolated track confidence).                                                                                                                                                              | 0.5-1.5 typical; `0` disables it.                                                                                            |
| `delta_t`                        | Frame lookback used to compute each track's per-corner velocities.                                                                                                                                                                                                                | 1-3 typical. Larger values smooth velocity estimate over more frames.                                                        |

!!! warning "Frame input is ignored by Hybrid-SORT"

    `HybridSORTTracker.update()` accepts `frame` for API consistency with other trackers, but Hybrid-SORT does not use image/frame pixels. If you pass `frame` with a non-`None` value, the tracker emits a `UserWarning` and ignores it.

## Run on video, webcam, or RTSP stream

These examples use OpenCV for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually 0 for the default camera.

=== "CLI"

    Run Hybrid-SORT on a video without writing any Python. See the [CLI reference](../guides/cli.md) for every argument, including `--source 0` for a webcam or an `rtsp://` URL for a stream.

    ```bash
    trackers track \
        --source <SOURCE_VIDEO_PATH> \
        --tracker hybridsort \
        --output.video output.mp4
    ```

=== "Video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers import HybridSORTTracker

    tracker = HybridSORTTracker()
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

        cv2.imshow("RF-DETR + Hybrid-SORT", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "Webcam"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers import HybridSORTTracker

    tracker = HybridSORTTracker()
    model = RFDETRMedium()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_capture = cv2.VideoCapture("<WEBCAM_INDEX>")
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open webcam")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb)
        detections = tracker.update(detections)

        annotated_frame = box_annotator.annotate(frame_bgr, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=detections.tracker_id)

        cv2.imshow("RF-DETR + Hybrid-SORT", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "RTSP"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers import HybridSORTTracker

    tracker = HybridSORTTracker()
    model = RFDETRMedium()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_capture = cv2.VideoCapture("<RTSP_STREAM_URL>")
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open RTSP stream")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb)
        detections = tracker.update(detections)

        annotated_frame = box_annotator.annotate(frame_bgr, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=detections.tracker_id)

        cv2.imshow("RF-DETR + Hybrid-SORT", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

## Reference

Yang, M., Han, G., Yan, B., Zhang, W., Qi, J., Lu, H., and Wang, D. (2024). Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking. AAAI. [arXiv:2308.00783](https://arxiv.org/abs/2308.00783)
