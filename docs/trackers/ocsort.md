---
comments: true
description: OC-SORT (Observation-Centric SORT) enhances SORT with three mechanisms for robust tracking under occlusion and non-linear motion, improving identity consistency on crowded scenes.
---

# OC-SORT

## Overview

OC-SORT remains Simple, Online, and Real-Time like ([SORT](sort.md)) but improves robustness during occlusion and non-linear motion.
It recognizes limitations from SORT and the linear motion assumption of the Kalman filter, and adds three mechanisms to enhance tracking. These
mechanisms help having better Kalman Filter parameters after an occlusion, add a term to the association process to incorporate how consistent is the direction with the new association with respect to the tracks' previous direction and add a second-stage association step between the last observation of unmatched tracks and the unmatched observations after the usual association to attempt to recover tracks that were lost
due to object stopping or short-term occlusion.

## Comparison

For comparisons with other trackers, plus dataset context and evaluation details, see the [tracker comparison](comparison.md) page.

|  Dataset  | HOTA | IDF1 | MOTA |
| :-------: | :--: | :--: | :--: |
|   MOT17   | 61.9 | 76.1 | 76.7 |
| SportsMOT | 71.5 | 71.2 | 95.2 |
| SoccerNet | 78.6 | 72.7 | 94.5 |

## Watch It in Action

<iframe width="100%" style="aspect-ratio: 16/9;" src="https://www.youtube.com/embed/u0k2dTZ0vfs" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Algorithm

OC-SORT extends [SORT](sort.md) with three observation-centric mechanisms that address the linear motion assumption's failures during occlusion and non-linear trajectories. It retains the same Kalman filter and Hungarian algorithm backbone but adds corrections that use stored observations rather than relying solely on the filter's predicted state.

**OCM (Observation-Centric Momentum).** When a track reappears after being unmatched for several frames, the Kalman filter's velocity estimate may have drifted because it continued predicting without corrections. OCM re-computes the velocity using the time gap (`delta_t`) between the last matched observation and the current detection, then feeds this corrected velocity back into the filter. This produces a more accurate state estimate at the moment of re-association.

**OCV (Observation-Centric Velocity).** Standard IoU matching ignores direction of motion. OCV adds a velocity direction consistency term to the association cost: if a candidate match would require the track to have moved in a direction inconsistent with its recent observation history, that match is penalized. The `direction_consistency_weight` parameter controls the strength of this penalty. This helps prevent cross-identity matches when two objects are close but moving in different directions.

**ORU (Observation-Centric Re-Update).** After the primary Hungarian matching, ORU performs a second-stage re-association between the last known positions of unmatched (lost) tracks and unmatched current detections. This recovers tracks that were briefly lost due to a momentary stop or 1-2 frame occlusion, where the Kalman prediction drifted too far from the actual position for primary matching to succeed.

Together, these three mechanisms make OC-SORT effective for group dancing, sports, and other scenarios where objects follow non-linear paths, stop and restart, or are occluded in dense groups.

## Key Parameters

| Parameter                      | Purpose                                                     | Tuning guidance                                                                                                              |
| ------------------------------ | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `lost_track_buffer`            | Frames to keep an unmatched track alive before deletion.    | Higher tolerates longer occlusions but risks false re-association. 10-30 for most scenes; up to 60 for very long occlusions. |
| `track_activation_threshold`   | Minimum detection confidence to create or continue a track. | Higher reduces spurious tracks; lower catches weak detections. 0.5-0.9 typical.                                              |
| `minimum_consecutive_frames`   | Consecutive detections required to confirm a new track.     | 1 confirms immediately; 2-3 filters out single-frame false positives.                                                        |
| `minimum_iou_threshold`        | Minimum IoU to accept a track-detection match.              | Lower associates through more displacement between frames. 0.1-0.3 typical.                                                  |
| `direction_consistency_weight` | Strength of the OCV velocity direction penalty.             | 0.1-0.3 typical. Higher enforces stricter directional consistency, useful in crowded scenes.                                 |
| `delta_t`                      | Frame gap for OCM velocity re-estimation after occlusion.   | 1-3 typical. Larger values smooth velocity estimate over more frames.                                                        |

## Run on video, webcam, or RTSP stream

These examples use OpenCV for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually 0 for the default camera.

=== "Video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers import OCSORTTracker

    tracker = OCSORTTracker()
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
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections, labels=detections.tracker_id
        )

        cv2.imshow("RF-DETR + OC-SORT", annotated_frame)
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
    from trackers import OCSORTTracker

    tracker = OCSORTTracker()
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
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections, labels=detections.tracker_id
        )

        cv2.imshow("RF-DETR + OC-SORT", annotated_frame)
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
    from trackers import OCSORTTracker

    tracker = OCSORTTracker()
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
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections, labels=detections.tracker_id
        )

        cv2.imshow("RF-DETR + OC-SORT", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

## Reference

Cao, J., Pang, J., Weng, X., Khirodkar, R., and Kitani, K. (2023). Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking. CVPR. [arXiv:2203.14360](https://arxiv.org/abs/2203.14360)
