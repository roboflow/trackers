---
comments: true
---

# BoT-SORT

## Overview

BoT-SORT extends ([ByteTrack](../bytetrack/tracker.md)) with camera motion compensation (CMC) to handle moving cameras and dynamic scenes. It uses the same two-stage association strategy: first matching high-confidence detections, then recovering low-confidence ones, but applies a transformation estimated from optical flow to warp Kalman filter predictions before matching. This compensates for camera ego-motion that would otherwise degrade IoU overlap and cause missed associations. Additionally, BoT-SORT fuses IoU similarity with detection confidence scores during association, splits tracks into confirmed, unconfirmed, and lost categories for more precise lifecycle management, and only activates new tracklets after a two-frame confirmation (except on the very first frame). The result is more stable tracking under camera motion while retaining real-time speed.

## Comparison

For comparisons with other trackers, plus dataset context and evaluation details, see the [tracker comparison](comparison.md) page.

|  Dataset  | HOTA | IDF1 | MOTA |
| :-------: | :--: | :--: | :--: |
|   MOT17   | 63.7 | 78.7| 79.2 |
| SportsMOT | 73.8 | 73.4 |96.9 |
| SoccerNet | 84.5 | 79.3 | 96.6 |

## Run on video, webcam, or RTSP stream

These examples use `opencv-python` for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually 0 for the default camera.

!!! warning

    Unlike other trackers, `BoTSORTTracker.update()` requires the current
    video frame for camera motion compensation. Pass the BGR frame as the
    second argument.

=== "Video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers.core.botsort import BoTSORTTracker

    tracker = BoTSORTTracker()
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
        detections = tracker.update(detections, frame_bgr)

        annotated_frame = box_annotator.annotate(frame_bgr, detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame,
            detections,
            labels=detections.tracker_id,
        )

        cv2.imshow("RF-DETR + BoT-SORT", annotated_frame)
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
    from trackers.core.botsort import BoTSORTTracker

    tracker = BoTSORTTracker()
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
        detections = tracker.update(detections, frame_bgr)

        annotated_frame = box_annotator.annotate(frame_bgr, detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame,
            detections,
            labels=detections.tracker_id,
        )

        cv2.imshow("RF-DETR + BoT-SORT", annotated_frame)
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
    from trackers.core.botsort import BoTSORTTracker

    tracker = BoTSORTTracker()
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
        detections = tracker.update(detections, frame_bgr)

        annotated_frame = box_annotator.annotate(frame_bgr, detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame,
            detections,
            labels=detections.tracker_id,
        )

        cv2.imshow("RF-DETR + BoT-SORT", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```
