---
comments: true
description: SORT (Simple Online and Realtime Tracking) uses a Kalman filter and Hungarian algorithm to track objects in real time using only bounding-box geometry — fast, lightweight, and easy to integrate.
---

# SORT

## Overview

SORT is a classic online, tracking-by-detection method that predicts object motion with a Kalman filter and matches predicted tracks to detections using the Hungarian algorithm based on Intersection over Union (IoU). The tracker uses only geometric cues from bounding boxes, without appearance features, so it runs extremely fast and scales to hundreds of frames per second on typical hardware. Detections from a strong CNN detector feed SORT, which updates each track’s state via a constant velocity motion model and prunes stale tracks. Because SORT lacks explicit re-identification or appearance cues, it can suffer identity switches and fragmented tracks under long occlusions or heavy crowding.

## Comparison

For comparisons with other trackers, plus dataset context and evaluation details, see the [tracker comparison](comparison.md) page.

|  Dataset  | HOTA | IDF1 | MOTA |
| :-------: | :--: | :--: | :--: |
|   MOT17   | 58.4 | 69.9 | 67.2 |
| SportsMOT | 70.9 | 68.9 | 95.7 |
| SoccerNet | 81.6 | 76.2 | 95.1 |

## Watch It in Action

<iframe width="100%" style="aspect-ratio: 16/9;" src="https://www.youtube.com/embed/u0k2dTZ0vfs" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Algorithm

SORT models each tracked object with a seven-dimensional state vector `[x, y, s, r, dx, dy, ds]`, where `x` and `y` are the bounding-box center coordinates, `s` is the box area (scale), `r` is the aspect ratio (held constant across predictions), and `dx`, `dy`, `ds` are the corresponding velocities.

**Prediction.** A constant-velocity Kalman filter propagates each track's state to the next frame. The filter assumes linear motion between frames and maintains an uncertainty covariance that grows when a track is not matched to a detection.

**Association.** The Hungarian algorithm solves the assignment between predicted track positions and incoming detections. The cost matrix is computed from pairwise IoU between predicted boxes and detection boxes. Any assignment where the IoU falls below `minimum_iou_threshold` is rejected, leaving both the track and the detection unmatched.

**Track lifecycle.** When a detection is not matched to any existing track, a new tentative track is created. The track is promoted to confirmed status only after it receives `minimum_consecutive_frames` consecutive matches. A confirmed track that goes unmatched for more than `lost_track_buffer` frames is deleted. This lifecycle prevents single spurious detections from creating permanent tracks and limits unbounded growth of the track set.

**Limitations.** SORT uses no appearance features. When two objects cross paths or one object occludes another, the IoU-only matching can swap identities, producing ID switches. For scenes with frequent occlusion, [ByteTrack](bytetrack.md) and [OC-SORT](ocsort.md) provide mechanisms that reduce these errors.

## Key Parameters

| Parameter                    | Purpose                                                     | Tuning guidance                                                                                                              |
| ---------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `lost_track_buffer`          | Frames to keep an unmatched track alive before deletion.    | Higher tolerates longer occlusions but risks false re-association. 10-30 for most scenes; up to 60 for very long occlusions. |
| `track_activation_threshold` | Minimum detection confidence to create or continue a track. | Higher reduces spurious tracks; lower catches weak detections. 0.5-0.9 typical.                                              |
| `minimum_consecutive_frames` | Consecutive detections required to confirm a new track.     | 1 confirms immediately; 2-3 filters out single-frame false positives.                                                        |
| `minimum_iou_threshold`      | Minimum IoU to accept a track-detection match.              | Lower associates through more displacement between frames. 0.1-0.3 typical.                                                  |

## Run on video, webcam, or RTSP stream

These examples use `opencv-python` for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually 0 for the default camera.

=== "Video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers import SORTTracker

    tracker = SORTTracker()
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
            annotated_frame,
            detections,
            labels=detections.tracker_id,
        )

        cv2.imshow("RF-DETR + SORT", annotated_frame)
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
    from trackers import SORTTracker

    tracker = SORTTracker()
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
            annotated_frame,
            detections,
            labels=detections.tracker_id,
        )

        cv2.imshow("RF-DETR + SORT", annotated_frame)
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
    from trackers import SORTTracker

    tracker = SORTTracker()
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
            annotated_frame,
            detections,
            labels=detections.tracker_id,
        )

        cv2.imshow("RF-DETR + SORT", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

## Reference

Bewley, A., Ge, Z., Ott, L., Ramos, F., and Upcroft, B. (2016). Simple online and realtime tracking. ICIP. [arXiv:1602.00763](https://arxiv.org/abs/1602.00763)
