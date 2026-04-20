---
comments: true
description: ByteTrack improves multi-object tracking by associating every detection box — including low-confidence ones — to reduce missed tracks and fragmentation while maintaining real-time performance.
---

# ByteTrack

## Overview

ByteTrack builds on the same Kalman filter plus Hungarian algorithm framework as SORT but changes the data association strategy to use almost every detection box regardless of confidence score. It runs a two-stage matching: first match high-confidence detections to tracks, then match low-confidence detections to any unmatched tracks using IoU. This reduces missed tracks and fragmentation for occluded or weak detections while retaining simplicity and high frame rates. ByteTrack has set state-of-the-art results on standard MOT benchmarks with real-time performance, because it recovers valid low-score detections instead of discarding them.

## Comparison

For comparisons with other trackers, plus dataset context and evaluation details, see the [tracker comparison](comparison.md) page.

|  Dataset  | HOTA | IDF1 | MOTA |
| :-------: | :--: | :--: | :--: |
|   MOT17   | 60.1 | 73.2 | 74.1 |
| SportsMOT | 73.0 | 72.5 | 96.4 |
| SoccerNet | 84.0 | 78.1 | 97.8 |

## Watch It in Action

<iframe width="100%" style="aspect-ratio: 16/9;" src="https://www.youtube.com/embed/u0k2dTZ0vfs" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Algorithm

ByteTrack builds on the same Kalman filter and Hungarian algorithm framework as [SORT](sort.md) but changes how detections are associated to tracks. Instead of discarding low-confidence detections, ByteTrack uses a two-stage matching strategy that recovers valid objects the detector scored low due to occlusion, blur, or partial visibility.

**Stage 1 -- high-confidence matching.** Detections with confidence above `high_conf_det_threshold` are matched to confirmed tracks using IoU-based Hungarian assignment, identical to SORT. Unmatched tracks and unmatched high-confidence detections pass to the next stage.

**Stage 2 -- low-confidence matching.** Detections with confidence between `track_activation_threshold` and `high_conf_det_threshold` are matched to the remaining unmatched tracks using IoU. This second pass associates weak detections to already-established tracks, recovering objects that would otherwise be lost. Detections below `track_activation_threshold` are discarded entirely and never start new tracks.

**Track lifecycle.** New tracks are initialized only from unmatched high-confidence detections (stage 1). A new track is promoted to confirmed status after `minimum_consecutive_frames` consecutive matches. Tracks that go unmatched for more than `lost_track_buffer` frames are deleted.

**Key insight.** Discarding low-confidence detections outright loses genuinely valid objects that happen to have a low score in one or a few frames. ByteTrack recaptures these by associating them with tracks that already have an established identity and motion history, rather than treating them as new objects. This produces fewer missed tracks and fewer ID switches with almost no additional computation over SORT.

## Key Parameters

| Parameter                    | Purpose                                                          | Tuning guidance                                                                                                              |
| ---------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `lost_track_buffer`          | Frames to keep an unmatched track alive before deletion.         | Higher tolerates longer occlusions but risks false re-association. 10-30 for most scenes; up to 60 for very long occlusions. |
| `track_activation_threshold` | Minimum detection confidence to use in any matching stage.       | Higher reduces spurious tracks; lower catches weak detections. 0.5-0.9 typical.                                              |
| `minimum_consecutive_frames` | Consecutive detections required to confirm a new track.          | 1 confirms immediately; 2-3 filters out single-frame false positives.                                                        |
| `minimum_iou_threshold`      | Minimum IoU to accept a track-detection match.                   | Lower associates through more displacement between frames. 0.1-0.3 typical.                                                  |
| `high_conf_det_threshold`    | Confidence threshold separating stage-1 from stage-2 detections. | 0.5-0.7 typical. Lower sends more detections to stage 1; higher relies more on stage-2 recovery.                             |

## Run on video, webcam, or RTSP stream

These examples use `opencv-python` for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually 0 for the default camera.

=== "Video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers import ByteTrackTracker

    tracker = ByteTrackTracker()
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

        cv2.imshow("RF-DETR + ByteTrack", annotated_frame)
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
    from trackers import ByteTrackTracker

    tracker = ByteTrackTracker()
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

        cv2.imshow("RF-DETR + ByteTrack", annotated_frame)
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
    from trackers import ByteTrackTracker

    tracker = ByteTrackTracker()
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

        cv2.imshow("RF-DETR + ByteTrack", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

## Reference

Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., and Wang, X. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV. [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
