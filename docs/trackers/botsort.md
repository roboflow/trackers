---
comments: true
description: BoT-SORT extends ByteTrack with camera motion compensation and confidence-aware association to improve identity stability in camera moving and crowded scenes.
---

# BoT-SORT

## Overview

BoT-SORT extends [ByteTrack](bytetrack.md) with camera motion compensation (CMC) to handle moving cameras and dynamic scenes. It keeps ByteTrack's two-stage association strategy (high-confidence matching followed by low-confidence recovery), but first applies a frame-to-frame geometric transform estimated from optical flow so predictions are compared in the correct camera coordinate frame. This reduces missed matches and ID-switches when camera ego-motion causes apparent object jumps. BoT-SORT also combines IoU similarity with detection confidence during association and uses stricter track confirmation logic for more stable identities.

## How does BoT-SORT compare to other trackers?

For comparisons with other trackers, plus dataset context and evaluation details, see the [tracker comparison](../evaluations/results.md) page.

<!-- BENCH-XREF copy-of: [docs/evaluations/results.md](../evaluations/results.md) BoT-SORT row in mot17-default/sportsmot-default/soccernet-default tables. Also duplicated in [docs/index.md](../index.md) (Algorithms table), [README.md](../../README.md) (Algorithms table), and [mcbyte.md](mcbyte.md) (BoT-SORT baseline row, matching tab per benchmark). No DanceTrack row here by design. Update results.md first, then mirror here. -->

|  Dataset  | HOTA | IDF1 | MOTA |
| :-------: | :--: | :--: | :--: |
|   MOT17   | 63.7 | 78.7 | 79.2 |
| SportsMOT | 73.8 | 73.4 | 96.9 |
| SoccerNet | 84.5 | 79.3 | 96.6 |

## Watch It in Action

<video title="BoT-SORT demo video" width="100%" style="aspect-ratio: 16/9;" controls>
  <source src="https://github.com/user-attachments/assets/c8fdc1df-7e3b-4d44-bad0-d08208ddc6a0" type="video/mp4">
</video>

## Algorithm

BoT-SORT keeps the same tracking-by-detection backbone as [ByteTrack](bytetrack.md) but adds camera-motion-aware prediction and confidence-aware association.

**CMC (Camera Motion Compensation).** Before data association, BoT-SORT estimates global camera motion between consecutive frames (typically from sparse optical flow) and warps each track's Kalman-predicted box into the current frame. Without this step, a panning or moving camera can make stationary or slow-moving targets appear to jump, degrading IoU overlap and causing false unmatched tracks.

**Two-stage association.** BoT-SORT performs ByteTrack-style matching in two passes: high-confidence detections first, then lower-confidence detections against remaining *tracked* (non-lost) tracks that were just updated the previous frame. This recovers objects that are briefly weakly scored due to blur, occlusion, or scale change.

**Confidence-aware matching.** Association costs blend geometric overlap (IoU) with detection confidence so that stronger detections are preferred when multiple matches are plausible.

**Track lifecycle.** New tracks are initiated and confirmed with a conservative policy (`minimum_consecutive_frames`) to reduce one-frame false positives. Tracks that remain unmatched longer than `lost_track_buffer` are removed.

## Key Parameters

| Parameter                                 | Purpose                                                                                                                     | Tuning guidance                                                                                                                                                                                                                        |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lost_track_buffer`                       | Frames to keep an unmatched track alive before deletion (specified in 30 FPS units, scaled proportionally by `frame_rate`). | Higher tolerates longer occlusions/camera shake but can increase false re-association. 10-30 common; up to 60 for long gaps.                                                                                                           |
| `track_activation_threshold`              | Minimum detection confidence required to start a new track.                                                                 | Higher reduces noisy track creation; lower retains harder objects. 0.5-0.9 typical depending on detector quality. This does not control low-confidence association, which still discards detections at a fixed `0.1` confidence floor. |
| `minimum_consecutive_frames`              | Consecutive matches required before confirming a new track.                                                                 | 1 for immediate activation; 2-3 improves robustness against flicker and false positives.                                                                                                                                               |
| `minimum_iou_threshold_first_assoc`       | Minimum fused similarity (IoU x detection confidence) for the first association pass with high-confidence detections.       | Lower helps maintain matches under fast motion or imperfect compensation; higher is stricter and reduces risky matches.                                                                                                                |
| `minimum_iou_threshold_second_assoc`      | Minimum IoU for the second association pass with lower-confidence detections.                                               | Typically set higher than the first-pass threshold: this pass has no confidence fusion, so a stricter geometric threshold guards against over-matching low-confidence detections.                                                      |
| `minimum_iou_threshold_unconfirmed_assoc` | Minimum IoU when associating unconfirmed tracks.                                                                            | Higher values make tentative tracks harder to confirm spuriously; lower values help short-lived or noisy objects survive.                                                                                                              |
| `high_conf_det_threshold`                 | Confidence split between stage-1 and stage-2 detections.                                                                    | 0.5-0.7 common. Higher shifts more detections to recovery stage; lower gives stage-1 broader coverage.                                                                                                                                 |
| `enable_cmc`                              | Enables camera motion compensation before association.                                                                      | Keep enabled for moving-camera footage (sports, drone, handheld). Disable mainly for static cameras if you need maximal speed.                                                                                                         |

## ReID appearance (optional)

BoT-SORT can fuse appearance embeddings with IoU during association via an optional `reid_model`. For ReID numbers go to the [ReID guide](../guides/reid.md#bot-sort-with-and-without-reid), which holds the tables for both fusion rules and best configuration per dataset. Also learn how to install, use and choose parameters on the same page.

## Run on video, webcam, or RTSP stream

These examples use `opencv-python` for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually 0 for the default camera.

!!! tip

    Pass the current video frame as `tracker.update(detections, frame=frame_bgr)` to enable Camera Motion Compensation.

=== "CLI"

    Run BoT-SORT on a video without writing any Python. See the [CLI reference](../guides/cli.md) for every argument, including `--source 0` for a webcam or an `rtsp://` URL for a stream.

    ```bash
    trackers track \
        --source <SOURCE_VIDEO_PATH> \
        --tracker botsort \
        --output.video output.mp4
    ```

=== "Video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers import BoTSORTTracker

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
        detections = tracker.update(detections, frame=frame_bgr)

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
        detections = tracker.update(detections, frame=frame_bgr)

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
        detections = tracker.update(detections, frame=frame_bgr)

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

## Reference

Aharon, N., Orfaig, R., and Bobrovsky, B.-Z. (2023). BoT-SORT: Robust Associations Multi-Pedestrian Tracking. [arXiv:2206.14651](https://arxiv.org/abs/2206.14651)
