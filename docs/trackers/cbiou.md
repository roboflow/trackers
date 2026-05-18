---
title: C-BIoU: Cascaded-Buffered IoU Tracker | Trackers
comments: true
description: Cascaded-Buffered IoU (C-BIoU) tracker from Yang et al. (WACV 2023). Cascaded BIoU matching with buffer scales b1 and b2 (b1 < b2) for irregular motion and similar appearances.
---

# C-BIoU (Cascaded-Buffered IoU)

## Overview

**C-BIoU** implements the tracker from Yang et al., [*Hard To Track Objects with Irregular Motions and Similar Appearances? Make It Easier by Buffering the Matching Space*](https://openaccess.thecvf.com/content/WACV2023/papers/Yang_Hard_To_Track_Objects_With_Irregular_Motions_and_Similar_Appearances_WACV_2023_paper.pdf) (WACV 2023).

The core idea is **Buffered IoU (BIoU)**: before computing overlap, each bounding box is expanded by a margin proportional to its width and height. That widens the matching space so tracks and detections can be associated even when raw boxes barely touch or miss due to fast motion or detector jitter.

The paper uses **cascaded matching** with two buffer scales: **b1** (small, first pass) and **b2** (large, second pass on remaining unmatched pairs). You should keep **b1 < b2**. On SoccerNet, the authors report grid-searching **b1 = 0.7** and **b2 = 1.0** (with a BIoU match threshold of **0.01** on that dataset). In this library, `buffer_ratio_first` is **b1** and `buffer_ratio_second` is **b2**.

C-BIoU follows the same tracking-by-detection backbone as [BoT-SORT](botsort.md) (Kalman prediction, two-stage high/low confidence association, unconfirmed matching) but **does not use camera motion compensation**. Only detection boxes are required, which suits MOT-benchmark and file-based workflows.

## How does C-BIoU compare to other trackers?

For comparisons with other trackers, plus dataset context and evaluation details, see the [tracker comparison](comparison.md) page.

Measured with this library (YOLOX detections on MOT17 and SportsMOT test; oracle detections on SoccerNet test and DanceTrack val). Default buffers: 
`buffer_ratio_first=0.3`, `buffer_ratio_second=0.5`.

=== "Default parameters"

|  Dataset  |   HOTA   |   IDF1   |   MOTA   |
| :-------: | :------: | :------: | :------: |
|   MOT17   |   63.0   |   79.1   |   77.4   |
| SportsMOT |   73.1   |   72.6   |   96.7   |
| SoccerNet |   82.6   |   76.6   |   97.0   |
| DanceTrack |  53.8   |   53.8   |   90.1   |

=== "Tuned parameters"

Tuned with Optuna (`trackers_cbiou_tuning.ipynb`): MOT17 val-half, SportsMOT val, SoccerNet train, DanceTrack train; evaluated on the splits below.

|  Dataset  |   HOTA   |   IDF1   |   MOTA   |
| :-------: | :------: | :------: | :------: |
|   MOT17   |   63.0   |   79.1   |   77.2   |
| SportsMOT |   72.5   |   72.2   |   96.9   |
| SoccerNet |   85.5   |   79.6   |   99.3   |
| DanceTrack |  53.3   |   54.4   |   89.2   |

C-BIoU is aimed at sports and dance scenes with irregular motion and similar-looking objects (SoccerNet, DanceTrack, SportsMOT), where the paper reports strong gains over SORT-style baselines.

## Algorithm

C-BIoU keeps the [ByteTrack](bytetrack.md)-style association pipeline used in [BoT-SORT](botsort.md) but replaces plain IoU with **cascaded Buffered IoU** at each association step.

**First association (b1).** High-confidence detections are matched to confirmed and lost tracks using BIoU with `buffer_ratio_first` (paper **b1**, small buffer). Costs are fused with detection confidence.

**Second association (b2).** Remaining *tracked* tracks (not lost) are matched to low-confidence detections using BIoU with `buffer_ratio_second` (paper **b2**, large buffer). Score fusion is not applied here. In the paper, the large buffer is applied to **remaining unmatched track/detection pairs** after the first cascade; here it is wired to ByteTrack's low-confidence recovery stage.

**Unconfirmed association (b1).** Leftover high-confidence detections are matched to unconfirmed tracks using the same buffer as pass 1. Unmatched unconfirmed tracks are removed. This step is from ByteTrack lifecycle logic, not the paper's two-buffer cascade.

**Track lifecycle.** New tracks are initiated and confirmed with a conservative policy (`minimum_consecutive_frames`) to reduce one-frame false positives. Tracks that remain unmatched longer than `lost_track_buffer` are removed.

## Key Parameters

| Parameter                                 | Purpose                                                            | Tuning guidance                                                                                                                                                                                                                        |
| ----------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lost_track_buffer`                       | Frames to keep an unmatched track alive before deletion.           | Higher tolerates longer occlusions but risks false re-association. 10-30 for most scenes; up to 60 for very long occlusions.                                                                                                           |
| `track_activation_threshold`              | Minimum detection confidence required to start a new track.        | Higher reduces noisy track creation; lower retains harder objects. 0.5-0.9 typical depending on detector quality. This does not control low-confidence association, which still discards detections at a fixed `0.1` confidence floor. |
| `minimum_consecutive_frames`              | Consecutive matches required before confirming a new track.        | 1 for immediate activation; 2-3 improves robustness against flicker and false positives.                                                                                                                                               |
| `minimum_iou_threshold_first_assoc`       | Minimum fused BIoU similarity for the first association pass.      | Paper uses very low values on some datasets (e.g. 0.01 on SoccerNet). Lower helps maintain matches under fast motion; higher is stricter.                                                                                              |
| `minimum_iou_threshold_second_assoc`      | Minimum BIoU similarity for the second association pass.           | Usually set lower than the first-pass threshold to recover weak detections without over-matching.                                                                                                                                      |
| `minimum_iou_threshold_unconfirmed_assoc` | Minimum fused BIoU similarity when associating unconfirmed tracks. | Higher values make tentative tracks harder to confirm spuriously; lower values help short-lived or noisy objects survive.                                                                                                              |
| `high_conf_det_threshold`                 | Confidence split between stage-1 and stage-2 detections.           | 0.5-0.7 common. Higher shifts more detections to recovery stage; lower gives stage-1 broader coverage.                                                                                                                                 |
| `buffer_ratio_first`                      | Paper **b1**, small BIoU buffer for the first association pass.    | Typical range 0.1-0.7. Should be **less than** `buffer_ratio_second`.                                                                                                                                                                  |
| `buffer_ratio_second`                     | Paper **b2**, large BIoU buffer for the second association pass.   | Typical range 0.2-1.0. Should be **greater than** `buffer_ratio_first`.                                                                                                                                                                |

!!! warning "Buffer ordering (b1 < b2)"

    Always set `buffer_ratio_first` < `buffer_ratio_second`. The cascaded matcher applies the **smaller** buffer first, then the **larger** buffer only on pairs that remain unmatched. Reversing the order (b1 ≥ b2) is not consistent with the paper and usually hurts performance.

!!! warning "Frame input is ignored by C-BIoU"

    `CBIoUTracker.update()` accepts `frame` for API consistency with other trackers, but C-BIoU does not use image/frame pixels.
    If you pass `frame` with a non-`None` value, the tracker emits a `UserWarning` and ignores it.

## Run on video, webcam, or RTSP stream

These examples use `opencv-python` for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually 0 for the default camera.

=== "Video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers import CBIoUTracker

    tracker = CBIoUTracker(
        buffer_ratio_first=0.3,
        buffer_ratio_second=0.5,
    )
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

        cv2.imshow("RF-DETR + C-BIoU", annotated_frame)
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
    from trackers import CBIoUTracker

    tracker = CBIoUTracker(
        buffer_ratio_first=0.3,
        buffer_ratio_second=0.5,
    )
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

        cv2.imshow("RF-DETR + C-BIoU", annotated_frame)
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
    from trackers import CBIoUTracker

    tracker = CBIoUTracker(
        buffer_ratio_first=0.3,
        buffer_ratio_second=0.5,
    )
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

        cv2.imshow("RF-DETR + C-BIoU", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

For BIoU mathematics and using `BIoU(buffer_ratio=...)` on other trackers, see [IoU variants](../learn/iou.md#biou). To tune hyperparameters with Optuna, see [Hyperparameter tuning](../learn/tune.md).

## Reference

Yang, F., Odashima, S., Masui, S., and Jiang, S. (2023). Hard To Track Objects with Irregular Motions and Similar Appearances? Make It Easier by Buffering the Matching Space. WACV 2023. [arXiv:2211.14317](https://arxiv.org/abs/2211.14317)
