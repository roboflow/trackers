# Track Objects

Combine object detection with multi-object tracking to follow objects through video sequences, maintaining consistent IDs even through occlusions and fast motion.

**What you'll learn:**

- Run tracking from the command line with a single command
- Configure detection models and tracking algorithms
- Visualize results with bounding boxes, IDs, and trajectories
- Build custom tracking pipelines in Python

---

## Install

Use the base install for tracking with your own detector. The `detection` extra adds `inference-models` for built-in detection.

```text
pip install trackers
```

```text
pip install trackers[detection]
```

For more options, see the [install guide](install.md).

---

## Quickstart

Read frames from video files, webcams, RTSP streams, or image directories. Each frame flows through detection to find objects, then through tracking to assign IDs.

=== "CLI"

    Track objects with one command. Uses RF-DETR Nano and ByteTrack by default.

    ```text
    trackers track --source source.mp4 --output output.mp4
    ```

=== "Python"

    While `trackers` focuses on ID assignment, this example uses `inference-models` for detection and `supervision` for format conversion to demonstrate end-to-end usage.

    ```python
    import cv2

    import supervision as sv
    from inference import get_model
    from trackers import ByteTrackTracker

    model = get_model("rfdetr-nano")
    tracker = ByteTrackTracker()

    cap = cv2.VideoCapture("source.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections)
    ```

---

## Trackers

Trackers assign stable IDs to detections across frames, maintaining object identity through motion and occlusion.

=== "CLI"

    Select a tracker with `--tracker` and tune its behavior with `--tracker.*` arguments.

    ```text
    trackers track --source source.mp4 --tracker bytetrack \
        --tracker.lost_track_buffer 60 \
        --tracker.minimum_consecutive_frames 5
    ```

    Adjust these settings to tune ID stability and noise rejection:

    - `--tracker` — Tracking algorithm to use. Options: `bytetrack`, `sort`. Default: `bytetrack`.
    - `--tracker.lost_track_buffer` — Number of frames to retain a track without matching detections. Higher values improve occlusion handling but risk ID drift. Default: `30`.
    - `--tracker.track_activation_threshold` — Minimum detection confidence to start a new track. Lower values catch more objects but increase false positives. Default: `0.25`.
    - `--tracker.minimum_consecutive_frames` — Consecutive detections required before a track is confirmed. Suppresses spurious detections from becoming tracks. Default: `3`.
    - `--tracker.minimum_iou_threshold` — Minimum IoU overlap to match a detection to an existing track. Higher values require tighter spatial alignment. Default: `0.3`.

=== "Python"

    Customize the tracker by passing parameters to the constructor, then call `update()` each frame and `reset()` between videos.

    ```python
    import cv2

    import supervision as sv
    from inference import get_model
    from trackers import ByteTrackTracker

    model = get_model("rfdetr-nano")
    tracker = ByteTrackTracker(
        lost_track_buffer=60,
        minimum_consecutive_frames=5,
    )

    cap = cv2.VideoCapture("source.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections)
    ```

---

## Detectors

Trackers don't detect objects—they link detections across frames. A detection or segmentation model provides per-frame bounding boxes or masks that the tracker uses to assign and maintain IDs.

=== "CLI"

    Configure detection with `--model.*` arguments. Filter by confidence and class before tracking.

    ```text
    trackers track --source source.mp4 --model rfdetr-medium \
        --model.confidence 0.3 \
        --model.device cuda \
        --classes 0,2
    ```

    Use these options to configure model inference:

    - `--model` — Model identifier. Pretrained options: `rfdetr-nano`, `rfdetr-small`, `rfdetr-medium`, `rfdetr-large`. Segmentation variants: `rfdetr-seg-nano`, `rfdetr-seg-small`, `rfdetr-seg-medium`, `rfdetr-seg-large`. Default: `rfdetr-nano`.
    - `--model.confidence` — Minimum confidence threshold. Lower values increase recall (more detections) but may add noise. Default: `0.5`.
    - `--model.device` — Compute device. Options: `auto`, `cpu`, `cuda`, `cuda:0`, `mps`. Default: `auto`.
    - `--classes` — Comma-separated class IDs to track. Example: `0` for persons, `0,2` for persons and cars (COCO). Default: all classes.
    - `--model.api_key` — Roboflow API key for custom hosted models. Default: none.

=== "Python"

    Trackers are modular—combine any detection library with any tracker. This example uses `inference` with RF-DETR.

    ```python
    import cv2

    import supervision as sv
    from inference import get_model
    from trackers import ByteTrackTracker

    model = get_model("rfdetr-nano")
    tracker = ByteTrackTracker()

    cap = cv2.VideoCapture("source.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections)
    ```

---

## Visualization

Visualization renders tracking results for debugging, demos, and qualitative evaluation.

=== "CLI"

    Enable display and annotation options to see results in real time or in saved video.

    ```text
    trackers track --source source.mp4 --display \
        --show-labels --show-confidence --show-trajectories
    ```

    Annotations include class labels, IDs, and confidence values, with per-ID coloring for easy tracking.

    - `--display` — Opens a live preview window. Press `q` or `ESC` to quit. Default: `false`.
    - `--show-boxes` — Draw bounding boxes around tracked objects. Default: `true`.
    - `--show-masks` — Draw segmentation masks. Only available with segmentation models (`rfdetr-seg-*`). Default: `false`.
    - `--show-confidence` — Show detection confidence scores in labels. Default: `false`.
    - `--show-labels` — Show class names in labels. Default: `false`.
    - `--show-ids` — Show tracker IDs in labels. Default: `true`.
    - `--show-trajectories` — Draw motion trails showing recent positions of each track. Default: `false`.

=== "Python"

    Use `supervision` annotators to draw results on frames before saving or displaying.

    ```python
    import cv2

    import supervision as sv
    from inference import get_model
    from trackers import ByteTrackTracker

    model = get_model("rfdetr-nano")
    tracker = ByteTrackTracker()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture("source.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections)

        frame = box_annotator.annotate(frame, detections)
        frame = label_annotator.annotate(frame, detections)
    ```

---

## Source

`trackers` accepts video files, webcams, RTSP streams, and directories of images as input sources.

=== "CLI"

    Pass videos, webcams, streams, or image directories as input.

    ```text
    trackers track --source source.mp4
    ```

    - `--source` — Input source. Accepts file paths (`.mp4`, `.avi`), device indices (`0`, `1`), stream URLs (`rtsp://`), or directories containing images.

=== "Python"

    Use `opencv-python`'s `VideoCapture` to read frames from files, webcams, or streams.

    ```python
    import cv2

    import supervision as sv
    from inference import get_model
    from trackers import ByteTrackTracker

    model = get_model("rfdetr-nano")
    tracker = ByteTrackTracker()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections)
    ```

---

## Output

Save tracking results as annotated video files or display them in real time.

=== "CLI"

    Specify an output path to save annotated video.

    ```text
    trackers track --source source.mp4 --output output.mp4 --overwrite
    ```

    - `--output` — Path for output video. If a directory is given, saves as `output.mp4` inside it. Default: none.
    - `--overwrite` — Allow overwriting existing output files. Without this flag, existing files cause an error. Default: `false`.

=== "Python"

    Use `opencv-python`'s `VideoWriter` to save annotated frames with full control over codec and frame rate.

    ```python
    import cv2

    import supervision as sv
    from inference import get_model
    from trackers import ByteTrackTracker

    model = get_model("rfdetr-nano")
    tracker = ByteTrackTracker()
    box_annotator = sv.BoxAnnotator()

    cap = cv2.VideoCapture("source.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections)

        frame = box_annotator.annotate(frame, detections)
        out.write(frame)

    cap.release()
    out.release()
    ```

---

## Integration

`trackers` works with any detection or segmentation library. Convert model output to `supervision` format and pass it to the tracker.

```python
import cv2

import supervision as sv
from ultralytics import YOLO
from trackers import ByteTrackTracker

model = YOLO("yolo11n.pt")
tracker = ByteTrackTracker()

cap = cv2.VideoCapture("source.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update(detections)
```
