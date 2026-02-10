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

    <table>
      <colgroup>
        <col style="width: 40%">
        <col style="width: 40%">
        <col style="width: 20%">
      </colgroup>
      <thead>
        <tr>
          <th>Argument</th>
          <th>Description</th>
          <th>Default</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>--tracker</code></td>
          <td>Tracking algorithm. Options: <code>bytetrack</code>, <code>sort</code>.</td>
          <td><code>bytetrack</code></td>
        </tr>
        <tr>
          <td><code>--tracker.lost_track_buffer</code></td>
          <td>Frames to retain a track without detections. Higher values improve occlusion handling but risk ID drift.</td>
          <td><code>30</code></td>
        </tr>
        <tr>
          <td><code>--tracker.track_activation_threshold</code></td>
          <td>Minimum confidence to start a new track. Lower values catch more objects but increase false positives.</td>
          <td><code>0.25</code></td>
        </tr>
        <tr>
          <td><code>--tracker.minimum_consecutive_frames</code></td>
          <td>Consecutive detections required before a track is confirmed. Suppresses spurious detections.</td>
          <td><code>3</code></td>
        </tr>
        <tr>
          <td><code>--tracker.minimum_iou_threshold</code></td>
          <td>Minimum IoU overlap to match a detection to an existing track. Higher values require tighter alignment.</td>
          <td><code>0.3</code></td>
        </tr>
      </tbody>
    </table>

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

    <table>
      <colgroup>
        <col style="width: 40%">
        <col style="width: 40%">
        <col style="width: 20%">
      </colgroup>
      <thead>
        <tr>
          <th>Argument</th>
          <th>Description</th>
          <th>Default</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>--model</code></td>
          <td>Model identifier. Pretrained: <code>rfdetr-nano</code>, <code>rfdetr-small</code>, <code>rfdetr-medium</code>, <code>rfdetr-large</code>. Segmentation: <code>rfdetr-seg-*</code>.</td>
          <td><code>rfdetr-nano</code></td>
        </tr>
        <tr>
          <td><code>--model.confidence</code></td>
          <td>Minimum confidence threshold. Lower values increase recall but may add noise.</td>
          <td><code>0.5</code></td>
        </tr>
        <tr>
          <td><code>--model.device</code></td>
          <td>Compute device. Options: <code>auto</code>, <code>cpu</code>, <code>cuda</code>, <code>cuda:0</code>, <code>mps</code>.</td>
          <td><code>auto</code></td>
        </tr>
        <tr>
          <td><code>--classes</code></td>
          <td>Comma-separated class IDs to track. Example: <code>0</code> for persons, <code>0,2</code> for persons and cars.</td>
          <td>all</td>
        </tr>
        <tr>
          <td><code>--model.api_key</code></td>
          <td>Roboflow API key for custom hosted models.</td>
          <td>none</td>
        </tr>
      </tbody>
    </table>

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

    <table>
      <colgroup>
        <col style="width: 40%">
        <col style="width: 40%">
        <col style="width: 20%">
      </colgroup>
      <thead>
        <tr>
          <th>Argument</th>
          <th>Description</th>
          <th>Default</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>--display</code></td>
          <td>Opens a live preview window. Press <code>q</code> or <code>ESC</code> to quit.</td>
          <td><code>false</code></td>
        </tr>
        <tr>
          <td><code>--show-boxes</code></td>
          <td>Draw bounding boxes around tracked objects.</td>
          <td><code>true</code></td>
        </tr>
        <tr>
          <td><code>--show-masks</code></td>
          <td>Draw segmentation masks. Only available with <code>rfdetr-seg-*</code> models.</td>
          <td><code>false</code></td>
        </tr>
        <tr>
          <td><code>--show-confidence</code></td>
          <td>Show detection confidence scores in labels.</td>
          <td><code>false</code></td>
        </tr>
        <tr>
          <td><code>--show-labels</code></td>
          <td>Show class names in labels.</td>
          <td><code>false</code></td>
        </tr>
        <tr>
          <td><code>--show-ids</code></td>
          <td>Show tracker IDs in labels.</td>
          <td><code>true</code></td>
        </tr>
        <tr>
          <td><code>--show-trajectories</code></td>
          <td>Draw motion trails showing recent positions of each track.</td>
          <td><code>false</code></td>
        </tr>
      </tbody>
    </table>

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

    <table>
      <colgroup>
        <col style="width: 40%">
        <col style="width: 40%">
        <col style="width: 20%">
      </colgroup>
      <thead>
        <tr>
          <th>Argument</th>
          <th>Description</th>
          <th>Default</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>--source</code></td>
          <td>Input source. Accepts file paths (<code>.mp4</code>, <code>.avi</code>), device indices (<code>0</code>, <code>1</code>), stream URLs (<code>rtsp://</code>), or image directories.</td>
          <td>—</td>
        </tr>
      </tbody>
    </table>

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

    <table>
      <colgroup>
        <col style="width: 40%">
        <col style="width: 40%">
        <col style="width: 20%">
      </colgroup>
      <thead>
        <tr>
          <th>Argument</th>
          <th>Description</th>
          <th>Default</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>--output</code></td>
          <td>Path for output video. If a directory is given, saves as <code>output.mp4</code> inside it.</td>
          <td>none</td>
        </tr>
        <tr>
          <td><code>--overwrite</code></td>
          <td>Allow overwriting existing output files. Without this flag, existing files cause an error.</td>
          <td><code>false</code></td>
        </tr>
      </tbody>
    </table>

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
