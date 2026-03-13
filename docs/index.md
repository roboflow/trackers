---
comments: true
---

<div align="center">

<img src="assets/logo-trackers-violet.svg" alt="Trackers Logo" width="200" height="200">

</div>

Plug-and-play multi-object tracking for any detection model. Clean, modular implementations of SORT, ByteTrack, and OC-SORT under the Apache 2.0 license.

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/track-objects-page.mp4" type="video/mp4">
</video>

---

## Install

Get started by installing the package.

```text
pip install trackers
```

For more options, see the [install guide](learn/install.md).

---

[![Watch: Building Real-Time Multi-Object Tracking with RF-DETR and Trackers](https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/roboflow-piotr-rf-detr-trackers-v1b-callout.png)](https://www.youtube.com/watch?v=u0k2dTZ0vfs)

---

## Track from CLI

Track objects in any video with a single command.

```text
trackers track --source video.mp4 --output output.mp4
```

Configure the detection model, tracking algorithm, and visualization with additional flags.

```text
trackers track \
    --source video.mp4 \
    --output output.mp4 \
    --model rfdetr-medium \
    --tracker bytetrack \
    --show-labels \
    --show-trajectories
```

For all CLI options, see the [tracking guide](learn/track.md).

---

## Track from Python

Use `trackers` as a Python library alongside your existing detector. Two lines to add tracking to any pipeline.

```python hl_lines="4 7 17"
import cv2
import supervision as sv
from inference import get_model
from trackers import ByteTrackTracker

model = get_model(model_id="rfdetr-nano")
tracker = ByteTrackTracker()

cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.infer(frame)[0]
    detections = sv.Detections.from_inference(result)
    detections = tracker.update(detections)
```

---

## Evaluate

Benchmark your tracker against ground truth with standard MOT metrics.

```text
trackers eval \
    --gt-dir ./data/mot17/val \
    --tracker-dir results \
    --metrics CLEAR HOTA Identity \
    --columns MOTA HOTA IDF1
```

```
Sequence                        MOTA    HOTA    IDF1
----------------------------------------------------
MOT17-02-FRCNN                30.192  35.475  38.515
MOT17-04-FRCNN                48.912  55.096  61.854
MOT17-05-FRCNN                52.755  45.515  55.705
----------------------------------------------------
COMBINED                      47.406  50.355  56.600
```

For the full evaluation workflow, see the [evaluation guide](learn/evaluate.md).

---

## Tracking Algorithms

Clean, modular implementations of leading trackers.

| Algorithm | MOT17 HOTA | SportsMOT HOTA | SoccerNet HOTA |
| :-------: | :--------: | :------------: | :------------: |
|   SORT    |    58.4    |      70.9      |      81.6      |
| ByteTrack |    60.1    |    **73.0**    |    **84.0**    |
|  OC-SORT  |  **61.9**  |      71.5      |      78.6      |

For detailed benchmarks, see the [tracker comparison](trackers/comparison.md).

---

## Tutorials

<div class="grid cards" markdown>

- **How to Track Objects with SORT**

    ---

    [![](https://storage.googleapis.com/com-roboflow-marketing/trackers/assets/sort-sample.png)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-sort-tracker.ipynb)

    End-to-end example showing how to run RF-DETR detection with the SORT tracker.

    [:simple-googlecolab: Run Google Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-sort-tracker.ipynb)

- **How to Track Objects with ByteTrack**

    ---

    [![](https://storage.googleapis.com/com-roboflow-marketing/trackers/assets/bytetrack-sample.png)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-bytetrack-tracker.ipynb)

    End-to-end example showing how to run RF-DETR detection with the ByteTrack tracker.

    [:simple-googlecolab: Run Google Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-bytetrack-tracker.ipynb)

</div>
