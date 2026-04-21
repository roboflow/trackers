---
comments: true
description: Get started with Roboflow Trackers — install SORT, ByteTrack, and OC-SORT, run your first tracking pipeline, and evaluate results with HOTA, IDF1, and MOTA metrics.
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

Point at a video, webcam, RTSP stream, or image directory. Get tracked output.

```bash
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

Plug trackers into your existing detection pipeline. Works with any detector.

```python hl_lines="4 7 17"
import cv2
import supervision as sv
from inference import get_model
from trackers import ByteTrackTracker

model = get_model(model_id="rfdetr-medium")
tracker = ByteTrackTracker()

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = model.infer(frame)[0]
    detections = sv.Detections.from_inference(result)
    tracked = tracker.update(detections)
```

For more examples, see the [tracking guide](learn/track.md).

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
MOT17-09-FRCNN                51.441  50.108  57.038
MOT17-10-FRCNN                51.832  49.648  55.797
MOT17-11-FRCNN                55.501  49.401  55.061
MOT17-13-FRCNN                60.488  58.651  69.884
----------------------------------------------------
COMBINED                      47.406  50.355  56.600
```

For the full evaluation workflow, see the [evaluation guide](learn/evaluate.md).

---

## Algorithms

Clean, modular implementations of leading trackers. All HOTA scores use default parameters.

|                   Algorithm                   |                           Description                           | MOT17 HOTA | SportsMOT HOTA | SoccerNet HOTA | DanceTrack HOTA |
| :-------------------------------------------: | :-------------------------------------------------------------: | :--------: | :------------: | :------------: | :-------------: |
|   [SORT](https://arxiv.org/abs/1602.00763)    |          Kalman filter + Hungarian matching baseline.           |    58.4    |      70.9      |      81.6      |      45.0       |
| [ByteTrack](https://arxiv.org/abs/2110.06864) | Two-stage association using high and low confidence detections. |    60.1    |    **73.0**    |    **84.0**    |      50.2       |
|  [OC-SORT](https://arxiv.org/abs/2203.14360)  |          Observation-centric recovery for lost tracks.          |  **61.9**  |      71.7      |      78.4      |    **51.8**     |

For detailed benchmarks and tuned configurations, see the [tracker comparison](trackers/comparison.md).

---

## Download Datasets

Pull benchmark datasets for evaluation with a single command.

```bash
trackers download mot17 \
    --split val \
    --asset annotations,detections
```

|   Dataset   |                               Description                               |         Splits         |                Assets                 |     License     |
| :---------: | :---------------------------------------------------------------------: | :--------------------: | :-----------------------------------: | :-------------: |
|   `mot17`   |    Pedestrian tracking with crowded scenes and frequent occlusions.     | `train`, `val`, `test` | `frames`, `annotations`, `detections` | CC BY-NC-SA 3.0 |
| `sportsmot` | Sports broadcast tracking with fast motion and similar-looking targets. | `train`, `val`, `test` |        `frames`, `annotations`        |    CC BY 4.0    |

For more download options, see the [download guide](learn/download.md).

---

## Try It

Try trackers in your browser with our [Hugging Face Playground](https://huggingface.co/spaces/roboflow/trackers).

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

---

## FAQ

**What is multi-object tracking and how does it differ from object detection?**

Object detection finds and classifies objects in a single image frame. Multi-object tracking
assigns a persistent ID to each detected object across video frames, maintaining continuity
through occlusions, re-entries, and camera motion. Trackers use a detect-then-track approach:
a detector runs on each frame, and the tracker links detections across time using motion
models and spatial matching.

**Which tracker should I use?**

Start with ByteTrack — out of the box, it performs best across two out of four benchmarks in our evaluation and
handles variable-confidence detectors well, while providing real time latency. Use SORT if speed or device constraints require the lightest possible tracker. Use OC-SORT when camera motion is significant or objects follow
non-linear paths. See the [tracker comparison](trackers/comparison.md) for benchmark scores.

**Do I need a specific detector?**

No. Roboflow Trackers works with any detector that outputs `supervision.Detections` objects.
The library ships example pipelines using RF-DETR but is compatible with YOLO, Detectron2,
and any custom model. The tracker never inspects the detection model directly.

**What MOT datasets does the library support?**

MOT17, MOT20, SportsMOT, SoccerNet-tracking, and DanceTrack are supported for download and
evaluation. Use `trackers download <dataset>` to pull frames, annotations, and pre-computed
detections in one command. See the [download guide](learn/download.md) for asset options.

**How do I evaluate my tracker?**

Run `trackers eval` against a directory of ground-truth MOT-format text files. The evaluation
pipeline computes HOTA, IDF1, and MOTA and prints a per-sequence and combined score table.
See the [evaluation guide](learn/evaluate.md) for the full workflow.
