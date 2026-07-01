<div align="center">
    <img width="200" src="https://raw.githubusercontent.com/roboflow/trackers/refs/heads/release/stable/docs/assets/logo-trackers-violet.svg" alt="trackers logo">
    <h1>trackers</h1>
    <p>Plug-and-play multi-object tracking for any detection model.</p>

[![version](https://badge.fury.io/py/trackers.svg)](https://badge.fury.io/py/trackers)
[![downloads](https://img.shields.io/pypi/dm/trackers)](https://pypistats.org/packages/trackers)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/trackers/blob/release/stable/LICENSE.md)
[![python-version](https://img.shields.io/pypi/pyversions/trackers)](https://badge.fury.io/py/trackers)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-objects-with-bytetrack-tracker.ipynb)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

</div>

Keeping track of objects across video frames is one of those problems that sounds simple until you try it — occlusions, fast motion, similar-looking targets, and moving cameras all conspire against you. `trackers` gives you clean, benchmarked implementations of SORT, ByteTrack, OC-SORT, BoT-SORT, and C-BIoU so you can skip the plumbing and focus on your application. It speaks `supervision.Detections` natively, which means it slots into any detector you already use — YOLO, DETR, RT-DETR, or anything else — without glue code. Whether you are a researcher comparing algorithms, an engineer shipping a production pipeline, or a hobbyist building something cool, `trackers` gives you a single consistent interface for all of them. Requires Python ≥ 3.10.

## Why trackers?

- **Clean-room implementations.** Every algorithm is re-implemented from the original paper — not a thin wrapper around someone else's code. You can read it, understand it, and modify it.
- **Detector-agnostic.** Works with YOLO, DETR, RT-DETR, or any model that produces bounding boxes. No inference library required or assumed.
- **`supervision.Detections` native.** Plugs directly into the supervision ecosystem. Pass detections in, get tracked detections back — zero glue code.
- **Benchmarked across four datasets.** MOT17, SportsMOT, SoccerNet, and DanceTrack — at default parameters and after hyperparameter tuning, so you know what to expect before you deploy.
- **Tunable out of the box.** Built-in Optuna-based hyperparameter search via `trackers tune` so you can optimize for your specific scene and detector.
- **Camera motion compensation.** BoT-SORT handles moving cameras natively, keeping track IDs stable even when the whole frame shifts.

## Install

```bash
pip install trackers
```

<details>
<summary>Install from source</summary>

```bash
pip install git+https://github.com/roboflow/trackers.git
```

</details>

For more options, see the [install guide](https://trackers.roboflow.com/develop/learn/install/).

[![Watch: Building Real-Time Multi-Object Tracking with RF-DETR and Trackers](https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/roboflow-piotr-rf-detr-trackers-v1b-callout.png)](https://www.youtube.com/watch?v=u0k2dTZ0vfs)

## Quick Start

Add tracking to your existing detection pipeline in a few lines. Every tracker shares the same `update(detections, frame=None)` interface, so switching algorithms later is a one-line change. The example below uses `inference` as the detector — swap it for any detector that returns `supervision.Detections`.

```python
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

For more examples, see the [tracking guide](https://trackers.roboflow.com/develop/learn/track/).

## Track from CLI

Prefer the terminal? Point `trackers track` at a video, webcam feed, RTSP stream, or image directory and it handles detection, tracking, and annotated output in one command — no Python script required.

```bash
trackers track \
    --source video.mp4 \
    --output output.mp4 \
    --model rfdetr-medium \
    --tracker bytetrack \
    --show-labels \
    --show-trajectories
```

For all CLI options, see the [tracking guide](https://trackers.roboflow.com/develop/learn/track/).

## Algorithms

Each tracker below is a faithful implementation of its original paper. Pick the one that fits your scene, or run the benchmark to find out which performs best on your data.

|                                                                           Algorithm                                                                           |                           Description                           | MOT17 HOTA | SportsMOT HOTA | SoccerNet HOTA | DanceTrack HOTA |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------: | :--------: | :------------: | :------------: | :-------------: |
|                                                           [SORT](https://arxiv.org/abs/1602.00763)                                                            |          Kalman filter + Hungarian matching baseline.           |    58.4    |      70.9      |      81.6      |      47.2       |
|                                                         [ByteTrack](https://arxiv.org/abs/2110.06864)                                                         | Two-stage association using high and low confidence detections. |    60.1    |      73.0      |      84.0      |      53.3       |
|                                                          [OC-SORT](https://arxiv.org/abs/2203.14360)                                                          |          Observation-centric recovery for lost tracks.          |    61.9    |      71.7      |      78.4      |      54.1       |
|                                                         [BoT-SORT](https://arxiv.org/abs/2206.14651)                                                          |                   Camera motion compensation                    |  **63.7**  |    **73.8**    |    **84.5**    |    **57.8**     |
| [C-BIoU](https://openaccess.thecvf.com/content/WACV2023/papers/Yang_Hard_To_Track_Objects_With_Irregular_Motions_and_Similar_Appearances_WACV_2023_paper.pdf) |  Cascaded buffered IoU matching for fast or irregular motion.   |    63.0    |      73.1      |      82.6      |      56.7       |

All scores use default parameters on the standard split. See the [tracker comparison](https://trackers.roboflow.com/develop/trackers/comparison/) for tuned numbers and methodology.

## Evaluate

Once you have tracking results, you want to know how good they are. `trackers eval` computes CLEAR, HOTA, and Identity metrics against ground-truth annotations and prints a per-sequence breakdown alongside the combined score.

```bash
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

For the full evaluation workflow, see the [evaluation guide](https://trackers.roboflow.com/develop/learn/evaluate/).

## Download Datasets

Need benchmark data to evaluate against? `trackers download` pulls MOT17, SportsMOT, and other supported datasets with a single command, handling splits and assets selectively so you only download what you need.

```bash
trackers download mot17 \
    --split val \
    --asset annotations,detections
```

|   Dataset   |                               Description                               |         Splits         |                Assets                 |     License     |
| :---------: | :---------------------------------------------------------------------: | :--------------------: | :-----------------------------------: | :-------------: |
|   `mot17`   |    Pedestrian tracking with crowded scenes and frequent occlusions.     | `train`, `val`, `test` | `frames`, `annotations`, `detections` | CC BY-NC-SA 3.0 |
| `sportsmot` | Sports broadcast tracking with fast motion and similar-looking targets. | `train`, `val`, `test` |        `frames`, `annotations`        |    CC BY 4.0    |

For more download options, see the [download guide](https://trackers.roboflow.com/develop/learn/download/).

## Try It

Want to see it in action before writing any code? Try trackers in your browser with our [Hugging Face Playground](https://huggingface.co/spaces/roboflow/trackers) — no install required.

## Where to go next

- **New to tracking?** Start with the [tracking guide](https://trackers.roboflow.com/develop/learn/track/) — it walks through the Python API and CLI end to end.
- **Want benchmarks?** The [tracker comparison](https://trackers.roboflow.com/develop/trackers/comparison/) covers all four algorithms across all four datasets, at default and tuned parameters, with guidance on which to pick for your scene.
- **Building a research pipeline?** The [evaluation guide](https://trackers.roboflow.com/develop/learn/evaluate/) and [download guide](https://trackers.roboflow.com/develop/learn/download/) cover the full offline benchmarking workflow.
- **Full API reference** → [trackers.roboflow.com](https://trackers.roboflow.com)
- **Try without installing** → [Hugging Face Playground](https://huggingface.co/spaces/roboflow/trackers)
- **Questions?** Find us on [Discord](https://discord.gg/GbfgXGJ8Bk).

## Contributing

We welcome contributions. Read our [contributor guidelines](https://github.com/roboflow/trackers/blob/release/stable/CONTRIBUTING.md) to get started.

## License

The code is released under the [Apache 2.0 license](https://github.com/roboflow/trackers/blob/release/stable/LICENSE).
