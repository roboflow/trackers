<div align="center">
    <img width="200" src="https://raw.githubusercontent.com/roboflow/trackers/refs/heads/release/stable/docs/assets/logo-trackers-violet.svg" alt="trackers logo">
    <h1>trackers</h1>
    <p>Plug-and-play multi-object tracking for any detection model.</p>

[![version](https://badge.fury.io/py/trackers.svg)](https://badge.fury.io/py/trackers) [![downloads](https://img.shields.io/pypi/dm/trackers)](https://pypistats.org/packages/trackers) [![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/trackers/blob/release/stable/LICENSE) [![python-version](https://img.shields.io/pypi/pyversions/trackers)](https://badge.fury.io/py/trackers) [![try it](https://img.shields.io/badge/try_it-Hugging%20Face%20Playground-yellow)](https://huggingface.co/spaces/roboflow/trackers) [![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

</div>

`trackers` gives you clean-room, benchmarked implementations of SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU, and McByte — so occlusions, fast motion, and moving cameras stop being your problem to solve from scratch. It speaks `supervision.Detections` natively, slotting into any detector you already use — YOLO, DETR, RT-DETR, or anything else — without glue code. One consistent interface, whether you're a researcher comparing algorithms, an engineer shipping a production pipeline, or a hobbyist building something cool. Requires Python ≥ 3.10.

## Why trackers?

- **Clean-room implementations.** Every algorithm is re-implemented from the original paper — not vendored or wrapped. You can read it, understand it, and modify it.
- **Apache 2.0, no copyleft.** Ship it inside closed-source products — unlike AGPL-3.0 alternatives such as BoxMOT.
- **Detector-agnostic.** Works with YOLO, DETR, RT-DETR, or any model that produces bounding boxes. No inference library required or assumed.
- **`supervision.Detections` native.** Plugs directly into the supervision ecosystem. Pass detections in, get tracked detections back — zero glue code.
- **Benchmarked across four datasets.** MOT17, SportsMOT, SoccerNet, and DanceTrack — at default parameters and after hyperparameter tuning (McByte: defaults only, by design), so you know what to expect before you deploy.
- **Tunable with one extra.** Optuna-based hyperparameter search via `trackers tune` (`pip install "trackers[tune]"`) so you can optimize for your specific scene and detector.
- **Camera motion compensation.** BoT-SORT and McByte handle moving cameras natively, keeping track IDs stable even when the whole frame shifts.
- **Optional appearance ReID.** BoT-SORT can fuse visual embeddings with motion for harder association scenes: install `trackers[reid]` (pulls in the [`reid`](https://github.com/roboflow/re-ID) package), pass a `reid.ReIDModel` as `reid_model`, and supply `frame=` to `update()`.

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

For more options, see the [install guide](https://trackers.roboflow.com/develop/guides/install/).

[![Watch: Building Real-Time Multi-Object Tracking with RF-DETR and Trackers](https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/roboflow-piotr-rf-detr-trackers-v1b-callout.png)](https://www.youtube.com/watch?v=u0k2dTZ0vfs)

## Quick Start

Add tracking to your existing detection pipeline in a few lines. Every tracker shares the same `update(detections, frame=None)` interface, so switching algorithms later is a one-line change. The example below uses the `inference` package as the detector (`pip install inference` — it is not part of the base `trackers` install) — swap it for any detector that returns `supervision.Detections`.

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

For more examples, see the [tracking guide](https://trackers.roboflow.com/develop/guides/track/).

## Track from CLI

Prefer the terminal? Point `trackers track` at a video, webcam feed, RTSP stream, or image directory and it handles detection, tracking, and annotated output in one command — no Python script required.

```bash
trackers track \
    --source video.mp4 \
    --output.video output.mp4 \
    --detection.model rfdetr-medium \
    --tracker bytetrack \
    --show.labels \
    --show.trajectories
```

For all CLI options, see the [tracking guide](https://trackers.roboflow.com/develop/guides/track/).

## Algorithms

Each tracker below is a faithful implementation of its original paper's motion and association pipeline; appearance/ReID branches are not included where papers offer them — see each tracker's docs page for the exact scope. Pick the one that fits your scene, or run the benchmark to find out which performs best on your data.

<!-- BENCH-XREF copy-of: [docs/evaluations/results.md](docs/evaluations/results.md) mot17/sportsmot/soccernet/dancetrack-default tables, HOTA column only, all 6 rows (SORT/ByteTrack/OC-SORT/BoT-SORT/C-BIoU/McByte). Also duplicated in [docs/index.md](docs/index.md)'s Algorithms table (all 6 rows). Update results.md first, then mirror both copies. -->

| Algorithm | Description | MOT17 HOTA | SportsMOT HOTA | SoccerNet HOTA | DanceTrack HOTA |
| :-------------------------------------------: | :------------------------------------------------------------------------------: | :--------: | :------------: | :------------: | :-------------: |
| [SORT](https://arxiv.org/abs/1602.00763) | Kalman filter + Hungarian matching baseline. | 58.4 | 70.8 | 81.6 | 47.2 |
| [ByteTrack](https://arxiv.org/abs/2110.06864) | Two-stage association using high and low confidence detections. | 60.1 | 73.0 | 84.0 | 53.3 |
| [OC-SORT](https://arxiv.org/abs/2203.14360) | Observation-centric recovery for lost tracks. | 61.9 | 71.7 | 78.4 | 54.1 |
| [BoT-SORT](https://arxiv.org/abs/2206.14651) | Camera motion compensation. | 63.7 | 73.8 | 84.5 | 57.8 |
| [C-BIoU](https://arxiv.org/abs/2211.14317) | Cascaded buffered IoU matching for fast or irregular motion. | 63.0 | 73.1 | 82.6 | 56.7 |
| [McByte](https://arxiv.org/abs/2506.01373) | Mask-conditioned tracking — adds propagated SAM/Cutie masks as a matching cue.\* | **64.1** | **76.5** | **85.0** | **67.2** |

\*McByte needs optional heavyweight deps (`torch`, SAM, Cutie) not installed by default. It tops HOTA on all four benchmarks above — see the [McByte docs](https://trackers.roboflow.com/develop/trackers/mcbyte/) for setup.

All scores use default parameters on the standard split. Detections come from a YOLOX detector (MOT17, SportsMOT, DanceTrack) or oracle ground-truth boxes (SoccerNet) — absolute numbers shift with detector quality. See the [tracker comparison](https://trackers.roboflow.com/develop/evaluations/results/) for tuned numbers and methodology.

## Evaluate

Once you have tracking results, you want to know how good they are. `trackers eval` computes CLEAR, HOTA, and Identity metrics against ground-truth annotations and prints a per-sequence breakdown alongside the combined score.

```bash
trackers eval \
    --gt_dir ./data/mot17/val \
    --predictions_dir results \
    --metrics '[CLEAR,HOTA,Identity]' \
    --columns '[MOTA,HOTA,IDF1]'
```

Example output — a tracker run on MOT17 val using the dataset's public detections; absolute numbers depend on detector quality (see [Detection Quality Matters](https://trackers.roboflow.com/develop/guides/detection-quality/)), so they are not comparable to the YOLOX-based benchmark table above.

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

For the full evaluation workflow, see the [evaluation guide](https://trackers.roboflow.com/develop/evaluations/evaluate/).

## Download Datasets

Need benchmark data to evaluate against? `trackers download` pulls MOT17 and SportsMOT with a single command, handling splits and assets selectively so you only download what you need.

```bash
trackers download --name mot17 \
    --split val \
    --asset annotations,detections
```

| Dataset | Description | Splits | Assets | License |
| :---------: | :---------------------------------------------------------------------: | :--------------------: | :-------------------------------------: | :-------------: |
| `mot17` | Pedestrian tracking with crowded scenes and frequent occlusions. | `train`, `val`, `test` | `frames`, `annotations`\*, `detections` | CC BY-NC-SA 3.0 |
| `sportsmot` | Sports broadcast tracking with fast motion and similar-looking targets. | `train`, `val`, `test` | `frames`, `annotations`\* | CC BY 4.0 |

\*Annotations are available for `train` and `val` only — `test` splits withhold ground truth for held-out evaluation (SportsMOT `test` ships `frames` only).

For more download options, see the [download guide](https://trackers.roboflow.com/develop/evaluations/download/).

## Where to go next

- **New to tracking?** Start with the [tracking guide](https://trackers.roboflow.com/develop/guides/track/) — it walks through the Python API and CLI end to end.
- **Want benchmarks?** The [tracker comparison](https://trackers.roboflow.com/develop/evaluations/results/) covers all six algorithms across all four datasets, at default and tuned parameters, with guidance on which to pick for your scene.
- **Building a research pipeline?** The [evaluation guide](https://trackers.roboflow.com/develop/evaluations/evaluate/) and [download guide](https://trackers.roboflow.com/develop/evaluations/download/) cover the full offline benchmarking workflow.
- **Full API reference** → [trackers.roboflow.com](https://trackers.roboflow.com)
- **Try without installing** → [Hugging Face Playground](https://huggingface.co/spaces/roboflow/trackers) — see it in action in your browser before writing any code.
- **Questions?** Find us on [Discord](https://discord.gg/GbfgXGJ8Bk).

Releases follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — see the [CHANGELOG](https://github.com/roboflow/trackers/blob/release/stable/CHANGELOG.md) for release history.

## Citation

If you use trackers in academic work, please cite the library:

<details>
<summary>BibTeX</summary>

```bibtex
@software{roboflow_trackers,
  author  = {{Roboflow}},
  title   = {Roboflow Trackers},
  url     = {https://github.com/roboflow/trackers},
  year    = {2025},
  license = {Apache-2.0}
}
```

</details>

For citations of the individual tracking algorithms, follow the paper links in the [Algorithms](#algorithms) table.

## Contributing

We welcome contributions. Read our [contributor guidelines](https://github.com/roboflow/trackers/blob/release/stable/CONTRIBUTING.md) to get started.

## License

The code is released under the [Apache 2.0 license](https://github.com/roboflow/trackers/blob/release/stable/LICENSE).
