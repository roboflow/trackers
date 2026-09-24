---
title: Tune Tracker Hyperparameters — Optuna Guide | Trackers
description: Optimize tracker hyperparameters with the Trackers Tuner class and CLI using Optuna, MOT-format detections, and evaluation metrics like HOTA, MOTA, and IDF1.
---

# Tune Trackers

Use Optuna to tune tracker hyperparameters automatically and maximize your target metric on MOT-format evaluation data.

**What you'll learn:**

- Install tuning dependencies
- Prepare ground truth and detection files for tuning
- Run tuning from CLI and Python
- Save and apply the best parameter set

---

## Install

Install the tuning extra to enable Optuna-based hyperparameter search.

```text
pip install "trackers[tune]"
```

For more options, see the [install guide](install.md).

---

## Prepare Data

The tuner needs matching MOT files for ground truth and detections.

By default, the **first trial** evaluates a baseline parameter set before Optuna samples further combinations. That trial counts toward `--n-trials` / `n_trials`. Set `enqueue_defaults=False` on `Tuner` to disable this behavior.

For each `search_space` key, the baseline uses the tracker's default when it lies within the search space.

Options that are not tuned (or differ from `__init__`) are set with `fixed_params` on `Tuner`. They apply to every trial, including the baseline, override the same key in `search_space` if present, and are returned from `run()` merged into the best parameter dict.

=== "Python"

    ```python
    from trackers.tune import Tuner

    # Detection-only BoTSORT (no frames, CMC off)
    tuner = Tuner(
        tracker_id="botsort",
        gt_dir="./data/gt",
        detections_dir="./data/detections",
        fixed_params={"enable_cmc": False},
        n_trials=50,
    )

    # BoTSORT with CMC (MOT-style images required)
    tuner = Tuner(
        tracker_id="botsort",
        gt_dir="./data/gt",
        detections_dir="./data/detections",
        images_dir="./data/images",
        fixed_params={"enable_cmc": True},
        n_trials=50,
    )
    ```

=== "CLI"

    ```text
    trackers tune \
        --tracker botsort \
        --gt_dir ./data/gt \
        --detections_dir ./data/detections \
        --fixed_params '{"enable_cmc": false}'
    ```

Images are read from `{images_dir}/{sequence}/img1/` using MOT-style stems: 6-digit (`000001.jpg`, MOT17/SportsMOT) or 8-digit (`00000001.jpg`, DanceTrack), plus common extensions (`.jpg`, `.png`, …).

```text
data
├── gt
│   ├── MOT17-02-FRCNN.txt
│   ├── MOT17-04-FRCNN.txt
│   └── ...
└── detections
    ├── MOT17-02-FRCNN.txt
    ├── MOT17-04-FRCNN.txt
    └── ...
```

Each sequence must exist in both directories with the same filename (`{sequence}.txt`).

Use MOT format lines:

```text
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
```

For detections, use `id=-1`. For more details on the format and evaluation workflow, see the [evaluation guide](../evaluations/evaluate.md).

---

## Quickstart

=== "CLI"

    Tune ByteTrack and optimize `HOTA`.

    ```text
    trackers tune \
        --tracker bytetrack \
        --gt_dir ./data/gt \
        --detections_dir ./data/detections \
        --objective HOTA \
        --metrics '[CLEAR,HOTA,Identity]' \
        --n_trials 50 \
        --output ./results/bytetrack-best.json
    ```

=== "Python"

    Run the same tuning flow with the `Tuner` class.

    ```python
    from trackers.tune import Tuner

    tuner = Tuner(
        tracker_id="bytetrack",
        gt_dir="./data/gt",
        detections_dir="./data/detections",
        objective="HOTA",
        metrics=["CLEAR", "HOTA", "Identity"],
        n_trials=50,
        seed=42,
    )

    best_params = tuner.run()
    print(best_params)
    ```

---

## Tune a Sequence Subset

Use a seqmap file when you want to tune on a specific subset of sequences.

```text
# seqmap.txt
MOT17-02-FRCNN
MOT17-04-FRCNN
MOT17-09-FRCNN
```

=== "CLI"

    ```text
    trackers tune \
        --tracker bytetrack \
        --gt_dir ./data/gt \
        --detections_dir ./data/detections \
        --seqmap ./seqmap.txt
    ```

=== "Python"

    ```python
    from trackers.tune import Tuner

    tuner = Tuner(
        tracker_id="bytetrack",
        gt_dir="./data/gt",
        detections_dir="./data/detections",
        seqmap="./seqmap.txt",
        n_trials=25,
    )

    best_params = tuner.run()
    print(best_params)
    ```

---

## Tune ReID Thresholds

Pass an encoder to BoT-SORT and the search adds `reid_appearance_threshold` and `reid_proximity_threshold`. Tune the two together: a looser proximity gate usually needs a stricter appearance threshold. To keep tuned motion parameters and search only these two, fix everything else. `images_dir` is required, because the encoder reads the frames.

Every trial replays the same detections, so `cache_embeddings=True` embeds each one once and reuses it, which saves most of a study's time. The embeddings then stay in memory for the whole study, about 1 GB per 500k detections, which is why it is off by default.

=== "CLI"

    ```text
    trackers tune \
        --tracker botsort \
        --gt_dir ./data/gt \
        --detections_dir ./data/detections \
        --images_dir ./data/images \
        --reid.model fastreid_mot17_sbs50 \
        --fixed_params '{"lost_track_buffer": 30, "minimum_consecutive_frames": 2, "minimum_iou_threshold_first_assoc": 0.2, "minimum_iou_threshold_second_assoc": 0.5, "minimum_iou_threshold_unconfirmed_assoc": 0.2, "high_conf_det_threshold": 0.5, "track_activation_threshold": 0.6, "cmc_downscale": 2}' \
        --n_trials 20
    ```

=== "Python"

    ```python
    from reid import ReIDModel
    from trackers.tune import Tuner

    encoder = ReIDModel.from_pretrained("fastreid_mot17_sbs50")
    tuned_motion = {
        "lost_track_buffer": 30,
        "minimum_consecutive_frames": 2,
        "minimum_iou_threshold_first_assoc": 0.2,
        "minimum_iou_threshold_second_assoc": 0.5,
        "minimum_iou_threshold_unconfirmed_assoc": 0.2,
        "high_conf_det_threshold": 0.5,
        "track_activation_threshold": 0.6,
        "cmc_downscale": 2,
    }

    tuner = Tuner(
        tracker_id="botsort",
        gt_dir="./data/gt",
        detections_dir="./data/detections",
        images_dir="./data/images",
        fixed_params={**tuned_motion, "reid_model": encoder},
        n_trials=20,
    )
    best_params = tuner.run()
    ```

Each detection is embedded once and reused by later trials, so the first trial is the slow one.

---

## Use Best Parameters

Apply tuned values by unpacking the saved JSON dictionary into your tracker constructor.

```python
import json

from trackers import ByteTrackTracker

with open("./results/bytetrack-best.json", "r", encoding="utf-8") as f:
    best_params = json.load(f)

tracker = ByteTrackTracker(**best_params)
```

## CLI Reference

All arguments accepted by `trackers tune`.

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
      <td>Tracker name to tune. Common values: <code>bytetrack</code>, <code>sort</code>, <code>ocsort</code>.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--gt_dir</code></td>
      <td>Directory with ground-truth MOT files (<code>{sequence}.txt</code>).</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--detections_dir</code></td>
      <td>Directory with detection MOT files (<code>{sequence}.txt</code>), one file per sequence.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--objective</code></td>
      <td>Metric to maximize: <code>MOTA</code>, <code>HOTA</code>, or <code>IDF1</code>.</td>
      <td><code>HOTA</code></td>
    </tr>
    <tr>
      <td><code>--n_trials</code></td>
      <td>Number of Optuna trials to run.</td>
      <td><code>100</code></td>
    </tr>
    <tr>
      <td><code>--fixed_params</code></td>
      <td>Tracker constructor kwargs held constant across every trial (JSON object). Applied on top of, and taking priority over, any overlapping <code>search_space</code> key; also merged into the returned best-parameter dict.</td>
      <td>None</td>
    </tr>
    <tr>
      <td><code>--images_dir</code></td>
      <td>MOT-style image root for frame-based features such as CMC. Frames are read from <code>{images_dir}/{sequence}/img1/</code>.</td>
      <td>None</td>
    </tr>
    <tr>
      <td><code>--reid.model</code></td>
      <td>ReID encoder for BoT-SORT: alias, <code>hf://</code> URL, or local path. Adds <code>reid_appearance_threshold</code> and <code>reid_proximity_threshold</code> to the search. Requires <code>--images_dir</code>. Also accepts <code>--reid.device</code> and <code>--reid.architecture</code>.</td>
      <td>None</td>
    </tr>
    <tr>
      <td><code>--search_space</code></td>
      <td>Search-space entries that replace or extend the tracker's own for this run (JSON object, same format as the tracker's <code>search_space</code>).</td>
      <td>None</td>
    </tr>
    <tr>
      <td><code>--enqueue_defaults</code></td>
      <td>Evaluate the tracker's default parameters as the first trial before Optuna sampling begins. Negate with <code>--no_enqueue_defaults</code>.</td>
      <td><code>true</code></td>
    </tr>
    <tr>
      <td><code>--metrics</code></td>
      <td>Metric families to compute: <code>CLEAR</code>, <code>HOTA</code>, <code>Identity</code>. The family required by <code>--objective</code> is added automatically.</td>
      <td><code>CLEAR</code></td>
    </tr>
    <tr>
      <td><code>--threshold</code></td>
      <td>IoU threshold used during evaluation matching for <code>CLEAR</code> and <code>Identity</code>. Higher values make scoring stricter, lower values make it more permissive.</td>
      <td><code>0.5</code></td>
    </tr>
    <tr>
      <td><code>--seqmap</code></td>
      <td>Optional path to a sequence map file. When set, only listed sequences are tuned.</td>
      <td>all files in <code>--detections_dir</code></td>
    </tr>
    <tr>
      <td><code>--seed</code></td>
      <td>Random seed for Optuna's TPE sampler (reproducible sampled trials).</td>
      <td>None</td>
    </tr>
    <tr>
      <td><code>--output</code></td>
      <td>Path to save best parameters as JSON.</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
