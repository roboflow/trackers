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

By default, the **first trial** evaluates a baseline parameter set before Optuna
samples further combinations. That trial counts toward `--n-trials` / `n_trials`.
Set `enqueue_defaults=False` on `Tuner` to disable this behavior.

For each `search_space` key, the baseline uses the tracker's default
when it lies within the search space.

Options that are not tuned (or differ from `__init__`) are set with
`fixed_params` on `Tuner`. They apply to every trial, including the baseline,
override the same key in `search_space` if present, and are returned from
`run()` merged into the best parameter dict.

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
        --gt-dir ./data/gt \
        --detections-dir ./data/detections \
        --fixed-params '{"enable_cmc": false}'
    ```

Images are read from `{images_dir}/{sequence}/img1/` using MOT-style stems:
6-digit (`000001.jpg`, MOT17/SportsMOT) or 8-digit (`00000001.jpg`, DanceTrack),
plus common extensions (`.jpg`, `.png`, …).

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

For detections, use `id=-1`. For more details on the format and evaluation workflow, see the [evaluation guide](evaluate.md).

---

## Quickstart

=== "CLI"

    Tune ByteTrack and optimize `HOTA`.

    ```text
    trackers tune \
        --tracker bytetrack \
        --gt-dir ./data/gt \
        --detections-dir ./data/detections \
        --objective HOTA \
        --metrics CLEAR HOTA Identity \
        --n-trials 50 \
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
        --gt-dir ./data/gt \
        --detections-dir ./data/detections \
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
      <td><code>--gt-dir</code></td>
      <td>Directory with ground-truth MOT files (<code>{sequence}.txt</code>).</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--detections-dir</code></td>
      <td>Directory with detection MOT files (<code>{sequence}.txt</code>), one file per sequence.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--objective</code></td>
      <td>Metric to maximize: <code>MOTA</code>, <code>HOTA</code>, or <code>IDF1</code>.</td>
      <td><code>HOTA</code></td>
    </tr>
    <tr>
      <td><code>--n-trials</code></td>
      <td>Number of Optuna trials to run.</td>
      <td><code>100</code></td>
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
      <td>all files in <code>--detections-dir</code></td>
    </tr>
    <tr>
      <td><code>--seed</code></td>
      <td>Random seed for Optuna's TPE sampler (reproducible sampled trials).</td>
      <td>None</td>
    </tr>
    <tr>
      <td><code>--output</code>, <code>-o</code></td>
      <td>Path to save best parameters as JSON.</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
