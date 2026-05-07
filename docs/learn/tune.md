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

---

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
      <td>Tracker ID to tune (for example: <code>bytetrack</code>, <code>sort</code>, <code>ocsort</code>).</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--gt-dir</code></td>
      <td>Directory containing ground-truth MOT files.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--detections-dir</code></td>
      <td>Directory containing pre-computed detection MOT files.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--objective</code></td>
      <td>Metric to maximize. Options: <code>MOTA</code>, <code>HOTA</code>, <code>IDF1</code>.</td>
      <td><code>HOTA</code></td>
    </tr>
    <tr>
      <td><code>--n-trials</code></td>
      <td>Number of Optuna trials.</td>
      <td><code>100</code></td>
    </tr>
    <tr>
      <td><code>--metrics</code></td>
      <td>Metric families to compute. Options: <code>CLEAR</code>, <code>HOTA</code>, <code>Identity</code>.</td>
      <td><code>CLEAR</code></td>
    </tr>
    <tr>
      <td><code>--threshold</code></td>
      <td>IoU threshold used by CLEAR and Identity matching.</td>
      <td><code>0.5</code></td>
    </tr>
    <tr>
      <td><code>--seqmap</code></td>
      <td>Optional sequence list file; only listed sequences are tuned.</td>
      <td>all sequences in detections dir</td>
    </tr>
    <tr>
      <td><code>--output</code></td>
      <td>Path to save best parameters as JSON.</td>
      <td>none</td>
    </tr>
  </tbody>
</table>
