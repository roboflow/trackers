---
title: Evaluate Tracker Performance — HOTA, IDF1, MOTA Metrics | Trackers
description: Measure tracker accuracy with standard MOT metrics — HOTA, IDF1, and MOTA — using Roboflow Trackers built-in evaluation pipeline on MOT benchmark datasets.
---

# Evaluate Trackers

Measure tracker quality with standard MOT metrics to get reproducible scores for development and benchmarking.

**What you'll learn:**

- Download ground-truth annotations and detections for evaluation
- Run tracking on pre-computed detections
- Evaluate tracking results against ground truth

<video width="100%" controls autoplay muted loop>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/trackers/docs/datasets/MOT17_MOT17-04-DPM-1280x720.mp4" type="video/mp4">
</video>
<p align="center" style="margin-top: -0.4em;"><small>Visualization of ground-truth annotations for MOT17.</small></p>

---

## Install

Get started by installing the package.

```text
pip install trackers
```

For more options, see the [install guide](install.md).

---

## Download Data

Download ground-truth annotations and detections for MOT17 validation — see the [download guide](download.md) for the full command and output layout. This guide assumes data lands at `./data/mot17/val/<sequence>/{det,gt}/det.txt` and `gt.txt`, matching that guide's default output.

---

## Run Tracking

Feed the pre-computed detections into a tracker and write the results to a file for evaluation. Pass `--detection.mot_file` to provide precomputed MOT detector output and `--output.mot_results` to save the tracker output in MOT format — see the [track guide](track.md) for more on `trackers track`.

```text
trackers track \
    --detection.mot_file ./data/mot17/val/MOT17-02-FRCNN/det/det.txt \
    --tracker bytetrack \
    --output.mot_results results/MOT17-02-FRCNN.txt
```

---

## Evaluate

Compare the tracker output against ground truth to compute standard MOT metrics.

```text
trackers eval \
    --gt ./data/mot17/val/MOT17-02-FRCNN/gt/gt.txt \
    --predictions results/MOT17-02-FRCNN.txt \
    --metrics '[CLEAR,HOTA,Identity]' \
    --columns '[MOTA,HOTA,IDF1]'
```

**Output:**

```
                                MOTA    HOTA    IDF1
----------------------------------------------------
gt                            30.192  35.475  38.515
```

---

## Data Format

Ground-truth and tracker output files use the MOT Challenge text format.

```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
```

**Example:**

```
1,1,100,200,50,80,1,-1,-1,-1
1,2,300,150,60,90,1,-1,-1,-1
2,1,105,198,50,80,1,-1,-1,-1
```

Each line contains the frame number, object ID, bounding box (`left`, `top`, `width`, `height`), confidence score, and 3D position (set to `-1` when unused).

---

## Multi-Sequence Evaluation

Evaluate all sequences at once and get per-sequence results plus a combined aggregate.

```text
trackers eval \
    --gt_dir ./data/mot17/val \
    --predictions_dir results \
    --metrics '[CLEAR,HOTA,Identity]' \
    --columns '[MOTA,HOTA,IDF1]' \
    --output results.json
```

**Output:**

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

Use `--output` to save the full results to a JSON file for later analysis.

---

## CLI Reference

All arguments accepted by `trackers eval`.

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
      <td><code>--gt</code></td>
      <td>Path to a single ground-truth file in MOT format.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--predictions</code></td>
      <td>Path to a single tracker predictions file in MOT format. Formerly <code>--tracker</code>.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--gt_dir</code></td>
      <td>Directory containing ground-truth files for multi-sequence evaluation.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--predictions_dir</code></td>
      <td>Directory containing tracker prediction files for multi-sequence evaluation. Formerly <code>--tracker_dir</code>.</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>--seqmap</code></td>
      <td>Sequence map file listing sequences to evaluate. If omitted, all sequences in the directory are evaluated.</td>
      <td>all</td>
    </tr>
    <tr>
      <td><code>--metrics</code></td>
      <td>Metric families to compute. Options: <code>CLEAR</code>, <code>HOTA</code>, <code>Identity</code>.</td>
      <td><code>CLEAR</code></td>
    </tr>
    <tr>
      <td><code>--threshold</code></td>
      <td>IoU threshold for CLEAR and Identity matching. HOTA evaluates across multiple thresholds internally.</td>
      <td><code>0.5</code></td>
    </tr>
    <tr>
      <td><code>--columns</code></td>
      <td>Metric columns to display. If omitted, all columns for the selected metrics are shown.</td>
      <td>auto</td>
    </tr>
    <tr>
      <td><code>--output</code></td>
      <td>Save results to a JSON file at the given path.</td>
      <td>none</td>
    </tr>
  </tbody>
</table>
