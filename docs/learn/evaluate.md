# Evaluate Tracking Results

This guide explains how to evaluate multi-object tracking (MOT) results with `trackers.eval`. It covers dataset formats, CLI and Python usage, supported metrics, and advanced options.

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Supported metrics](#supported-metrics)
- [API reference (summary)](#api-reference-summary)
- [Expected dataset format](#expected-dataset-format)
- [Dataset layouts](#dataset-layouts)
- [Evaluation datasets](#evaluation-datasets)
- [Result objects and output](#result-objects-and-output)
- [Advanced usage](#advanced-usage)

## Installation

```bash
pip install trackers
```

For full setup options, see the [Install guide](install.md).

## Quickstart

=== "Python SDK"

    ```python
    from trackers.eval import evaluate_mot_sequence, evaluate_mot_sequences

    # Single sequence evaluation
    seq_result = evaluate_mot_sequence(
        gt_path="data/gt/MOT17-02.txt",
        tracker_path="data/trackers/MOT17-02.txt",
        metrics=["CLEAR", "HOTA", "Identity"],
    )
    print(seq_result.table(columns=["MOTA", "HOTA", "IDF1", "IDSW"]))

    # Benchmark evaluation (multiple sequences)
    bench_result = evaluate_mot_sequences(
        gt_dir="data/gt",
        tracker_dir="data/trackers",
        seqmap="data/seqmap.txt",
        metrics=["CLEAR", "HOTA", "Identity"],
    )
    print(bench_result.table(columns=["MOTA", "HOTA", "IDF1", "IDSW"]))
    ```

    Example output:

    ```
    Sequence                         MOTA   HOTA   IDF1  IDSW
    ---------------------------------------------------------
    MOT17-02                         75.600 62.300 72.100   42
    ```

=== "CLI"

    ```bash
    # Single sequence
    python -m trackers.scripts eval \
      --gt data/gt/MOT17-02.txt \
      --tracker data/trackers/MOT17-02.txt \
      --metrics CLEAR HOTA Identity \
      --columns MOTA HOTA IDF1 IDSW

    # Benchmark (flat format)
    python -m trackers.scripts eval \
      --gt-dir data/gt \
      --tracker-dir data/trackers \
      --seqmap data/seqmap.txt \
      --metrics CLEAR HOTA Identity \
      --columns MOTA HOTA IDF1 IDSW
    ```

    Example output:

    ```
    Sequence                         MOTA   HOTA   IDF1  IDSW
    ---------------------------------------------------------
    MOT17-02                         75.600 62.300 72.100   42
    ```

!!! note
    Float metrics are stored as fractions (0 to 1) but displayed as percentages in tables.

See also: [Evals API](../api-evals.md), [CLI](../cli.md)

## Supported metrics

Trackers provides TrackEval-compatible metrics with numerical parity.

| Family   | Metrics                                                                 | When to use it |
|----------|-------------------------------------------------------------------------|----------------|
| CLEAR    | MOTA, MOTP, MODA, IDSW, MT, PT, ML, Frag, CLR_Re, CLR_Pr, MTR, PTR, MLR, sMOTA | Classic MOT accuracy and errors |
| HOTA     | HOTA, DetA, AssA, LocA, DetRe, DetPr, AssRe, AssPr, OWTA               | Balanced detection + association quality |
| Identity | IDF1, IDR, IDP, IDTP, IDFN, IDFP                                        | Global ID consistency over time |

!!! tip
    Use HOTA for a balanced overall score, IDF1 when identity stability matters, and MOTA for detection-heavy evaluation.

See also: [Result objects and output](#result-objects-and-output)

## API reference (summary)

### `evaluate_mot_sequence`

```python
evaluate_mot_sequence(
    gt_path,
    tracker_path,
    metrics=None,
    threshold=0.5,
) -> SequenceResult
```

- `gt_path`: Path to ground truth MOT file.
- `tracker_path`: Path to tracker MOT file.
- `metrics`: List of metric families (`CLEAR`, `HOTA`, `Identity`). Default: `["CLEAR"]`.
- `threshold`: IoU threshold for CLEAR and Identity. Default: `0.5`.

### `evaluate_mot_sequences`

```python
evaluate_mot_sequences(
    gt_dir,
    tracker_dir,
    seqmap=None,
    metrics=None,
    threshold=0.5,
    benchmark=None,
    split=None,
    tracker_name=None,
) -> BenchmarkResult
```

- `gt_dir`: Ground truth directory (flat or MOT17 layout).
- `tracker_dir`: Tracker output directory.
- `seqmap`: Optional list of sequences to evaluate.
- `benchmark`, `split`, `tracker_name`: Override auto-detection for MOT17 layouts.

See also: [Evals API](../api-evals.md)

## Expected dataset format

Both ground truth and tracker predictions must be in MOT Challenge text format. Each line represents one detection:

```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
```

Example:

```
1,1,100,200,50,80,1,-1,-1,-1
1,2,300,150,60,90,1,-1,-1,-1
2,1,105,198,50,80,1,-1,-1,-1
```

- `frame`: Frame number (1-indexed).
- `id`: Object ID (unique per track).
- `bb_left`, `bb_top`: Top-left corner of bounding box.
- `bb_width`, `bb_height`: Bounding box dimensions.
- `conf`: Confidence score. Use `1` for ground truth files.
- `x`, `y`, `z`: 3D position (use `-1` if not available).

!!! warning
    Tracker output files must also follow the same MOT text format. The evaluator does not accept JSON or COCO-style formats.

See also: [Dataset layouts](#dataset-layouts)

## Dataset layouts

Trackers supports two dataset layouts:

=== "MOT layout"

    ```
    gt_dir/
      └── {benchmark}-{split}/
          ├── sequence1/
          │   └── gt/gt.txt
          └── sequence2/
              └── gt/gt.txt
    tracker_dir/
      └── {benchmark}-{split}/
          └── {tracker_name}/data/
              ├── sequence1.txt
              └── sequence2.txt
    ```

=== "Flat layout"

    ```
    gt_dir/
      ├── sequence1.txt
      └── sequence2.txt
    tracker_dir/
      ├── sequence1.txt
      └── sequence2.txt
    ```

Flat layout with `seqmap.txt`:

```
data/
├── gt/
│   ├── sequence1.txt
│   ├── sequence2.txt
│   └── sequence3.txt
├── trackers/
│   ├── sequence1.txt
│   ├── sequence2.txt
│   └── sequence3.txt
└── seqmap.txt
```

`seqmap.txt` lists sequence names:

```
name
sequence1
sequence2
sequence3
```

MOT-style example:

```
data/
├── SportsMOT/
│   └── val/
│       ├── sequence1/
│       │   └── gt/
│       │       └── gt.txt
│       ├── sequence2/
│       │   └── gt/
│       │       └── gt.txt
│       └── seqmaps/
│           └── val.txt
└── trackers/
    └── SportsMOT/
        └── val/
            └── my_tracker/
                └── data/
                    ├── sequence1.txt
                    └── sequence2.txt
```

!!! note
    The evaluator auto-detects flat vs MOT17 layout. For MOT17, it also infers `benchmark`, `split`, and `tracker_name` from the directory structure.

See also: [Advanced usage](#advanced-usage)

## Evaluation datasets

We currently evaluate against several standard MOT benchmarks. These will be available through a hosted dataset API in a future release. For now, download and prepare the datasets locally.

- **MOT17**: Pedestrian tracking with heavy occlusions and crowded scenes. Emphasizes identity persistence under occlusion.
- **SportsMOT**: Sports broadcast tracking with rapid motion, camera pans, and similar-looking targets. Emphasizes association under speed and appearance ambiguity.
- **SoccerNet-tracking**: Long, continuous soccer sequences with dense interactions. Emphasizes long-term ID stability.
- **DanceTrack**: Highly dynamic, fast-moving targets with frequent overlaps and motion changes. Emphasizes association under fast motion and appearance changes.

See also: [Trackers comparison](../trackers/comparison.md)

## Result objects and output

`trackers.eval` returns structured result objects with helper methods.

### SequenceResult

- Attributes: `sequence`, `CLEAR`, `HOTA`, `Identity`
- Methods: `table()`, `json()`, `to_dict()`

### BenchmarkResult

- Attributes: `sequences`, `aggregate`
- Methods: `table()`, `json()`, `save()`, `load()`, `to_dict()`

Example table output:

```
Sequence                         MOTA   HOTA   IDF1  IDSW
---------------------------------------------------------
MOT17-02                         75.600 62.300 72.100   42
```

Example JSON output:

```
{
  "sequences": {
    "MOT17-02": {
      "sequence": "MOT17-02",
      "CLEAR": {"MOTA": 0.756, "MOTP": 0.813, "...": "..."},
      "HOTA": {"HOTA": 0.623, "DetA": 0.712, "...": "..."},
      "Identity": {"IDF1": 0.721, "IDR": 0.704, "...": "..."}
    }
  },
  "aggregate": {
    "sequence": "COMBINED",
    "CLEAR": {"MOTA": 0.743, "MOTP": 0.809, "...": "..."},
    "HOTA": {"HOTA": 0.611, "DetA": 0.701, "...": "..."},
    "Identity": {"IDF1": 0.709, "IDR": 0.692, "...": "..."}
  }
}
```

!!! tip
    Use `result.table(columns=[...])` or `--columns` to limit output to the fields you care about.

See also: [Evals API](../api-evals.md)

## Advanced usage

### Auto-detection and overrides

For MOT17 layouts, `evaluate_mot_sequences` auto-detects:

- `benchmark` from the `{benchmark}-{split}` directory
- `split` (e.g., `train`, `val`, `test`)
- `tracker_name` (folder under `tracker_dir/{benchmark}-{split}/`)

If multiple options are found, pass overrides:

```python
result = evaluate_mot_sequences(
    gt_dir="data",
    tracker_dir="data/trackers",
    benchmark="MOT17",
    split="train",
    tracker_name="ByteTrack",
)
```

CLI equivalent:

```bash
python -m trackers.scripts eval \
  --gt-dir data \
  --tracker-dir data/trackers \
  --benchmark MOT17 \
  --split train \
  --tracker-name ByteTrack
```

### Thresholds

- `threshold` controls IoU matching for CLEAR and Identity.
- HOTA internally evaluates across multiple thresholds (0.05 to 0.95).

### Custom columns and JSON output

```bash
python -m trackers.scripts eval \
  --gt-dir data/gt \
  --tracker-dir data/trackers \
  --seqmap data/seqmap.txt \
  --metrics CLEAR HOTA Identity \
  --columns MOTA HOTA IDF1 IDSW \
  --output results.json
```

See also: [CLI](../cli.md), [Supported metrics](#supported-metrics)
