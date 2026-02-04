# Evaluate Trackers

Measure how well your multi-object tracker performs using standard MOT metrics (CLEAR, HOTA, Identity). Get clear, reproducible scores for development, comparison, and publication.

**What you'll learn:**

- Evaluate single and multi-sequence tracking results
- Interpret HOTA, MOTA, and IDF1 scores
- Prepare datasets in MOT Challenge format

---

## Installation

```bash
pip install trackers
```

For alternative methods, see the [Install guide](install.md).

---

## Quickstart

Evaluate a single sequence by pointing to your ground-truth and tracker files, then view the results as a formatted table.

=== "Python"

    ```python
    from trackers.eval import evaluate_mot_sequence

    result = evaluate_mot_sequence(
        gt_path="data/gt/MOT17-02-FRCNN.txt",
        tracker_path="data/trackers/MOT17-02-FRCNN.txt",
        metrics=["CLEAR", "HOTA", "Identity"],
    )

    print(result.table(columns=["MOTA", "HOTA", "IDF1", "IDSW"]))
    ```

    **Output:**

    ```
    Sequence                        MOTA    HOTA    IDF1  IDSW
    ----------------------------------------------------------
    MOT17-02-FRCNN                75.600  62.300  72.100    42
    ```

=== "CLI"

    ```bash
    trackers eval \
        --gt data/gt/MOT17-02-FRCNN.txt \
        --tracker data/trackers/MOT17-02-FRCNN.txt \
        --metrics CLEAR HOTA Identity \
        --columns MOTA HOTA IDF1 IDSW
    ```

    **Output:**

    ```
    Sequence                        MOTA    HOTA    IDF1  IDSW
    ----------------------------------------------------------
    MOT17-02-FRCNN                75.600  62.300  72.100    42
    ```

---

## Data Format

Ground truth and tracker files use MOT Challenge text format — a simple comma-separated .txt file where each line describes one detection.

```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
```

**Example:**

```
1,1,100,200,50,80,1,-1,-1,-1
1,2,300,150,60,90,1,-1,-1,-1
2,1,105,198,50,80,1,-1,-1,-1
```

**Fields:**

- `frame` — Frame number (1-indexed)
- `id` — Unique object ID per track
- `bb_left`, `bb_top` — Top-left bounding box corner
- `bb_width`, `bb_height` — Bounding box dimensions
- `conf` — Confidence score (1 for ground truth)
- `x`, `y`, `z` — 3D coordinates (-1 if unused)

---

## Directory Layouts

The evaluator automatically detects whether you're using a flat or MOT-style structure. It also tries to infer benchmark name, split, and tracker name from folder names.

=== "MOT Layout"

    Standard MOT Challenge nested structure.

    ```
    data/
    ├── MOT17-train/
    │   ├── MOT17-02-FRCNN/
    │   │   └── gt/gt.txt
    │   ├── MOT17-04-FRCNN/
    │   │   └── gt/gt.txt
    │   └── MOT17-05-FRCNN/
    │       └── gt/gt.txt
    └── trackers/
        └── MOT17-train/
            └── ByteTrack/
                └── data/
                    ├── MOT17-02-FRCNN.txt
                    ├── MOT17-04-FRCNN.txt
                    └── MOT17-05-FRCNN.txt
    ```

    **Python**

    ```python
    from trackers.eval import evaluate_mot_sequences

    result = evaluate_mot_sequences(
        gt_dir="data",
        tracker_dir="data/trackers",
        benchmark="MOT17",
        split="train",
        tracker_name="ByteTrack",
    )
    ```

    **CLI**

    ```bash
    trackers eval \
      --gt-dir data \
      --tracker-dir data/trackers \
      --benchmark MOT17 \
      --split train \
      --tracker-name ByteTrack
    ```

=== "Flat Layout"

    One `.txt` file per sequence, placed directly in the directories.

    ```
    data/
    ├── gt/
    │   ├── MOT17-02-FRCNN.txt
    │   ├── MOT17-04-FRCNN.txt
    │   └── MOT17-05-FRCNN.txt
    └── trackers/
        ├── MOT17-02-FRCNN.txt
        ├── MOT17-04-FRCNN.txt
        └── MOT17-05-FRCNN.txt
    ```

    **Python**

    ```python
    from trackers.eval import evaluate_mot_sequences

    result = evaluate_mot_sequences(
        gt_dir="data/gt",
        tracker_dir="data/trackers",
    )
    ```
    
    **CLI**

    ```bash
    trackers eval --gt-dir data/gt --tracker-dir data/trackers
    ```

---

## Multi-Sequence Evaluation

Run evaluation across many sequences and get both per-sequence results and a combined aggregate.

=== "CLI"

    ```bash
    trackers eval \
        --gt-dir data/gt \
        --tracker-dir data/trackers \
        --metrics CLEAR HOTA Identity \
        --columns MOTA HOTA IDF1 \
        --output results.json
    ```

=== "Python"

    ```python
    from trackers.eval import evaluate_mot_sequences

    result = evaluate_mot_sequences(
        gt_dir="data/gt",
        tracker_dir="data/trackers",
        metrics=["CLEAR", "HOTA", "Identity"],
    )

    print(result.table(columns=["MOTA", "HOTA", "IDF1"]))
    ```

**Output:**

```
Sequence                        MOTA    HOTA    IDF1
----------------------------------------------------
MOT17-02-FRCNN                75.600  62.300  72.100
MOT17-04-FRCNN                78.200  65.100  74.800
MOT17-05-FRCNN                71.300  59.800  69.200
----------------------------------------------------
COMBINED                      75.033  62.400  72.033
```