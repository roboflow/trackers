# CLI

This page covers the Trackers command-line interface for evaluation. The CLI is a thin wrapper around `trackers.eval` and is useful for quick checks and CI pipelines.

## Evaluate a single sequence

```bash
python -m trackers.scripts eval \
  --gt data/gt/MOT17-02.txt \
  --tracker data/trackers/MOT17-02.txt \
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
    Float metrics are printed as percentages with 3 decimal places to match TrackEval output.

## Evaluate a benchmark (multiple sequences)

```bash
python -m trackers.scripts eval \
  --gt-dir data/SportsMOT/val \
  --tracker-dir data/trackers/SportsMOT/val/my_tracker/data \
  --metrics CLEAR HOTA Identity
```

### Common options

- `--metrics CLEAR HOTA Identity`: choose one or more metric families.
- `--threshold 0.5`: IoU threshold for CLEAR and Identity.
- `--columns MOTA HOTA IDF1 IDSW`: limit output columns.
- `--output results.json`: save results to a JSON file.

See also: [Evaluate guide](learn/evaluate.md), [Evals API](api-evals.md)
