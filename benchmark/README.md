# MOT benchmark workflow

Makefile-driven pipeline for tuning, local evaluation, test-set submission, and Codabench upload using the trackers CLI.

Requires **`develop`** (trackers ≥ 2.3 with `track`, `eval`, `tune` CLIs). Install the repo editable from the parent directory:

```bash
cd benchmark
make setup DATASET=mot17
```

## Data layout

Place benchmark assets under `benchmark/data/` (or set `DATA_ROOT=`):

```
data/
  mot17/MOT17_yolox_dets/{val,test}/...
  sportsmot/sportsmot_yolox_dets/{val,test}/...
  dancetrack/dancetrack_yolox_dets/{train,val,test}/...
```

Use `trackers download` or your existing YOLOX det trees. For BoT-SORT CMC, also provide frame directories (`mot17/val`, `dancetrack/test_images`, etc.).

## Commands

```bash
make eval TRACKER=sort DATASET=mot17
make submit TRACKER=sort DATASET=dancetrack
make upload-codabench TRACKER=sort DATASET=mot17 CODABENCH_TOKEN=...
```

| Dataset | Codabench | Phase |
|---|---|---|
| MOT17 | [10049](https://www.codabench.org/competitions/10049/) | 16382 |
| SportsMOT | [13077](https://www.codabench.org/competitions/13077/) | 21402 |
| DanceTrack | [14885](https://www.codabench.org/competitions/14885/) | 24635 |

## Implementation notes

- **`make submit`** uses `scripts/submit_yolox.py` with library defaults (or `best_params.json`), not the shared `trackers track` CLI defaults.
- **`make eval`** passes explicit per-tracker flags via `scripts/tracker_flags.py`.
