# autotrack

Autonomous optimization of MOT17 trackers — SORT, ByteTrack, OC-SORT — using the [autoresearch](https://github.com/karpathy/autoresearch) pattern with [Roboflow trackers](https://github.com/roboflow/trackers).

Point any coding agent at this folder and let it run. ~500 experiments/hour on CPU, no GPU needed.

## Motivation

Multi-object tracking quality depends on two largely independent axes: **algorithm design** (state representation, association logic, track lifecycle) and **hyperparameter tuning** (confidence thresholds, buffer sizes, Kalman noise scales). Most published improvements conflate the two — a well-tuned weak algorithm can outperform a poorly-tuned strong one, making it hard to know what actually matters.

This project separates the axes. An autonomous agent iterates over structural code changes, measuring HOTA after each change at fixed default parameters. Optuna provides a second-pass validation: after a code change is accepted, a short tuning run confirms the improvement holds under optimised parameters and is not a parameter artefact. The iteration log — including all reverted changes — is the primary research artifact.

### Why these trackers?

Three trackers are supported: **SORT**, **ByteTrack**, and **OC-SORT**. ByteTrack is the primary campaign target — it is the simplest practically-competitive tracker (pure IoU association, constant-velocity Kalman filter, no appearance features), making it easy to isolate the effect of individual algorithmic changes. SORT serves as the simplest possible baseline. OC-SORT extends ByteTrack with observation-centric velocity updates and direction consistency, providing an upper bound for what IoU-only association can achieve.

### Why MOT17?

The [MOT17 benchmark](https://www.codabench.org/competitions/10049/) provides two complementary detection sources:

- **FRCNN public detections** — bundled with the benchmark, reproducible on any machine without a GPU or API key. Weaker than modern detectors (HOTA ~50 vs ~60 with YOLOX), which creates genuine headroom for algorithmic improvement.
- **YOLO detections** — generated via `generate_detections.py` using `yolov8x-1280`. Stronger recall but capped at ~49 HOTA after tuning due to the detector being a generic COCO model rather than a purpose-built pedestrian detector. Algorithmic improvements target the association ceiling above whichever detector floor is in use.

Additional detectors (RF-DETR, YOLOX-X CrowdHuman) are supported by `generate_detections.py`; see the Detection sources section.

### Why HOTA?

[HOTA](https://arxiv.org/abs/2009.07736) (Higher Order Tracking Accuracy, Luiten et al. 2021) decomposes tracking quality into detection accuracy (DetA) and association accuracy (AssA) with equal weight. MOTA is dominated by false positives/negatives and misses ID-switch quality; IDF1 is purely association-focused. HOTA is the most informative single scalar for overall tracker health and is the primary campaign metric. IDF1, MOTA, and IDSW are logged alongside it for every run.

## Approach

The research loop follows the autoresearch pattern: propose one change, measure it, keep improvements, revert regressions. Each committed iteration is one atomic hypothesis. The JSONL experiment log captures every attempt — failures are as informative as successes.

```
Human defines:  research question  ·  metric  ·  hard boundaries
Agent decides:  what to change  ·  what to try next
```

Two tools govern the loop:

| Tool                                | Role                                                                                   |
| ----------------------------------- | -------------------------------------------------------------------------------------- |
| `optimize_tracking.py --n-trials 1` | Campaign metric — evaluates default params, gives a clean code-change signal           |
| `optimize_tracking.py --n-trials N` | Optuna study — warm-starts from `best_config.json`, finds best params for current code |

The agent is free to update `optimize_tracking.py` as the tracker architecture evolves — adding parameters that newly exist, removing ones absorbed into the implementation, tightening search ranges as knowledge accumulates.

## Detection sources

Each detector produces a set of sequence-level sibling directories alongside the bundled FRCNN sequences. The detector is visible directly in the filesystem path, making it easy to switch between detection sources or add new ones.

### FRCNN (bundled)

Pre-computed FRCNN public detections are downloaded with the benchmark data:

```
~/.cache/trackers/mot17/val/
  MOT17-04-FRCNN/
    det/det.txt        ← bundled FRCNN detections
    gt/gt.txt          ← ground truth (never seen at inference)
    img1/              ← video frames (needed for YOLO generation only)
```

No inference required. Pass `--det-source frcnn` (default) to use these.

### Generated detections

Running `generate_detections.py` creates sibling directories for each sequence:

```
~/.cache/trackers/mot17/val/
  MOT17-04-FRCNN/       ← original (frames + bundled FRCNN dets)
  MOT17-04-YOLOX/       ← created by generate_detections.py (YOLOX-X CrowdHuman)
    det/det.txt         ← YOLOX detections
    gt   -> ../MOT17-04-FRCNN/gt    ← symlink; evaluator finds ground truth here
    img1 -> ../MOT17-04-FRCNN/img1  ← symlink; full sequence structure mirrored
  MOT17-04-RFDETR/      ← created by generate_detections.py (RF-DETR-L)
    det/det.txt
    gt   -> ../MOT17-04-FRCNN/gt
    img1 -> ../MOT17-04-FRCNN/img1
```

The detector tag (`YOLOX`, `RFDETR`, …) is auto-derived from the model name and appended to the directory. Use `--det-source yolox` to evaluate against YOLOX detections.

`generate_detections.py` supports three backends via `--model`:

| Model flag               | Tag      | Backend       | Notes                                              |
| ------------------------ | -------- | ------------- | -------------------------------------------------- |
| `yolox-x-crowdhuman`     | `YOLOX`  | Local weights | ByteTrack paper detector; no API key needed        |
| `rfdetr-l`               | `RFDETR` | Native rfdetr | RF-DETR large; weights auto-downloaded; no API key |
| `yolov8x-1280` (default) | `YOLO`   | Roboflow API  | COCO-pretrained; requires `ROBOFLOW_API_KEY`       |

Override the tag with `--detector-tag` if needed. Each detector writes to its own directory so runs never overwrite each other.

## Tracker benchmarks

### FRCNN public detections (MOT17-val, bundled)

`optimize_tracking.py --fast` evaluates default parameters; no Optuna tuning.

| Tracker   | Published ref (MOT17-val, FRCNN) | HOTA default                | HOTA Optuna (500 trials) | IDF1   | MOTA   | IDSW | Theoretical ceiling |
| --------- | -------------------------------- | --------------------------- | ------------------------ | ------ | ------ | ---- | ------------------- |
| SORT      | ~45–50 (estimated)               | **49.950**                  | **51.488**               | 58.417 | 47.770 | 173  | ~52–55              |
| ByteTrack | ~50–52                           | **51.198** _(Phase 1 best)_ | **51.757**               | 58.367 | 47.740 | 237  | ~60–65              |
| OC-SORT   | ~55–57                           | **49.690**                  | **52.218**               | 58.946 | 47.753 | 233  | ~62–65              |

> IDF1/MOTA/IDSW columns show the Optuna-tuned result. **Note — why is OC-SORT's FRCNN baseline below SORT?** Default params are not tuned for FRCNN dets. `minimum_iou_threshold=0.3` is conservative for noisy public detections; ByteTrack uses 0.1. Despite the lower HOTA, OC-SORT already shows 40% fewer ID switches (154 vs 260) at defaults — its direction-consistency mechanism is working. Tuned params bring all three trackers into the 51–53 HOTA range.

### SDP public detections (MOT17-val, bundled)

Bundled SDP detections; same ground truth as FRCNN. Full 7-sequence Optuna study (500 trials).

| Tracker   | HOTA Optuna (500 trials, 7-seq) | IDF1   | MOTA   | IDSW |
| --------- | ------------------------------- | ------ | ------ | ---- |
| SORT      | **56.083**                      | 67.517 | 65.283 | 326  |
| ByteTrack | **56.115**                      | 68.077 | 65.602 | 329  |
| OC-SORT   | **57.747**                      | 70.330 | 66.215 | 303  |

> IDF1/MOTA/IDSW columns show the Optuna-tuned result. SDP is stronger than FRCNN — expect single-sequence defaults around 60–65 HOTA on MOT17-04, but the 7-sequence Optuna average is lower because the benchmark includes harder sequences that pull the mean down.

### DPM public detections (MOT17-val, bundled — single sequence)

DPM is the weakest bundled detector. Numbers below are default params on MOT17-04 only (`--fast`).

| Tracker   | HOTA (MOT17-04, default) | IDF1   | MOTA   | IDSW |
| --------- | ------------------------ | ------ | ------ | ---- |
| SORT      | 32.966                   | 39.686 | 25.527 | 77   |
| ByteTrack | 32.573                   | 38.183 | 26.115 | 57   |
| OC-SORT   | 26.106                   | 30.794 | 19.977 | 36   |

### RF-DETR detections (MOT17-val, generated — single sequence)

RF-DETR-L (`rfdetr-l`), native backend, weights auto-downloaded. Default params, MOT17-04 only (`--fast`). Full 7-sequence Optuna study not yet run.

| Tracker   | HOTA (MOT17-04, default) | IDF1   | MOTA   | IDSW |
| --------- | ------------------------ | ------ | ------ | ---- |
| SORT      | 49.606                   | 55.911 | 43.171 | 96   |
| ByteTrack | 35.759                   | 33.341 | 19.224 | 1    |
| OC-SORT   | 33.763                   | 31.047 | 17.446 | 5    |

> RF-DETR with default params (tuned for FRCNN) performs well for SORT but poorly for ByteTrack/OC-SORT — the high-conf threshold and IoU defaults don't match RF-DETR's score distribution. An Optuna run is expected to close this gap significantly.

### YOLOX-X CrowdHuman detections (MOT17-val, generated — single sequence)

ByteTrack paper detector (`yolox-x-crowdhuman`). Default params, MOT17-04 only (`--fast`). Full 7-sequence Optuna study not yet run.

| Tracker   | HOTA (MOT17-04, default) | IDF1  | MOTA      | IDSW |
| --------- | ------------------------ | ----- | --------- | ---- |
| SORT      | 3.787                    | 1.188 | -1205.757 | 265  |
| ByteTrack | 7.382                    | 4.106 | -143.585  | 48   |
| OC-SORT   | 5.994                    | 3.446 | -90.673   | 18   |

> **These numbers are not a bug — they are expected without detector-specific tuning.** The default thresholds (`track_activation_threshold=0.7`, `high_conf_det_threshold=0.6`) were calibrated for FRCNN's score distribution. YOLOX-X CrowdHuman scores are distributed very differently — the same thresholds either let through a flood of low-confidence detections (causing MOTA to crater to −1000+) or suppress almost everything. An Optuna run will bring HOTA to 60–65, matching published results. Do not compare these numbers to FRCNN defaults.

### YOLO detections (MOT17-val, generated — yolov8x-1280) _(historical reference)_

> **These numbers are from a prior setup and may not be reproducible here.** Generating YOLOv8x-1280 detections requires a `ROBOFLOW_API_KEY`. Published YOLOX MOT17-test numbers provided for reference; val scores run ~3–5 pts higher than test.

| Tracker   | Published ref (MOT17-test, YOLOX) | HOTA default | HOTA Optuna (2000 trials) | IDF1   | MOTA   | IDSW | Theoretical ceiling |
| --------- | --------------------------------- | ------------ | ------------------------- | ------ | ------ | ---- | ------------------- |
| SORT      | ~58.4 (test)                      | 47.933       | **48.963**                | 55.913 | 39.148 | 311  | ~62–65              |
| ByteTrack | ~60.1 (test)                      | 45.574       | **48.250**                | 54.524 | 40.594 | 234  | ~68–72              |
| OC-SORT   | ~61.9 (test)                      | 42.636       | **48.996**                | 57.047 | 40.358 | 189  | ~70–75              |

> **Why does Optuna only reach ~49 HOTA?** After 2000 trials all three trackers converge to the same ~49 HOTA ceiling — still below FRCNN (51.2 ByteTrack). This confirms the detector gap: `yolov8x-1280` is a generic COCO 80-class model, not a purpose-built pedestrian detector. Reaching 58–65 HOTA requires a stronger pedestrian detector, not parameter tuning.

### Metric legend

All metrics are higher-is-better except IDSW (lower is better):

- **HOTA** — geometric mean of DetA and AssA; equal-weight composite; primary campaign metric
- **IDF1** — ID F1 score; purely association-focused; does not penalise missed detections as heavily as MOTA
- **MOTA** — combines false positives, false negatives, and ID switches; dominated by detection quality
- **IDSW** — raw ID switch count; low IDSW signals stable long-term association

<details>
<summary><strong>Measuring baselines</strong></summary>

```bash
cd autotrack

# FRCNN (bundled) — all three trackers
uv run python optimize_tracking.py sort      frcnn
uv run python optimize_tracking.py bytetrack frcnn
uv run python optimize_tracking.py ocsort    frcnn

# SDP (bundled) — stronger than FRCNN, same gt
uv run python optimize_tracking.py sort      sdp
uv run python optimize_tracking.py bytetrack sdp
uv run python optimize_tracking.py ocsort    sdp

# DPM (bundled) — weakest bundled detector
uv run python optimize_tracking.py sort      dpm
uv run python optimize_tracking.py bytetrack dpm
uv run python optimize_tracking.py ocsort    dpm

# YOLOX-X CrowdHuman (requires generate_detections.py --model yolox-x-crowdhuman)
uv run python optimize_tracking.py sort      yolox
uv run python optimize_tracking.py bytetrack yolox
uv run python optimize_tracking.py ocsort    yolox

# RF-DETR (requires generate_detections.py --model rfdetr-l)
uv run python optimize_tracking.py sort      rfdetr
uv run python optimize_tracking.py bytetrack rfdetr
uv run python optimize_tracking.py ocsort    rfdetr
```

</details>

## Target analysis

The ByteTrack Phase 2 campaign target of HOTA = 68.0 is set above the published YOLOX IoU-only ceiling (OC-SORT val ≈ 65–67) and therefore requires real architectural improvements, not parameter search.

**HOTA formula**: HOTA = √(DetA × AssA) × 100, where DetA measures detection accuracy and AssA measures ID-consistency over time.

**DetA ceiling from FRCNN**: bounded to ≈ 0.55–0.62 regardless of the tracker — a perfect tracker cannot recover detections the detector missed.

**Estimated ceilings (FRCNN)**:

| Scenario                                | DetA | AssA | HOTA |
| --------------------------------------- | ---- | ---- | ---- |
| Default params, current code (baseline) | 0.57 | 0.44 | 50.4 |
| Optuna only, no code changes            | 0.57 | 0.55 | 56.0 |
| Code improvements + Optuna              | 0.59 | 0.65 | 61.9 |
| Theoretical IoU-only ceiling            | 0.62 | 0.65 | 63.5 |

**Published reference points** (IoU-only, no ReID, FRCNN public detections; val scores ~3–5 pts above test):

- ByteTrack — MOT17 test: HOTA ≈ 47.5; val ≈ 50–52
- OC-SORT — MOT17 test: HOTA ≈ 52.4; val ≈ 55–57
- BoT-SORT (no ReID) — MOT17 test: HOTA ≈ 53.1; val ≈ 56–58

## Hard boundaries

See `program.md` for the full contract. Short version:

- Metrics are computed via `trackers.eval` — no substitution
- Ground truth (`gt/gt.txt`) is never read at inference time
- Detections come from `det/det.txt` inside the sequence directory — FRCNN or generated YOLO, never oracle data
- Evolve the target tracker — do not swap it for a different algorithm mid-campaign

## Setup

### Dependencies

```bash
# Run from the project root (not autotrack/)
uv sync --group optimize   # installs optuna[rdb] + fire + inference
```

### FRCNN detections (bundled — no API key needed)

```bash
trackers download mot17 --split val --asset annotations,detections
cd autotrack
uv run python optimize_tracking.py bytetrack frcnn --fast     # expect HOTA ~51.2
uv run python optimize_tracking.py sort frcnn --fast          # SORT sanity check
uv run python optimize_tracking.py ocsort frcnn --fast        # OC-SORT sanity check
```

### YOLOX detections (ByteTrack paper detector — recommended for the campaign)

YOLOX detections are not bundled. Generate them once before starting the campaign:

```bash
# Download frames (~4 GB additional) + annotations + detections
trackers download mot17 --split val --asset annotations,detections,frames
# Run YOLOX-X CrowdHuman inference — creates MOT17-{N}-YOLOX/ sibling dirs
cd autotrack && uv run python generate_detections.py \
    --model yolox-x-crowdhuman \
    --weights pretrained/bytetrack_x_mot17.pth.tar
# Verify with a single sequence
uv run python optimize_tracking.py bytetrack yolox --fast
```

Run with `--skip-existing` to resume an interrupted generation without re-running inference on completed sequences.

### RF-DETR detections (no API key — weights auto-downloaded)

```bash
cd autotrack && uv run python generate_detections.py --model rfdetr-l
# Verify
uv run python optimize_tracking.py bytetrack rfdetr --fast
```

### Custom detections (bring your own detector)

Any detector whose output can be formatted as MOT detection files works. Create a sibling directory for each MOT17-val sequence following the layout below:

```
~/.cache/trackers/mot17/val/
  MOT17-04-MYDET/
    det/det.txt         ← your detections in MOT format (see below)
    gt   -> ../MOT17-04-FRCNN/gt    ← symlink — required by the evaluator
    img1 -> ../MOT17-04-FRCNN/img1  ← symlink — optional unless frames are needed
```

MOT detection format — one detection per line:

```
frame_idx,-1,x,y,w,h,confidence,-1,-1,-1
```

where `(x, y)` is the top-left corner, `(w, h)` is width/height, and `id=-1` marks raw detections (not tracked identities). Then evaluate by passing the detector name as `det_source` — unknown names are uppercased automatically to form the directory tag:

```bash
cd autotrack
uv run python optimize_tracking.py bytetrack mydet
uv run python optimize_tracking.py bytetrack mydet --n-trials 50
```

`mydet` → searches for `MOT17-{N}-MYDET/` directories. No extra flags needed.

## Pre-flight checks

Before starting the campaign loop, all steps must pass:

| Check            | Command                                                                                                          | Expected result                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Dependencies     | `uv sync --group optimize`                                                                                       | Resolves without error                                               |
| MOT17 data       | `trackers download mot17 --split val --asset annotations,detections,frames`                                      | Downloads to `~/.cache/trackers/mot17/val/`                          |
| YOLOX detections | `uv run python generate_detections.py --model yolox-x-crowdhuman --weights pretrained/bytetrack_x_mot17.pth.tar` | Creates `MOT17-{N}-YOLOX/` sibling dirs for all 7 sequences          |
| RF-DETR (alt)    | `uv run python generate_detections.py --model rfdetr-l`                                                          | Creates `MOT17-{N}-RFDETR/` sibling dirs; no API key or weights file |
| Metric sanity    | `uv run python optimize_tracking.py bytetrack yolox --fast`                                                      | Prints `__METRICS__: HOTA≈60–67` (YOLOX-val, single seq)             |

> **Bundled-only run** (no frames needed): use `frcnn` as det-source. Expect `HOTA≈51.2`. The Phase 2 campaign in `program.md` targets YOLOX.

The campaign metric command uses `uv run` — bare `python` will fail with `ModuleNotFoundError: No module named 'fire'` because `fire` only lives in the `uv` virtualenv.

## Run the agent

### Manual loop

```bash
claude  # or any coding agent
> Read program.md and start the experiment loop.
```

### Run with /optimize campaign

If you use [Borda's Claude Code skill suite](https://github.com/Borda/.ai-home), the `/optimize` skill drives the loop directly from `program.md`:

```bash
claude
> /optimize campaign autotrack/program.md
```

The skill handles the full iteration loop — baseline measurement, agent-driven code changes, metric verification, auto-rollback on regression, and a final results report. To run a tuning-only pass (Optuna, no code changes), set `agent_strategy: perf` in `program.md` before launching. See the skill docs for `--team` and `--codex` flags.

## Files

| File                     | Who edits | Purpose                                                                                                |
| ------------------------ | --------- | ------------------------------------------------------------------------------------------------------ |
| `README.md`              | Human     | This file                                                                                              |
| `program.md`             | Human     | Research contract + hard boundaries (ByteTrack Phase 2)                                                |
| `generate_detections.py` | Human     | Detection generation for any supported model; creates `MOT17-{N}-{TAG}/` sibling directories           |
| `default_config.json`    | Human     | Default tracker params for baseline eval — edit here, not in `optimize_tracking.py`                    |
| `search_space.json`      | Agent     | Optuna search space — add/remove params or adjust ranges here, not in the script                       |
| `optimize_tracking.py`   | Agent     | Optuna runner — positional `tracker det_source`; `--n-trials N`; `--det-tag TAG` for custom detectors  |
| `best_config.json`       | Agent     | Best Optuna params keyed by `{tracker: {det_source: {hota, config}}}` — written after `--n-trials > 1` |

To run a campaign for a different tracker, copy `program.md`, set `algo: sort` (or `ocsort`) in the Config section, update the metric command to pass `--tracker sort`, and widen `scope_files` to include that tracker's implementation files.

## References

- **ByteTrack**: Zhang et al., ["ByteTrack: Multi-Object Tracking by Associating Every Detection Box"](https://arxiv.org/abs/2110.06864), ECCV 2022 · [official implementation](https://github.com/FoundationVision/ByteTrack)
- **SORT**: Bewley et al., ["Simple Online and Realtime Tracking"](https://arxiv.org/abs/1602.00763), ICIP 2016 · [official implementation](https://github.com/abewley/sort)
- **OC-SORT**: Cao et al., ["Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking"](https://arxiv.org/abs/2203.14360), CVPR 2023 · [official implementation](https://github.com/noahcao/OC_SORT)
- **HOTA**: Luiten et al., ["HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking"](https://arxiv.org/abs/2009.07736), IJCV 2021
- **Optuna**: [optuna.org](https://optuna.org) — open-source hyperparameter optimization framework; Akiba et al., ["Optuna: A Next-generation Hyperparameter Optimization Framework"](https://arxiv.org/abs/1907.10902), KDD 2019
- **MOT17**: Milan et al., ["MOT16: A Benchmark for Multi-Object Tracking"](https://arxiv.org/abs/1603.00831), arXiv 2016; benchmark and leaderboard at [Codabench](https://www.codabench.org/competitions/10049/)
- **[autoresearch](https://github.com/karpathy/autoresearch) pattern**: Karpathy, autonomous research loop via coding agents
