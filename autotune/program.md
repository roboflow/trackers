# Campaign: Tracker algorithmic improvement on MOT17

## Goal

**Research question**: Which architectural and algorithmic changes to the `{algo}` tracker improve multi-object tracking quality on MOT17 `{det_source}` detections?

The campaign measures HOTA at default parameters after each code change. Optuna is a **validation tool**, not the goal — every candidate improvement is evaluated at default params first; Optuna confirms the signal is real and not a parameter artefact.

The agent may rewrite the Kalman state representation, redesign the association pipeline, add new constructor parameters, and modify the shared `trackers/core/sort/utils.py` utilities (used by both SORT and ByteTrack). Every change must improve `{algo}` HOTA at default parameters to be kept.

## Metric

```
command: cd autotune && uv run python optimize_tracking.py {algo} {det_source} --n-trials 1 2>&1 | grep "^__METRICS__" | grep -oE "HOTA=[0-9.]+" | cut -d= -f2
direction: higher
```

## Guard

```
command: uv run pytest test/ -m "not integration" --ignore=test/scripts -q && cd autotune && uv run python guard.py
```

## Config

`algo` and `det_source` are **defaults** substituted into `{algo}` / `{det_source}` in metric and guard commands. Override per run via clarification syntax: `/optimize run program.md "algo=sort"` or `/optimize run program.md "algo=ocsort det_source=dpm"`.

```
algo: bytetrack
det_source: sdp
max_iterations: 20
agent_strategy: ml
scope_files:
  - trackers/**
  - autotune/optimize_tracking.py
  - autotune/generate_detections.py
  - autotune/search_space.json
  - autotune/default_config.json
out_of_scope_files:
  - trackers/eval/**
  - trackers/datasets/**
compute: local
```

## Notes

### Pre-flight checks

All three setup steps must pass before starting the campaign loop:

| Check         | Command                                                                        | Expected result                                  |
| ------------- | ------------------------------------------------------------------------------ | ------------------------------------------------ |
| Dependencies  | `uv sync --group optimize`                                                     | Resolves without error                           |
| MOT17 data    | `trackers download mot17 --split val --asset annotations,detections,frames`    | Downloads to `~/.cache/trackers/mot17/val/`      |
| Metric sanity | `cd autotune && uv run python optimize_tracking.py {algo} {det_source} --fast` | Prints `__METRICS__: HOTA` within expected range |

The guard uses `best_config.json` as the regression baseline — no separate seeding step required. The guard runs all three trackers (`bytetrack`, `sort`, `ocsort`) via `optimize_tracking.py sdp --n-trials 500` and fails if any tracker's HOTA drops more than 0.5% from its stored best.

> **Custom detector** — create `MOT17-{N}-MYDET/det/det.txt` sibling dirs (see README.md Custom detections section), just pass the name as `det_source`: `uv run python optimize_tracking.py {algo} mydet` — unknown names are uppercased to form the directory tag (`MYDET`).

### Evaluation protocol

- **Primary metric**: HOTA on MOT17-val, `{det_source}` detections. Stops at `max_iterations`.
- **Secondary metrics** (logged, not gated): IDF1, MOTA, IDSW. A change that improves HOTA but worsens IDSW significantly is a warning sign — log it.
- **Fast mode** (`--fast`): single sequence (~3 s), sanity check only; campaign metric (`--n-trials 1`) always runs the full eval (~7 s, all sequences).

**Current tuned baselines (SDP, 500 Optuna trials):**

| Tracker   | Tuned HOTA | Default HOTA |
| --------- | ---------- | ------------ |
| bytetrack | 59.131     | ~53.2        |
| sort      | 56.129     | ~47.8        |
| ocsort    | 57.867     | ~52.3        |

### Hard boundaries — these invalidate the experiment if violated

1. **Do not bypass `trackers.eval`**. The evaluation calls in `optimize_tracking.py` must go through `trackers.eval` unchanged — do not substitute custom metric code.
2. **Do not modify `trackers/eval/`**. The metric computation must be identical across all iterations.
3. **No ground-truth at inference time**. The tracker sees only detector output (`det/det.txt`). It must not read from `gt/` at any point.
4. **No external features**. The detector provides bounding boxes and confidence scores only. No appearance embeddings, no depth, no optical flow at association time unless derived purely from the bounding box sequence itself.
5. **The Kalman filter must remain a proper linear Kalman filter**. Learned components (neural prediction, learned motion model) require a separate research question and are out of scope here.
6. **API changes are allowed when well-justified**. Constructor signatures and `update()` may change if the change is architecturally motivated, accompanied by a rationale comment, and `optimize_tracking.py` is updated to match. The tracker must still implement `BaseTracker` and accept `sv.Detections` input. Do not rename the public tracker classes.

### Optuna's role

Optuna is used in two places only:

1. **Pre-campaign baseline** (run once by the human before starting the loop): run `python optimize_tracking.py {algo} {det_source} --n-trials 200`, save the best param config to `best_config.json`. This gives a tuned ceiling for the *current* code.

2. **Post-change validation** (optional, agent-initiated): after a code change is *kept* by the campaign loop, the agent may run a 50-trial mini-Optuna with the new code to confirm the improvement holds under tuned params and to update `best_config.json`. If tuned params *erase* the code change's improvement, that is a negative result — log it and revert.

The campaign metric always measures at **default parameters** (~7 s per run).

### Configuration files the agent may edit

| File                   | What to change there                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| `search_space.json`    | Add/remove Optuna parameters, widen/narrow ranges, switch to log scale                                       |
| `default_config.json`  | Update baseline defaults when a new parameter is added to the code                                           |
| `optimize_tracking.py` | Update `_build_tracker` when constructor signatures change — that is the **only** reason to edit this script |

### What the agent is free to change

Within the scope files, the agent has full freedom to:

- **Rewrite the Kalman state representation** — e.g., switch from xyxy-corners to (cx, cy, scale, ratio) or (cx, cy, w, h). Update `optimize_tracking.py` accordingly.
- **Redesign the association pipeline** — additional similarity metrics, gating strategies, cascade matching, score-weighted matching.
- **Integrate camera motion compensation** using `trackers/motion/estimator.py`; the infrastructure already exists in the codebase.
- **Add new constructor parameters** when motivated by the algorithm change; update `_build_tracker` and `_define_search_space` in `optimize_tracking.py` to expose the new knobs to Optuna.
- **Update `trackers/__init__.py`** if the public interface changes as a result of architectural improvements.
- Implement any classical (non-learned) tracking technique that improves HOTA.

Each iteration must propose and implement **one atomic hypothesis**. Compound changes (two ideas in one commit) make it impossible to know what worked.

### Failure logging

Every reverted change is a result, not a failure. The `experiments.jsonl` log captures what was tried and what didn't improve HOTA. After the campaign, this log is the primary research artifact.

If the campaign reaches `max_iterations` without achieving the goal, this is a valid research outcome. The `experiments.jsonl` log and final code state are the deliverables.

### ByteTrack Phase 1 findings — already in the code (do not re-implement for ByteTrack)

These hypotheses were implemented and kept in ByteTrack Phase 1. They are **not yet present in SORT or OC-SORT** and are promising first hypotheses for those trackers:

| Hypothesis                                     | HOTA delta on ByteTrack |
| ---------------------------------------------- | ----------------------- |
| Velocity decay β=0.95 during lost frames       | +0.667%                 |
| Q inflation on missed frames (alpha=0.1)       | +0.717%                 |
| Post-processing gap interpolation (max_gap=20) | +1.666%                 |
| P reset on re-detection after occlusion        | +1.674%                 |

### ByteTrack Phase 1 — tried and reverted (likely to regress for any tracker)

| Hypothesis                                   | Outcome                                         |
| -------------------------------------------- | ----------------------------------------------- |
| Non-uniform P init (pos/vel split)           | Both regressions on ByteTrack                   |
| Size-adaptive R (area scaling)               | Regression                                      |
| NSA Kalman confidence-gated R                | Regression                                      |
| Two-stage IoU threshold (Stage 1 ≠ Stage 2)  | Both regressions on ByteTrack                   |
| Immature track grace period (1 missed frame) | Regression                                      |
| Joseph-form covariance update                | No change (algebraically equivalent at float32) |
| Two-hit birth policy                         | Regression                                      |
| Per-axis velocity decay (pos/size split)     | Regression                                      |
| Velocity-only Q inflation                    | No change                                       |

### Research starting points — SOTA-inspired, not yet tried on all trackers

Provided as inspiration, not a prescribed order. Hypotheses apply to the active `{algo}` tracker unless noted.

**H-A: xcycsr state representation** Switch from xyxy-corners state `[x1,y1,x2,y2,vx1,vy1,vx2,vy2]` to center-based `[cx,cy,s,r,vcx,vcy,vs]` where `s = w*h` (area) and `r = w/h` (aspect ratio, often frozen). Corner velocities can be noisy when detectors shift boxes independently; center+area+ratio is more stable. Requires rewriting `kalman.py`, state↔bbox converters, and updating `optimize_tracking.py`. Applies to SORT and ByteTrack (both currently use xyxy corners).

**H-B: Camera motion compensation (CMC)** Apply a frame-to-frame homography (ECC or sparse optical flow) to transform Kalman state predictions before association. The infrastructure exists at `trackers/motion/estimator.py` (`MotionEstimator`). BoT-SORT's primary gain on moving-camera sequences. Applies to SORT and ByteTrack. If full optical flow is too expensive for the 7 s eval budget, consider estimating motion from the detection cloud centroid shift.

**H-C: Mahalanobis gate** Add a Mahalanobis distance gate using the predicted `P` matrix to discard geometrically impossible matches before the IoU similarity matrix is computed. Gate threshold is a tunable parameter. Applies to all trackers — modifies `get_iou_matrix` in `trackers/core/sort/utils.py`.

**H-D: OC-SORT observation-centric velocity re-estimation** On re-detection after occlusion, compute a "virtual trajectory" between the last observation and the current detection (position difference / frames elapsed) and use this to update the velocity state, replacing the decayed estimate. Already implemented in ByteTrack; not in SORT. Pure position arithmetic — no appearance features needed.

**H-E: Track confidence score** Maintain a per-track `score` (float) that initialises to detection confidence at birth, updates as a weighted average, decays during lost frames, resets on re-match. Use score as a minimum-gate for culling lost tracks early. Already in ByteTrack (partially); not in SORT.

**H-F: GIoU or DIoU as association metric** Replace IoU with Generalised IoU (GIoU) or Distance IoU (DIoU) in the similarity matrix. Both recover near-miss associations that pure IoU scores as zero. Implementation: modify `get_iou_matrix` in `trackers/core/sort/utils.py` — affects all trackers that use it.

**H-G: Adaptive confirmation threshold by object size** Make `minimum_consecutive_frames` adaptive: small/distant objects require more frames before confirmation; large/nearby objects are confirmed faster. New parameter: `size_conf_scale`. Modifies `get_alive_trackers` in `trackers/core/sort/utils.py` — affects all trackers.

**H-H: Separate high/low IoU thresholds (parametrised correctly)** Expose `stage1_iou` and `stage2_iou` as separate Optuna parameters with the constraint `stage2_iou ≤ stage1_iou`, default `stage1_iou = 0.1` so baseline is not broken. Relevant to ByteTrack's two-stage association; SORT uses single-stage so this maps differently.

### Agent warning — Kalman patch and state representation

`_apply_kalman_patch` in `optimize_tracking.py` overwrites Q, R, and P with uniform identity-scaled matrices. If the state representation is changed (H-A), the patch must be redesigned to work with the new state dimension and matrix structure.
