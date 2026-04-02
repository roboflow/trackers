# Campaign: ByteTrack algorithmic improvement on MOT17 — Phase 2

## Goal

**Research question**: Which architectural and algorithmic changes to the ByteTrack implementation improve multi-object tracking quality, now that all classical Kalman tuning ideas have been exhausted?

Phase 1 (iterations 1–20) explored Kalman parameter tuning, threshold tweaks, and post-processing. The kept improvements are baked into the current code (HOTA = 51.198). Phase 2 opens the scope to deeper architectural changes: state representation redesign, SOTA-inspired association strategies, camera motion compensation, and well-justified API evolution. The agent may rewrite components and change public signatures when the improvement is clearly motivated and the code change is documented.

Optuna is a **validation tool**, not the goal. Every candidate improvement is evaluated at default params first. Optuna confirms the signal is real and not a parameter artefact.

## Metric

```
command: cd autotrack && uv run python optimize_tracking.py bytetrack sdp --n-trials 1 2>&1 | grep "^__METRICS__" | grep -oE "HOTA=[0-9.]+" | cut -d= -f2
direction: higher
target: 68.0
```

## Guard

```
command: uv run pytest test/ -m "not integration" --ignore=test/scripts -q
```

## Config

```
algo: bytetrack
max_iterations: 20
agent_strategy: ml
det_source: sdp
scope_files:
  - trackers/**
  - autotrack/optimize_tracking.py
  - autotrack/generate_detections.py
  - autotrack/search_space.json
  - autotrack/default_config.json
out_of_scope_files:
  - trackers/eval/**
  - trackers/datasets/**
compute: local
```

## Notes

### Pre-flight checks

All three setup steps must pass before starting the campaign loop:

| Check         | Command                                                                     | Expected result                                        |
| ------------- | --------------------------------------------------------------------------- | ------------------------------------------------------ |
| Dependencies  | `uv sync --group optimize`                                                  | Resolves without error                                 |
| MOT17 data    | `trackers download mot17 --split val --asset annotations,detections,frames` | Downloads to `~/.cache/trackers/mot17/val/`            |
| Metric sanity | `cd autotrack && uv run python optimize_tracking.py bytetrack sdp --fast`   | Prints `__METRICS__: HOTA≈60–65` (SDP-val, single seq) |

> **Custom detector** — create `MOT17-{N}-MYDET/det/det.txt` sibling dirs (see README.md Custom detections section), just pass the name as `det_source`: `uv run python optimize_tracking.py bytetrack mydet` — unknown names are uppercased to form the directory tag (`MYDET`).

### Evaluation protocol

- **Primary metric**: HOTA on MOT17-val, SDP detections. Stops at target or `max_iterations`, whichever comes first.
- **Secondary metrics** (logged, not gated): IDF1, MOTA, IDSW. A change that improves HOTA but worsens IDSW significantly is a warning sign — log it.
- **Phase 1 baseline (SDP)**: HOTA = 53.941 at default parameters (campaign start).
- **Phase 2 baseline (SDP)**: HOTA = 53.941 (current code, default params — Phase 1 improvements are FRCNN-measured; SDP re-baseline TBD after first Phase 2 iteration).
- **Fast mode** (`--fast`): single sequence (~3 s), sanity check only; campaign metric (`--n-trials 1`) always runs the full eval (~7 s, all sequences).

### Hard boundaries — these invalidate the experiment if violated

1. **Do not bypass `trackers.eval`**. The evaluation calls in `optimize_tracking.py` must go through `trackers.eval` unchanged — do not substitute custom metric code.
2. **Do not modify `trackers/eval/`**. The metric computation must be identical across all iterations.
3. **No ground-truth at inference time**. The tracker sees only detector output (`det/det.txt`). It must not read from `gt/` at any point.
4. **No external features**. The detector provides bounding boxes and confidence scores only. No appearance embeddings, no depth, no optical flow at association time unless derived purely from the bounding box sequence itself.
5. **The Kalman filter must remain a proper linear Kalman filter**. Learned components (neural prediction, learned motion model) require a separate research question and are out of scope here.
6. **API changes are allowed when well-justified**. Constructor signatures and `update()` may change if the change is architecturally motivated, accompanied by a rationale comment, and `optimize_tracking.py` is updated to match. The tracker must still implement `BaseTracker` and accept `sv.Detections` input. Do not rename the public classes (`ByteTrackTracker`, `ByteTrackKalmanBoxTracker`).

### Optuna's role

Optuna is used in two places only:

1. **Pre-campaign baseline** (run once by the human before starting the loop): run `python optimize_tracking.py bytetrack sdp --n-trials 200`, save the best param config to `best_config.json`. This gives a tuned ceiling for the *current* code.

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
- **Add new constructor parameters** to `ByteTrackTracker` or `ByteTrackKalmanBoxTracker` when motivated by the algorithm change; update `_build_tracker` and `_define_search_space` in `optimize_tracking.py` to expose the new knobs to Optuna.
- **Update `trackers/__init__.py`** if the public interface changes as a result of architectural improvements (e.g., new parameters added to the exported class).
- Implement any classical (non-learned) tracking technique that improves HOTA.

Each iteration must propose and implement **one atomic hypothesis**. Compound changes (two ideas in one commit) make it impossible to know what worked.

### Failure logging

Every reverted change is a result, not a failure. The `experiments.jsonl` log captures what was tried and what didn't improve HOTA. After the campaign, this log is the primary research artifact.

### Phase 1 findings — what is already in the code

These hypotheses were implemented and kept in Phase 1. Do not re-implement them:

| Hypothesis                                     | Commit | HOTA delta |
| ---------------------------------------------- | ------ | ---------- |
| Velocity decay β=0.95 during lost frames       | i2     | +0.667%    |
| Q inflation on missed frames (alpha=0.1)       | i3     | +0.717%    |
| Post-processing gap interpolation (max_gap=20) | i9B    | +1.666%    |
| P reset on re-detection after occlusion        | i11    | +1.674%    |

### Phase 1 findings — tried and reverted (do not retry unless rationale changes)

These were tried in Phase 1 and caused regressions at default params. A retry is only worth attempting if the implementation approach changes fundamentally:

| Hypothesis                                   | Iterations | Outcome                                         |
| -------------------------------------------- | ---------- | ----------------------------------------------- |
| Non-uniform P init (pos/vel split)           | i1, i19    | Both regressions                                |
| Size-adaptive R (area scaling)               | i7         | Regression                                      |
| NSA Kalman confidence-gated R                | i18        | Regression                                      |
| Two-stage IoU threshold (Stage 1 ≠ Stage 2)  | i4, i10    | Both regressions                                |
| Immature track grace period (1 missed frame) | i5         | Regression                                      |
| Joseph-form covariance update                | i6         | No change (algebraically equivalent at float32) |
| Size-freeze during occlusion                 | i8         | Regression                                      |
| Two-hit birth policy                         | i12        | Regression                                      |
| Per-axis velocity decay (pos/size split)     | i13        | Regression                                      |
| Confidence-weighted IoU in Stage 1           | i14        | Regression                                      |
| Cascaded age-priority matching               | i16        | Regression                                      |
| Velocity-only Q inflation                    | i20        | No change                                       |

### Research starting points — Phase 2 (SOTA-inspired, not yet tried)

Provided as inspiration, not a prescribed order. The agent is free to pursue any of these, combine them, find something else entirely, or contradict them. The experiment log is the record of what was actually tried.

**H-A: xcycsr state representation** Switch from xyxy-corners state `[x1,y1,x2,y2,vx1,vy1,vx2,vy2]` to center-based `[cx,cy,s,r,vcx,vcy,vs]` where `s = w*h` (area) and `r = w/h` (aspect ratio, often frozen). The original SORT/ByteTrack paper uses this representation. Corner velocities can be noisy when detectors shift boxes independently; center+area+ratio is more stable. Requires rewriting `kalman.py`, state↔bbox converters, and updating `optimize_tracking.py`.

**H-B: Camera motion compensation (CMC)** Apply a frame-to-frame homography (ECC or sparse optical flow) to transform Kalman state predictions before association, so that static background objects are correctly compensated. The infrastructure exists at `trackers/motion/estimator.py` (`MotionEstimator`). BoT-SORT's primary gain on moving-camera sequences comes from this. Requires integrating the estimator into `ByteTrackTracker.update()` and updating `optimize_tracking.py` to accept image frames (the det files contain frame indices; images would need to be loaded from the sequence folders). If loading frames is too expensive for the 7 s eval budget, consider a lightweight version that estimates motion from the detection cloud itself (centroid shift) rather than optical flow.

**H-C: Mahalanobis gate** Add a Mahalanobis distance gate using the predicted `P` matrix to discard geometrically impossible matches before the IoU similarity matrix is computed. Used by DeepSORT and BoT-SORT to prune false positives in the assignment step. Gate threshold is a tunable parameter. Can be combined with IoU: `similarity = IoU * gate_mask` where `gate_mask[i,j] = (mahal_dist[i,j] < chi2_threshold)`.

**H-D: OC-SORT observation-centric velocity re-estimation** On re-detection after occlusion, compute a "virtual trajectory" between the last observation and the current detection (position difference / frames elapsed) and use this to update the velocity state, replacing the decayed estimate. OC-SORT's primary AssA gain. Pure position arithmetic — no appearance features needed.

**H-E: Track confidence score** Maintain a per-track `score` (float) that:

- Initialises to the detection confidence at birth
- Updates to a weighted average with each matched detection
- Decays multiplicatively each lost frame
- Resets toward the fresh detection confidence on re-match

Use score as a minimum-gate for keeping lost tracks in the active pool (tracks below `min_track_score` are culled early). ByteTrackV2 / StrongSORT pattern.

**H-F: GIoU or DIoU as association metric** Replace IoU with Generalised IoU (GIoU) or Distance IoU (DIoU) in the similarity matrix. GIoU penalises non-overlapping boxes more informatively (adds a penalty for the smallest enclosing box). DIoU adds a distance penalty between box centres. Both can recover near-miss associations that pure IoU scores as zero. Implementation: modify `get_iou_matrix` in `trackers/core/sort/utils.py` or add a parallel function.

**H-G: Adaptive confirmation threshold by object size** Make `minimum_consecutive_frames` adaptive: small/distant objects (small detection area) require more consecutive frames before confirmation; large/nearby objects are confirmed faster. Reduces false tracks from small noisy detections while keeping large objects confirmed quickly. New parameter: `size_conf_scale` — scale the confirmation requirement by inverse sqrt of normalised area.

**H-H: Separate high/low IoU thresholds (parametrised correctly)** The i4 attempt failed because the default Stage 1 threshold was set to 0.5, which was too aggressive. A fresh attempt should expose `stage1_iou` and `stage2_iou` as separate Optuna parameters with the constraint `stage2_iou ≤ stage1_iou`, and ensure the default `stage1_iou = 0.1` (matching current behavior) so the baseline is not broken. This is architecturally the right design; the prior attempt just used a bad default.

### Agent warning — Kalman patch and state representation

`_apply_kalman_patch` in `optimize_tracking.py` overwrites Q, R, and P with uniform identity-scaled matrices. If the state representation is changed (H-A), the patch must be redesigned to work with the new state dimension and matrix structure. After implementing H-A, replace `_apply_kalman_patch` with representation-aware parameter injection, or integrate the noise scales directly into the constructor.

### Current best config (Phase 2 start — SDP detections)

```json
{
  "hota": 53.941,
  "config": {
    "lost_track_buffer": 30,
    "track_activation_threshold": 0.7,
    "minimum_consecutive_frames": 2,
    "minimum_iou_threshold": 0.1,
    "high_conf_det_threshold": 0.6,
    "q_scale": 0.01,
    "r_scale": 0.1,
    "p_scale": 1.0,
    "velocity_decay": 0.95,
    "q_miss_alpha": 0.1,
    "max_interpolation_gap": 20,
    "p_reset_threshold": 5
  }
}
```
