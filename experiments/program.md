# Campaign: ByteTrack algorithmic improvement on MOT17

## Goal

**Research question**: Which algorithmic changes to the ByteTrack implementation actually
improve multi-object tracking quality — independent of parameter tuning?

The hypothesis is that the current implementation has correctness and design gaps in its
Kalman filter dynamics and association logic that limit HOTA regardless of how well
parameters are tuned. The agent's job is to find and fix those gaps, not to search the
parameter space.

Optuna is a **validation tool**, not the goal. Parameter tuning can mask bad algorithms
(a well-tuned bad model can beat a poorly-tuned good model) — so every candidate
improvement is evaluated at default params first, with optional post-change Optuna to
confirm the signal is real and not a parameter artefact.

## Metric

```
command: cd experiments && uv run python optimize_tracking.py --n-trials 1 2>&1 | grep "^__METRICS__" | grep -oE "HOTA=[0-9.]+" | cut -d= -f2
direction: higher
target: 60.0
```

## Guard

```
command: uv run pytest test/ -m "not integration" --ignore=test/scripts -q
```

## Config

```
max_iterations: 20
agent_strategy: ml
scope_files:
  - trackers/core/bytetrack/tracker.py
  - trackers/core/bytetrack/kalman.py
  - trackers/core/sort/utils.py
  - experiments/optimize_tracking.py
compute: local
```

## Notes

### Evaluation protocol

- **Primary metric**: HOTA on MOT17-val, FRCNN public detections. Stops at 60.0 or
    `max_iterations`, whichever comes first.
- **Secondary metrics** (logged, not gated): IDF1, MOTA, IDSW. A change that improves
    HOTA but worsens IDSW significantly is a warning sign — log it.
- **Baseline**: HOTA = 50.355 at default parameters (no CLI flags).
- **Target**: 60.0 — see `README.md → Target analysis` for the full derivation.
    - Optuna alone on current code: ceiling ≈ 53–55
    - Best published IoU-only trackers on MOT17 val / FRCNN: ≈ 56–58
    - Theoretical IoU-only ceiling (FRCNN): ≈ 60–65
    - 60.0 is at the theoretical frontier — requires real algorithmic improvements
- **Tuned reference**: MOT17 test with YOLOX detections → HOTA ≈ 60.5. Gap to FRCNN
    reflects detector quality; no tracker can compensate for missed detections.
- **Fast mode** (`--fast`): single sequence (~3 s), sanity check only; campaign metric
    (`--n-trials 1`) always runs the full eval (~7 s, all sequences).

### Hard boundaries — these invalidate the experiment if violated

1. **Do not bypass `trackers.eval`**. The evaluation calls in `optimize_tracking.py` must
    go through `trackers.eval` unchanged — do not substitute custom metric code.
2. **Do not modify `trackers/eval/`**. The metric computation must be identical across
    all iterations.
3. **No ground-truth at inference time**. The tracker sees only detector output
    (`det/det.txt` — FRCNN public detections). It must not read from `gt/` at any point.
4. **No external features**. The FRCNN detector provides bounding boxes and confidence
    scores only. No appearance embeddings, no depth, no optical flow at association time
    unless derived purely from the bounding box sequence itself.
5. **The Kalman filter must remain a proper linear Kalman filter**. Learned components
    (neural prediction, learned motion model) require a separate research question and are
    out of scope here.
6. **Do not change the public API** of `ByteTrackTracker` or `ByteTrackKalmanBoxTracker`.
    Constructor signatures and the `update()` method signature must remain unchanged.

### Optuna's role

Optuna is used in two places only:

1. **Pre-campaign baseline** (run once by the human before starting the loop):
    run `python optimize_tracking.py --n-trials 200`, save the best param config
    to `best_config.json`. This gives a tuned ceiling for the *current* code — any
    code change must beat this ceiling to be meaningful.

2. **Post-change validation** (optional, agent-initiated): after a code change is *kept*
    by the campaign loop, the agent may run a 50-trial mini-Optuna with the new code to
    confirm the improvement holds under tuned params and to update `best_config.json`.
    If tuned params *erase* the code change's improvement, that is an important negative
    result — log it and revert.

The campaign metric always measures at **default parameters** to keep the iteration loop
fast (~7 s per run). Optuna runs happen outside the metric verification step.

### What the agent is free to change

Within the scope files, the agent has full freedom to:

- Change the Kalman state representation, covariance initialization, and update equations
    in `kalman.py`
- Change the association logic, similarity metric, and track lifecycle in `tracker.py`
- Implement any classical (non-learned) tracking technique that improves HOTA
- Update `optimize_tracking.py` search space and tracker construction as architecture evolves

Each iteration must propose and implement **one atomic hypothesis**. Compound changes
(two ideas in one commit) make it impossible to know what worked.

### Failure logging

Every reverted change is a result, not a failure. The `experiments.jsonl` log captures
what was tried and what didn't improve HOTA. After the campaign, this log is the primary
research artifact — it answers "what does and doesn't matter for ByteTrack quality."

### Research starting points

Known gaps in the current implementation — provided as inspiration, not a
prescribed order. The agent is free to pursue any of these, combine them, find
something else entirely, or contradict them. The experiment log is the record of
what was actually tried.

- **Kalman P initialization**: `P = I(8)` is dimensionally inconsistent with pixel-scale state — position states live at 100–1000 px, velocity at 0–100 px/frame. Standard practice scales P_pos to detection uncertainty and P_vel to a large prior.
- **Size-adaptive R**: `R = 0.1·I(4)` is independent of box size. A 400×600 px pedestrian has far more pixel-level detection noise than a 50×50 px cyclist.
- **Two-threshold association**: `minimum_iou_threshold` is used identically for Stage 1 and Stage 2. The ByteTrack paper uses different thresholds — Stage 2 (occluded tracks) benefits from a looser match.
- **Immature track grace period**: any missed frame on an immature track kills it immediately. A short grace period stabilises track birth on intermittently detected objects.
- **Joseph-form covariance update**: `P = (I−KH)P` can go non-PSD on long tracks; the Joseph form `(I−KH)P(I−KH)ᵀ + KRKᵀ` is numerically stable.
- **Q inflation on missed frames**: constant Q regardless of `time_since_update` means a 10-frame occluded track has the same prediction confidence as a 1-frame one.
- **Velocity attenuation during lost frames**: constant-velocity prediction extrapolates linearly during occlusion — after 10+ frames the predicted box is far from the true position. Multiplying velocity states by a decay factor `β < 1` per missed frame prevents unbounded drift and is the primary mechanism behind OC-SORT/BoT-SORT AssA gains.

> **Agent warning — Kalman patch conflict**: `_apply_kalman_patch` in `optimize_tracking.py`
> overwrites Q, R, and P with uniform identity-scaled matrices. If H2 (size-adaptive R) or
> H1 (non-uniform P) is implemented in `kalman.py`, the patch will silently revert those
> matrices during Optuna runs. After implementing H1 or H2, update `_apply_kalman_patch` to
> preserve the new matrix structure (e.g., apply scale as a multiplier on the new non-uniform
> baseline, or remove the conflicting `r_scale` / `p_scale` Optuna params and replace them
> with the structural parameters introduced by the hypothesis).

> **H4 scope note**: `get_alive_trackers` lives in `trackers/core/sort/utils.py` (shared with
> SORT tracker). Modifying it will change SORT behaviour too — prefer implementing the grace
> period inside `tracker.py` to keep the change isolated to ByteTrack.

### Current best config (baseline, default params)

```json
{
  "hota": 50.355,
  "config": {
    "lost_track_buffer": 30,
    "track_activation_threshold": 0.7,
    "minimum_consecutive_frames": 2,
    "minimum_iou_threshold": 0.1,
    "high_conf_det_threshold": 0.6,
    "q_scale": 0.01,
    "r_scale": 0.1,
    "p_scale": 1.0
  }
}
```
