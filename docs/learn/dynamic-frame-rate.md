# Dynamic Frame Rate

Trackers normally assume **one `update()` call per video frame**. When frames are skipped — variable FPS, async inference, network drops, or batch gaps — fixed-rate prediction treats every processed step as a single frame of motion. Passing a **`timestamp`** in seconds tells the tracker how much time actually passed so Kalman prediction and lost-track pruning match the real gap.

**What you'll learn:**

- When to enable dynamic frame rate
- How fixed-rate and timestamp modes differ
- What `frame_step` and `elapsed_seconds` mean
- Edge cases (bootstrap, non-monotonic timestamps, mixed calls)
- OC-SORT-specific behaviour

For a minimal code example, see [Track Objects — Variable frame rate](track.md#variable-frame-rate).

---

## When to use it

Use **`timestamp=`** when the wall-clock gap between two processed updates can differ from one frame period. Typical cases:

- Frame drops on a live stream or conveyor camera
- Variable-FPS files where decode timestamps reflect capture time
- Async detectors with **irregular gaps**, as long as timestamps stay **monotonic** (capture time or PTS, not processing order)

Keep **`timestamp=None`** (omit the argument) when you process every frame in order at a steady rate. That preserves existing behaviour and benchmark numbers.

All four trackers support both modes: `SORTTracker`, `ByteTrackTracker`, `OCSORTTracker`, and `BoTSORTTracker`.

---

## Fixed rate vs dynamic rate

|                     | Fixed rate (default)                                 | Dynamic rate                                 |
| ------------------- | ---------------------------------------------------- | -------------------------------------------- |
| `timestamp`         | `None` (omit)                                        | Monotonic seconds, e.g. video clock          |
| Kalman `frame_step` | `1.0` per call (frame units)                         | `elapsed_seconds × frame_rate` (frame units) |
| Lost-track budget   | Frames (`lost_track_buffer`, scaled by `frame_rate`) | Seconds (`lost_track_buffer / 30`)           |

`frame_rate` is required in **both** modes. In fixed mode it scales frame-based thresholds. In dynamic mode it is the **reference FPS** used to bootstrap the first timestamped step and to convert gaps into frame units for the Kalman filter.

At a constant 25 FPS with no drops, `frame_step` stays `1.0` — dynamic mode matches fixed mode. Dynamic mode is meant for **variable** gaps between updates.

---

## Two time quantities

Kalman prediction and lost-track pruning intentionally use different units.

### `frame_step` (Kalman predict)

Used to scale the motion matrix `F` and process noise `Q`. Values are in **frame units**, not seconds:

- Fixed mode: always `1.0` per `update()`.
- Dynamic mode: `frame_step = elapsed_seconds × frame_rate`.

Velocity in the filter state is displacement **per frame period** (e.g. pixels per frame at the reference rate). Existing SORT / ByteTrack tuning assumes this convention.

### `elapsed_seconds` (lost-track pruning)

When `timestamp` is supplied, each tracklet accumulates **`time_since_update_seconds`** on predict and resets it on match. Tracks are pruned against a seconds budget derived from `lost_track_buffer`.

When `timestamp` is omitted, pruning uses the existing **frame-count** logic (`time_since_update`).

---

## How prediction scales

Implementation path:

1. `BaseTracker._predict_timing(timestamp)` builds a `PredictTiming` object.
2. Tracklets pass `timing.frame_step` into the state estimator's `predict()`.
3. **`KalmanMotionModel`** writes `F` and `Q` on the filter before each predict step.

**Motion (`F`).** Constant velocity: each position row picks up `velocity × frame_step` (`constant_velocity_F`).

**Process noise (`Q`).** Each tracklet sets a one-frame `Q` in `_configure_noise()`. At `frame_step = 1.0` that matrix is used unchanged. For other steps, kinematic blocks are rebuilt with **discrete white-noise acceleration (DWNA)** scaling — position variance ∝ `Δt⁴`, velocity ∝ `Δt²` — so uncertainty grows correctly on longer gaps. Multiplying the one-frame `Q` by `Δt` alone would be wrong in either direction.

Implementation: `trackers.utils.motion_models.KalmanMotionModel` and `ScalableProcessNoise.build_Q()`.

---

## Usage

=== "Python"

    ```python
    import cv2

    import supervision as sv
    from inference import get_model
    from trackers import ByteTrackTracker

    model = get_model("rfdetr-nano")
    tracker = ByteTrackTracker(frame_rate=30.0, lost_track_buffer=30)

    cap = cv2.VideoCapture("source.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections, timestamp=timestamp)
    ```

Use **capture time** (frame index / FPS, camera PTS, container timestamp). Avoid `time.time()` at receive/decode unless that is your intended timeline.

---

## Behaviour notes

- **Backward compatible:** omitting `timestamp` reproduces fixed-rate behaviour.
- **First timestamped call:** elapsed time bootstraps to `1 / frame_rate` so the first Kalman step uses `frame_step = 1.0`, not the absolute clock value.
- **Non-monotonic timestamps:** predict is skipped for that step; a warning is emitted.
- **Per-call mode:** only calls that pass `timestamp` use seconds-based pruning and scaled predict. A later call without `timestamp` reverts to `frame_step = 1.0` and frame-count pruning for that step.
- **Between videos:** call `tracker.reset()` so timestamp state does not carry over.

---

## OC-SORT caveat

OC-SORT's Observation-Centric Re-Update (ORU) virtual trajectory inside `_unfreeze_*` still advances the Kalman filter in **unit-frame** steps. Main-track predict and lost-track pruning respect `timestamp`; ORU gap length follows frame counts, not wall-clock seconds.

For timestamp mode with large gaps, prefer **`XYXYStateEstimator`** for OC-SORT. The default XCYCSR representation can produce invalid boxes after aggressive multi-frame predicts (`sqrt` of negative scale).

---

## Related

- [Track Objects](track.md) — CLI and Python tracking basics
- [State Estimators](state-estimators.md) — XYXY vs XCYCSR representations
- [Tune Trackers](tune.md) — hyperparameters such as `lost_track_buffer` and `frame_rate`
