# Dynamic Frame‑Rate Support for `trackers` — Feature Spec

> **Status:** Proposal — design document, not yet implemented.
> **Owner:** `@alexanderbodner`
> **Last updated:** 2026‑05‑26 (rev 4 — MVP scoping: SORT/ByteTrack only for utility test; OC‑SORT/BoT‑SORT deferred behind a decision gate)
> **Scope:** affects `SORTTracker`, `ByteTrackTracker`, `OCSORTTracker`, `BoTSORTTracker`,
> the shared `KalmanFilter`, all `BaseStateEstimator` subclasses, the CLI, and the eval/tune harnesses.

---

## 1. Motivation

All four trackers in `src/trackers/core/*` currently assume that **one `update()` call corresponds to a fixed time step**. Two hidden assumptions follow from that:

1. **Kalman `predict()` uses `dt = 1` frame.** The state‑transition matrix `F` is hardcoded with ones in the velocity columns (see `src/trackers/utils/state_representations.py`).
2. **Every temporal threshold is in *frames*, not seconds.** `lost_track_buffer`, `minimum_consecutive_frames`, OC‑SORT's `delta_t`, BoT‑SORT's `time_since_update > 1` lost‑gate, and OC‑SORT's ORU virtual trajectory length are all frame counts. `frame_rate` is only used once at init, to rescale `lost_track_buffer` against a reference 30 FPS.

In practice many users feed the tracker frames at a non‑uniform rate:

- a detector that runs every Nth frame to save compute,
- an asynchronous detector callback,
- RTSP streams with dropped frames,
- replays at higher FPS than the original capture,
- sports / surveillance datasets ranging from 14 FPS (MOT17‑13) to 60+ FPS.

Under all of these, motion predictions drift by the same factor as the timing error, and the lifetime thresholds become silently miscalibrated. State‑of‑the‑art trackers' HOTA/MOTA drops dramatically under varying frame rates ([Liu et al., 2022](https://arxiv.org/abs/2209.11404)).

## 2. Goals & non‑goals

### Goals

- Allow the user to feed an **initial reference frame rate** at construction time (already supported by the API surface — we will redefine what it means).
- Allow the user to optionally feed a **per‑update timestamp** in seconds (new). When provided, the Kalman filter's `predict` step uses the actual elapsed time, and lifetime thresholds become time‑denominated.
- Preserve **byte‑for‑byte backward compatibility** when timestamps are not supplied. Existing benchmarks must produce identical tracks for the constant‑FPS path.
- **First prove utility on the simplest trackers** (`SORT`, `ByteTrack`) before doing the more invasive refactors needed for `OC‑SORT` and `BoT‑SORT`. The phased rollout in §6 has an explicit go/no‑go gate after the first benchmark sweep.

### Stretch goals (only if the utility gate in §6 is passed)

- Apply the same refactor to `OC‑SORT` (delta_t in seconds, observations keyed by timestamp, ORU sub‑step rule, per‑second velocity).
- Apply the same refactor to `BoT‑SORT` (`tsu > 1` lost‑gate becomes `tsu > one_frame_period`).

### Non‑goals

- Estimating frame rate from video metadata automatically inside the tracker (caller's responsibility).
- Learned frame‑rate‑aware association modules (FAAM/FAPS style); we restrict ourselves to a motion‑model refactor.
- Adding appearance / Re‑ID features as a low‑FPS mitigation (orthogonal feature).
- Per‑object dt (different objects observed at different rates in the same update). Detections in a single `update()` call share one timestamp.

## 3. Public API

### 3.1 Construction (unchanged signatures, redefined semantics)

```python
tracker = ByteTrackTracker(
    frame_rate=30.0,                      # REQUIRED. Reference FPS used as prior;
                                          # must be > 0 even in dynamic mode.
    lost_track_buffer=30,                 # interpreted as buffer in frames AT 30 FPS reference
    maximum_time_without_update=None,     # NEW: override above in seconds
    process_noise_scale=1.0,              # NEW: scalar to scale σ_a² globally
    # ... existing params unchanged
)
```

`frame_rate` is **required** in both modes (Fixed and Dynamic). In Fixed mode it sets the per‑step `dt`. In Dynamic mode it is the *prior reference* — used to calibrate `σ_a²` for the `Q(Δt)` formula (§4.3), to bootstrap `dt` on the first call before a previous timestamp exists, and to convert frame‑denominated parameters (`lost_track_buffer`, OC‑SORT `delta_t`, BoT‑SORT lost‑gate) into seconds.

### 3.2 Update (new optional kwarg)

```python
detections = tracker.update(
    detections,
    frame=frame_image,        # already exists, unchanged (BoT-SORT CMC)
    timestamp=t_seconds,      # NEW: monotonic float seconds; default None
)
```

Two operating modes — mirroring the FraMOT [known FPS / unknown FPS] split:

| Mode | `frame_rate` | `timestamp` per update | Behavior |
|------|--------------|------------------------|----------|
| **Fixed** (default, backward‑compatible) | e.g. `30.0` | `None` | `dt = 1/frame_rate` every step → identical to today |
| **Dynamic** | reference for Q calibration | `float` seconds | `dt = t − t_prev`. We trust the caller's timestamps and apply no upper clamp (see §4.6). Internal guard: `dt ≤ 0` (duplicate / non‑monotonic timestamp) is treated as a no‑op predict and emits a one‑time warning. |

On the first call with a timestamp, `t_prev` is initialised to that timestamp and `dt = 1/frame_rate` is used (so the first step does not depend on a non‑existent previous timestamp). Subsequent `None` timestamps after a non‑`None` one emit a one‑time warning and fall back to `dt = 1/frame_rate`.

## 4. Theoretical background

### 4.1 Time‑parameterized state transition `F(Δt)`

The constant‑velocity (CV) model for a 1D coordinate `(p, ṗ)` is:

```
p_{k+1} = p_k + ṗ_k · Δt
ṗ_{k+1} = ṗ_k
```

i.e. `F(Δt) = [[1, Δt], [0, 1]]`. Today we use `Δt = 1` everywhere. For arbitrary `Δt` we apply this per coordinate; for our 8‑dim XYXY state the velocity columns become `F[i, i+4] = Δt` for `i ∈ {0..3}`, and similarly for XCYCWH (BoT‑SORT) and XCYCSR (OC‑SORT, with no velocity on the aspect ratio row).

### 4.2 Time‑parameterized process noise `Q(Δt)` — the heart of the refactor

`Q` is the covariance of unmodelled state evolution **integrated over one step**. The integral depends on `Δt`, so a constant `Q` is *only* correct at the one `Δt` for which it was tuned.

There are two canonical discretizations ([Bar‑Shalom et al., 2001, Ch. 6](https://www.wiley.com/en-us/Estimation+with+Applications+to+Tracking+and+Navigation%3A+Theory+Algorithms+and+Software-p-9780471416555)):

**(A) Discrete White Noise Acceleration (DWNA)** — acceleration is a piecewise‑constant random variable per step, redrawn independently each step. For a 1D CV model with state `(p, ṗ)`:

```
                  ⎡  Δt⁴/4    Δt³/2 ⎤
Q_DWNA(Δt) = σ_a² ⎢                 ⎥
                  ⎣  Δt³/2     Δt²  ⎦
```

This is `filterpy.common.Q_discrete_white_noise` ([FilterPy docs](https://filterpy.readthedocs.io/en/latest/common/common.html)).

**(B) Continuous White Noise Acceleration (CWNA)** — `a(t)` is continuous‑time white noise with spectral density `q`:

```
                ⎡  Δt³/3    Δt²/2 ⎤
Q_CWNA(Δt) = q  ⎢                 ⎥
                ⎣  Δt²/2     Δt   ⎦
```

This is `filterpy.common.Q_continuous_white_noise`.

The two are physically distinct stories but numerically interchangeable when `σ_a²` / `q` are tuned per setup. We adopt **DWNA** because (i) it is the more common choice in the MOT lineage (Deep‑SORT and successors), (ii) it cleanly matches the "one Kalman step per detection frame, fresh acceleration draw each frame" physical picture, and (iii) it is what FilterPy users encounter first.

**Why a constant `Q` fails when `Δt` varies.** Position uncertainty in DWNA scales as `Δt⁴`. Numerically:

| Δt (s) | Position term `σ_a² · Δt⁴/4` (relative to Δt=1) |
|--------|--------------------------------------------------|
| 1/60   | 7.7e‑6 |
| 1/30   | 1.2e‑4 |
| 1/10   | 1e‑2   |
| 1/2    | 0.0625 |
| 1      | 1      |
| 2      | 16     |

Keeping `Q` constant across this range is physically incorrect. The OC‑SORT paper documents the failure mode analytically: *"the scale of the noise of direction estimation is negatively correlated to the time difference between the two observation points, i.e. Δt … the choice of Δt requires a trade‑off"* ([Cao et al., 2023, §4](https://arxiv.org/abs/2203.14360)). The empirical mirror is in APPTracker, where lost‑track buffer must shrink with frame‑skip ratio to keep tracking accuracy up ([Zhou et al., 2022, Table 5](https://infzhou.github.io/folder/Zhou_APPTracker_Improving_Tracking_Multiple_Objects_in_Low-Frame-Rate_Videos_MM_2022.pdf)).

### 4.3 Back‑calibrating against today's `Q`

Today's `Q` is set as a constant diagonal (e.g. `Q = np.eye(8) * 0.01` in `src/trackers/core/sort/tracklet.py:75`). To preserve current behavior **at the reference `Δt = 1/frame_rate`**, we pick `σ_a²` per coordinate so that the *velocity* diagonal element of `Q_DWNA(1/frame_rate)` matches today's value:

```
σ_a²[i] = Q_today[i+4, i+4] / (Δt_ref)²       where Δt_ref = 1/frame_rate
```

For the default `frame_rate = 30.0` and today's `Q = 0.01 · I`, this gives `σ_a²[i] = 0.01 / (1/30)² = 9.0` (px/s²)² — which is exactly the physical interpretation we want documented.

The position diagonal `Q_DWNA(Δt_ref)[i,i]` will be `σ_a² · Δt_ref⁴/4` rather than today's `Q_today[i,i]`. This is a small numerical change at the reference rate; we absorb it into a single tunable scalar `process_noise_scale` (default `1.0`) and provide it as a knob for users who want byte‑for‑byte parity.

The off‑diagonal coupling terms `Q[i, i+4] = σ_a² · Δt³/2` are *new* — today's diagonal `Q` ignores position‑velocity correlation, which is technically incorrect even at `Δt = 1`. Adding them is a small but real improvement.

### 4.4 Time‑denominated lifetime thresholds

We switch `age` and `time_since_update` to **seconds** (`float`). Each frame‑count threshold gets a `*_seconds` counterpart, derived from existing parameters so users can ignore the change:

| Today (frames) | New canonical (seconds) | Conversion |
|----------------|-------------------------|------------|
| `maximum_frames_without_update` | `maximum_time_without_update` | `lost_track_buffer / 30.0` |
| `minimum_consecutive_frames` | `minimum_consecutive_observations` | unchanged — this is a *count of detections*, not time |
| BoT‑SORT lost‑gate `tsu > 1` | `tsu > one_frame_period` | `one_frame_period = 1.0 / frame_rate` |
| OC‑SORT `delta_t = 3` | `delta_t_seconds = 3 / 30.0 = 0.1 s` | rescaled at init |

This matches the APPTracker empirical finding that buffer‑in‑seconds is roughly invariant across frame skips.

### 4.5 OC‑SORT specifics

OC‑SORT touches the time axis in three places that need refactoring:

1. **OCM `delta_t` lookback** — currently keyed by `age - delta_t` (an integer frame count) into `self.observations`. Refactor: store observations as `(timestamp, bbox)` pairs and look up the observation closest to `t_now − delta_t_seconds`, subject to a minimum separation of `1/frame_rate` between the two observations used for velocity (one reference‑frame period; prevents amplifying noise when objects are observed back‑to‑back, exactly the σ² ∝ 1/Δt² warning from the OC‑SORT paper).
2. **ORU virtual trajectory** — currently runs `time_gap` predict/update sub‑steps. Refactor: with a real time gap `Δt_gap`, run `n = max(1, round(Δt_gap · frame_rate))` sub‑steps, each with `dt = Δt_gap / n`. This preserves the paper's original semantics of "one KF step per frame of expected motion" while staying correct under sparse sampling.
3. **Velocity for OCM direction cost** — currently `velocity = (bbox_new − bbox_prev) / 1`. Refactor: divide by the actual elapsed time between the two observations. The direction cost is scale‑invariant so the OCM angle test code stays unchanged; only the velocity construction needs the division.

### 4.6 Safety policy on `dt`

**Assumption.** The caller supplies timestamps from a single monotonic clock and the inter‑update gap is bounded by the application's natural cadence (camera frame rate, detector schedule). We do not defend against pathological gaps (e.g. minutes between updates, or unit errors where timestamps arrive in milliseconds instead of seconds).

This is a deliberate choice. The only public knob would have been a per‑step upper clamp, and we found three reasons to drop it: (1) any value we picked would be arbitrary across deployment scenarios; (2) it doubles the surface area users must reason about (`dt_max` vs. `maximum_time_without_update`); (3) the natural way the system reacts to a huge `Δt` — `Q` grows, gating widens, the track stays unmatched and gets pruned by `maximum_time_without_update` on the next step — is already correct behaviour for "a long time passed and we lost the object".

The one defensive guard we *do* keep is implementation‑level and has no public parameter:

| Case | Policy |
|------|--------|
| `dt ≤ 0` (duplicate timestamp, non‑monotonic input, clock wraparound) | Treat the call as a **no‑op predict** (skip `F`/`Q` application; do not advance `time_since_update`); association proceeds against the unchanged prior. Emit a `UserWarning` the first time it occurs per tracker instance. This avoids div‑by‑zero in OC‑SORT velocity and negative diagonals in `Q`. |

Rationale for the no‑op rather than a tunable threshold: `dt ≤ 0` is always an upstream bug, never a regime the user wants to tune. Skipping the predict step is the correct response regardless of any threshold value, so exposing it would only invite misconfiguration.

## 5. Concrete code touchpoints

### 5.1 Files changed

The table is split by PR so reviewers can see the minimal‑viable scope at a glance. "MVP" = lands in PRs 1–3; "deferred" = only lands if the §6 decision gate passes.

| Concern | File(s) | PR | Change |
|---------|---------|----|--------|
| KF F/Q builders | `src/trackers/utils/kalman_filter.py` | **PR 1 (MVP)** | extend `predict(dt)`; cache `(F, Q)` by `dt` |
| State estimators | `src/trackers/utils/state_representations.py` | **PR 1 (MVP)** | add `build_F(dt)`, `build_Q(dt)` for **all three** (XYXY, XCYCSR, XCYCWH); back‑calibrate today's σ² magnitudes |
| Tracklet base | `src/trackers/utils/base_tracklet.py` | **PR 1 (MVP, dt plumbing)** + **PR 2 (MVP, seconds counter)** | `predict(dt)` in PR 1; new `time_since_update_seconds` parallel field in PR 2 (existing integer `time_since_update` kept for OC‑SORT/BoT‑SORT) |
| Base tracker | `src/trackers/core/base.py` | **PR 2 (MVP)** | `update(..., timestamp=None)`; `_compute_dt`; `dt ≤ 0` no‑op + warn |
| SORT | `src/trackers/core/sort/{tracker,utils}.py` | **PR 2 (MVP)** | thread `dt` into tracklet `predict`; switch pruning to `time_since_update_seconds`; new `maximum_time_without_update` kwarg |
| ByteTrack | `src/trackers/core/bytetrack/{tracker,utils}.py` | **PR 2 (MVP)** | same as SORT |
| Benchmark sweep | `benchmark/scripts/dynamic_fps_sweep.py` (new), existing harness | **PR 3 (MVP)** | frame‑skip sweep on MOT17 + SportsMOT; Static vs. Dynamic plots |
| OC‑SORT (warning only) | `src/trackers/core/ocsort/tracker.py` | **PR 2 (MVP)** | accept `timestamp` kwarg but emit one‑time warning and fall back to `dt = 1/frame_rate` |
| BoT‑SORT (warning only) | `src/trackers/core/botsort/tracker.py` | **PR 2 (MVP)** | same as OC‑SORT |
| BoT‑SORT (real wiring) | `src/trackers/core/botsort/{tracker,tracklet,utils}.py` | **PR 5 (deferred)** | `tsu_seconds > one_frame_period`; drop the PR 2 warning; combine size‑scaled σ² with `Δt`‑polynomial |
| OC‑SORT (real wiring) | `src/trackers/core/ocsort/{tracker,tracklet,utils}.py` | **PR 6 (deferred)** | observations keyed by timestamp; `delta_t_seconds`; ORU sub‑step rule; velocity per‑second |
| CLI | `src/trackers/scripts/track.py` | **PR 7 (deferred)** | derive `timestamp = cap.get(CAP_PROP_POS_MSEC)/1000` when `--tracker.dynamic_dt true` |
| Demo | `demo/app.py` | **PR 7 (deferred)** | wire `frame_rate` and timestamps from `source_info` |
| Tuner | `src/trackers/tune/*` | **PR 7 (deferred)** | optionally include `maximum_time_without_update` in the search space |
| Tests | `tests/utils/test_kalman_filter.py`, per‑tracker tests | **PR 1–2 (MVP)** | back‑compat: `predict(1.0)` matches old numbers; equivalent‑timing test; frame‑skip equivalence test |
| Docs (this file) | `docs/design/dynamic-frame-rate.md` | every PR | revised as decisions are made |
| Per‑tracker docs | `docs/trackers/*.md` | after PR 4 verdict | "Variable frame rate" section per tracker (only for trackers that actually support it) |

### 5.2 Backward compatibility

The refactor is strictly additive at the API surface:

- All existing `__init__` params unchanged; same defaults.
- `timestamp=None` reproduces today's behavior exactly. The proof: when `timestamp=None`, `dt = 1/frame_rate` constantly, and `Q` is calibrated so that `Q_DWNA(1/frame_rate)` reproduces today's velocity diagonals.
- In PR 2, OC‑SORT and BoT‑SORT accept the `timestamp` kwarg for API uniformity but **do not yet act on it** — they warn once and fall back to Fixed mode. This keeps their numerical behaviour untouched while the utility hypothesis is being tested on SORT/ByteTrack.
- The CLI dynamic‑mode flag (`--tracker.dynamic_dt`) is introduced in PR 7, not earlier. The benchmark sweep in PR 3 calls the tracker API directly.
- Tests pin existing tracking outputs to within strict numerical tolerance for the `dt = 1` path on every tracker, every PR.

## 6. Phased rollout

The rollout is split into a **minimum‑viable spike** (PRs 1–3) that proves or disproves the utility on the two simplest trackers, an explicit **go/no‑go decision gate** (PR 4), and **conditional follow‑ups** (PRs 5–7) that only land if the gate is passed. Anything OC‑SORT‑ and BoT‑SORT‑specific is **deferred** until then.

### Minimum‑viable spike (always lands)

#### PR 1 — Time‑parameterized Kalman filter foundations

**Goal:** make the KF capable of advancing state by an arbitrary `Δt`, while leaving every existing call site behaving byte‑for‑byte identically.

**Files touched:**
- `src/trackers/utils/kalman_filter.py` — add `predict(dt: float = 1.0)`; cache `(F, Q)` keyed by `dt` to avoid rebuilding every step.
- `src/trackers/utils/state_representations.py` — for **all three** estimators (XYXY, XCYCSR, XCYCWH), add `build_F(dt)` and `build_Q(dt)` using the DWNA formula (§4.2). Back‑calibrate `σ_a²` per estimator so that `build_Q(1/frame_rate)` matches today's `Q` on the velocity diagonal (§4.3, §8). (Doing all three at once is essentially free — same template applied three times — and keeps later PRs from re‑touching this file.)
- `src/trackers/utils/base_tracklet.py` — `predict(dt: float = 1.0)` forwards `dt` to the estimator.

**Public API change:** none. Default `dt=1.0` everywhere → every existing caller behaves identically.

**Tests:**
- `predict(1.0)` produces numerically identical `x` and `P` to the current code for a fixed input sequence (parametrize over all three estimators).
- Synthetic 1D constant‑velocity trajectory `p_k = p_0 + v · t_k` with non‑uniform timestamps converges in position and velocity error (sanity check that `F(Δt)` / `Q(Δt)` are wired correctly).

**Risk:** low. No behaviour change at any existing call site.

#### PR 2 — Per‑update `timestamp` and time‑denominated pruning (SORT + ByteTrack only)

**Goal:** thread `timestamp` from `update()` down to `predict(dt)` and replace SORT and ByteTrack's frame‑counted lifetime threshold with a time‑denominated one. **OC‑SORT and BoT‑SORT are explicitly left unchanged** (they keep their integer frame counters and will silently ignore `timestamp` if passed, emitting a one‑time warning).

**Files touched:**
- `src/trackers/utils/base_tracklet.py` — add a parallel `time_since_update_seconds: float` field accumulated as `+= dt` in `predict(dt)`. Keep the existing integer `time_since_update` so OC‑SORT/BoT‑SORT remain byte‑identical.
- `src/trackers/core/base.py` — `update(..., timestamp: float | None = None)`; `_compute_dt(timestamp)` helper; track `_last_timestamp` per tracker instance; first‑call bootstrap with `dt = 1/frame_rate`; `dt ≤ 0` → no‑op + warn once (§4.6).
- `src/trackers/core/sort/tracker.py` + `src/trackers/core/sort/utils.py` — derive `maximum_time_without_update = lost_track_buffer / 30.0` at init; expose new optional `maximum_time_without_update` kwarg as override; switch `_get_alive_tracklets` comparison to seconds.
- `src/trackers/core/bytetrack/tracker.py` + `src/trackers/core/bytetrack/utils.py` — same pattern as SORT.
- `src/trackers/core/ocsort/*` + `src/trackers/core/botsort/*` — **untouched**. Their `update()` accepts `timestamp` but emits a `UserWarning("dynamic dt not yet supported for OC‑SORT; falling back to Fixed mode")` on first non‑`None` `timestamp` and proceeds with `dt = 1/frame_rate`.

**Public API change:**
- New optional kwarg `timestamp: float | None = None` on `BaseTracker.update`.
- New optional kwarg `maximum_time_without_update: float | None = None` on SORT and ByteTrack constructors.

**Tests:**
- Backwards compatibility: SORT and ByteTrack with `timestamp=None` produce identical tracks to today on the existing benchmark fixtures (within strict numerical tolerance).
- "Equivalent timing" test: a synthetic video of constant‑velocity boxes generates identical tracks whether you feed (a) every frame at 30 FPS with no timestamps, or (b) every frame at 30 FPS with `timestamp = frame_idx / 30.0`. The two paths should agree to ≤ 1e‑6.
- "Frame‑skip equivalence" test: feeding every 3rd frame with correct timestamps from a 30 FPS source produces results numerically close to feeding the same frames to a 10 FPS instance (validates that `dt` is doing real work).

**Risk:** low‑to‑medium. The new `time_since_update_seconds` field is purely additive; the only behavioural change is when SORT/ByteTrack are constructed with `frame_rate != 30.0` *and* the user supplies timestamps.

#### PR 3 — Benchmark sweep + utility verdict

**Goal:** quantify whether dynamic‑dt actually helps. **This is where the project lives or dies.**

**Files touched:**
- `benchmark/scripts/` — add a `frame_skip` flag to the existing benchmark harness so it can sample every Nth frame of MOT17 / SportsMOT and feed the surviving frames to the tracker with the correct timestamps.
- New script `benchmark/scripts/dynamic_fps_sweep.py` — runs SORT and ByteTrack in two modes:
  1. **Static**: tracker constructed with `frame_rate = source_fps / n_d`; timestamps not passed; today's behaviour rescaled.
  2. **Dynamic**: tracker constructed with `frame_rate = source_fps` (the *original* rate as prior); timestamps from sampled frame indices passed per update.
  …for `n_d ∈ {1, 2, 3, 6, 10}` on MOT17‑val and SportsMOT‑val.
- `docs/trackers/comparison.md` — append a "Variable frame rate" section with the resulting plots (HOTA / IDF1 / MOTA vs. `n_d`, Static vs. Dynamic curves).

**Public API change:** none.

**Tests:** the benchmark script itself is the test. No new unit tests beyond what PR 2 added.

**Risk:** low (no production code changes).

### Decision gate

#### PR 4 — Utility verdict (not a code PR; a documented decision)

After PR 3 lands, we read the plots and pick **exactly one** of:

| Verdict | Trigger | Next step |
|---------|---------|-----------|
| **Useful**     | Dynamic mode improves HOTA or IDF1 by ≥ 1 point at `n_d ≥ 3` on at least one dataset, without regressing the `n_d = 1` baseline | Proceed to PR 5–7 (OC‑SORT, BoT‑SORT, CLI). |
| **Inconclusive** | Mixed results, < 1 point delta, or sensitive to hyperparameters | Run an OC‑SORT pilot (smaller version of PR 5 scope) before committing to the full follow‑up. |
| **Not useful** | Dynamic mode matches or underperforms Static across the sweep | Stop. Keep PR 1 in main (it's a clean no‑op refactor); revert PR 2 or hide it behind a feature flag. Document the negative result in `docs/design/dynamic-frame-rate.md`. |

This is the **only mandatory decision gate** in the plan. Everything below is conditional on a "Useful" verdict.

### Conditional follow‑ups (only if PR 4 = "Useful")

#### PR 5 — BoT‑SORT extension (small)

- `src/trackers/core/botsort/tracker.py` — swap `tsu > 1` lost‑gate to `tsu_seconds > one_frame_period`; derive `maximum_time_without_update` like SORT/ByteTrack; remove the OC‑SORT/BoT‑SORT warning added in PR 2.
- Tests: backwards compatibility + frame‑skip equivalence (same shape as PR 2 tests).
- **Risk:** low. The state‑estimator support is already in PR 1; only tracker‑level wiring.

#### PR 6 — OC‑SORT time refactor (large)

- `src/trackers/core/ocsort/tracklet.py` — observations stored as `(timestamp, bbox)` pairs; `delta_t_seconds` replaces integer `delta_t`; velocity computed as `Δbbox / Δt_obs`; ORU virtual trajectory uses `n = max(1, round(Δt_gap · frame_rate))` sub‑steps each with `dt = Δt_gap / n`.
- `src/trackers/core/ocsort/utils.py` — `get_k_previous_obs` becomes a time‑indexed lookup; OCM cost matrix construction unchanged (direction cost is scale‑invariant).
- Tests: reproduce the OC‑SORT paper's Table 7 `Δt` ablation pattern on DanceTrack‑val; backwards‑compat at `frame_rate = 30` and `timestamp = None`.
- **Risk:** medium. ORU is the most subtle piece of the OC‑SORT codebase.

#### PR 7 — CLI / demo / tuner wiring

- `src/trackers/scripts/track.py` — derive `timestamp = cap.get(CAP_PROP_POS_MSEC) / 1000.0` (with `frame_idx / source_fps` fallback) when a new `--tracker.dynamic_dt true` flag is set; pass to `tracker.update`.
- `demo/app.py` — same wiring; use `source_info.fps`.
- `src/trackers/tune/*` — optionally include `maximum_time_without_update` in the search space.
- **Risk:** low.

### Summary

```
PR 1  →  PR 2  →  PR 3  ──► [decision]  ──► PR 5
                                         └─► PR 6
                                         └─► PR 7
```

PR 1 always lands (it's a no‑op refactor that other PRs need). PRs 2–3 ship the smallest possible feature that can be benchmarked against the static baseline. PR 4 is a written decision, not code. PRs 5–7 are only worth our time if the benchmark confirms the hypothesis.

## 7. Design decisions (resolved)

1. **Time unit at the API: seconds (`float`).** Alternative considered: nanosecond `int` (avoids float drift over long sessions). Rejected because `time.monotonic()` and OpenCV's `CAP_PROP_POS_MSEC` are both float‑native, and the typical session length is bounded by a video / camera run.
2. **Process‑noise discretization: DWNA (§4.2), no alternative exposed.** Acceleration is treated as a piecewise‑constant random variable per step with variance `σ_a²`, redrawn independently each step. CWNA is documented in §4.2 only as background — it is *not* offered as a runtime option, to keep the public surface small and avoid users accidentally mixing models within one experiment. We can add an ablation knob later if benchmarks justify it.
3. **No public `dt` clamps.** We assume the caller's `dt` is well‑formed: small and positive, on the order of one frame period. The natural behaviour of the pipeline — `Q` grows, gating widens, unmatched track is pruned by `maximum_time_without_update` — already covers the "long gap" case correctly, so a `dt_max` knob would be redundant surface area. The single internal defensive guard is a no‑op predict on `dt ≤ 0` (input bug; one‑time warning); see §4.6.
4. **`frame_rate` is required, in both modes.** In Fixed mode it sets the per‑step `dt`. In Dynamic mode it is the *prior reference* used to (a) calibrate `σ_a²` for `Q(Δt)`, (b) bootstrap `dt` on the very first `update()` call before any `t_prev` exists, and (c) convert frame‑denominated parameters (`lost_track_buffer`, OC‑SORT `delta_t`, BoT‑SORT lost‑gate) into seconds. Auto‑estimating `frame_rate` from the first N timestamps was considered and rejected: it makes the first N updates depend on call patterns rather than on a documented value, which is hard to test and reason about.

## 8. Worked example — back‑calibration for SORT XYXY

Current calibration (`src/trackers/core/sort/tracklet.py:75`): `Q = np.eye(8) * 0.01`, with `frame_rate = 30.0` default.

DWNA equivalent:

```python
Δt_ref = 1.0 / 30.0
σ_a² = 0.01 / Δt_ref**2     # = 9.0 (px/s²)² per coordinate
```

Apply per coordinate `i ∈ {0..3}` (one for each of `x1, y1, x2, y2`):

```python
Q_DWNA(Δt)[i,   i  ] = σ_a²[i] · Δt⁴ / 4
Q_DWNA(Δt)[i,   i+4] = σ_a²[i] · Δt³ / 2
Q_DWNA(Δt)[i+4, i  ] = σ_a²[i] · Δt³ / 2
Q_DWNA(Δt)[i+4, i+4] = σ_a²[i] · Δt²
```

At `Δt = 1/30` we get `Q_DWNA[i+4, i+4] = 9.0 · (1/30)² = 0.01` ✓ (matches today's velocity diagonal).
At `Δt = 1` we get `Q_DWNA[i+4, i+4] = 9.0` — 900× more uncertain — which is the *correct* prior if the user is actually feeding frames 1 second apart.

The position diagonal at `Δt = 1/30` is `9.0 · (1/30)⁴ / 4 ≈ 2.78e‑6` — a hair smaller than today's `0.01`. We absorb the small discrepancy into `process_noise_scale` so users can pin parity.

## 9. References

### Discretization & process noise

- Yaakov Bar‑Shalom, X. Rong Li, Thiagalingam Kirubarajan. *Estimation with Applications to Tracking and Navigation: Theory, Algorithms, and Software.* Wiley, 2001. **§6.2 (CWNA), §6.3 (DWNA)** — canonical derivations.
- Robert Grover Brown & Patrick Y. C. Hwang. *Introduction to Random Signals and Applied Kalman Filtering.* 4th ed., Wiley, 2012. §5.5–§5.6.
- Mohinder S. Grewal & Angus P. Andrews. *Kalman Filtering: Theory and Practice Using MATLAB.* 4th ed., Wiley, 2014. §4.5 on discretizing continuous‑time models.
- Roger Labbe. *Kalman and Bayesian Filters in Python.* Chapter 7 §7.5 "Design of the Process Noise Matrix" — [Github](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb).
- FilterPy library docs — [`Q_discrete_white_noise`](https://filterpy.readthedocs.io/en/latest/common/common.html#filterpy-common-q-discrete-white-noise) and [`Q_continuous_white_noise`](https://filterpy.readthedocs.io/en/latest/common/common.html#filterpy-common-q-continuous-white-noise).
- Wikipedia, "[Kalman filter — Details](https://en.wikipedia.org/wiki/Kalman_filter#Details)" — derivation of `F(dt)` and the process noise integral.

### Tracker‑specific motivations

- Jinkun Cao et al. *Observation‑Centric SORT: Rethinking SORT for Robust Multi‑Object Tracking.* CVPR 2023. [arxiv:2203.14360](https://arxiv.org/abs/2203.14360). §3‑§4 derive the σ² ∝ 1/Δt² scaling of velocity noise; Table 7 ablates `Δt ∈ {1, 2, 3, 6}` for OCM.
- Fan Zhou et al. *APPTracker: Improving Tracking Multiple Objects in Low‑Frame‑Rate Videos.* ACM MM 2022. [PDF](https://infzhou.github.io/folder/Zhou_APPTracker_Improving_Tracking_Multiple_Objects_in_Low-Frame-Rate_Videos_MM_2022.pdf). Table 5 reports buffer sizes per frame‑skip — empirical confirmation of seconds‑denominated thresholds.
- Wei Liu et al. *Towards Frame Rate Agnostic Multi‑Object Tracking* (FAPS). [arxiv:2209.11404](https://arxiv.org/abs/2209.11404); code: [Helicopt/FraMOT](https://github.com/Helicopt/FraMOT). Defines the known‑FPS / unknown‑FPS API split we adopt.
- Yifu Zhang et al. *ByteTrack: Multi‑Object Tracking by Associating Every Detection Box.* ECCV 2022. Author confirms low‑FPS weakness in [ifzhang/ByteTrack#64](https://github.com/ifzhang/ByteTrack/issues/64).
- Nir Aharon, Roy Orfaig, Ben‑Zion Bobrovsky. *BoT‑SORT: Robust Associations Multi‑Pedestrian Tracking.* arXiv 2022.
- Ziwei Wang et al. *Asynchronous Blob Tracker for Event Cameras.* [arxiv:2307.10593](https://arxiv.org/abs/2307.10593) — extreme variable‑dt EKF formulation; useful as a reference for the per‑update timestamp pattern.
