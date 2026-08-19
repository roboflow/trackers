# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🚀 Added

- **`BoTSORTTracker(reid_fusion="adaptive")`** — Deep OC-SORT's adaptive appearance weighting as an alternative to BoT-SORT's `min(d_iou, d_app)` fusion: adds a weighted appearance term whose weight grows when the best match stands clear of the runner-up. New `reid_appearance_weight` and `reid_adaptive_weight_cap` parameters; default fusion is unchanged. `reid_appearance_threshold` has no effect under adaptive fusion. `fuse_adaptive_reid_association` lives in `trackers.core.reid.fusion` beside the BoT-SORT rule so other trackers can reuse it ([#571](https://github.com/roboflow/trackers/pull/571)).
- **`trackers benchmark mcbyte` subcommand** — runs McByte over complete MOT17, DanceTrack, SportsMOT, or SoccerNet-tracking benchmark test sets and writes MOTChallenge-format results, with flags for dataset selection (`--dataset`), per-dataset detection/image roots (`--dataset_roots`), device, output location, CMC method/downscale, isolated mask matching, skip-existing, and partial-result retention ([#541](https://github.com/roboflow/trackers/pull/541), [#543](https://github.com/roboflow/trackers/pull/543)).
- **`trackers inspect` command group** — visual validation commands for the mask stack and the tracker that uses it: `inspect sam` (box-prompted mask generation), `inspect cutie` (mask propagation), `inspect mask-manager` (mask lifecycle over a frame range, driven either from command-line boxes or from a MOT ground-truth file), and `inspect mcbyte` (locked-IoU baseline against full mask-conditioned McByte). Each writes annotated per-frame images into a timestamped run directory ([#543](https://github.com/roboflow/trackers/pull/543)).

### 🔄 Deprecated

- **`trackers eval --tracker`/`--tracker_dir` renamed to `--predictions`/`--predictions_dir`** — old spellings still parse and emit `FutureWarning`, will be removed in v2.10.0. Use `--predictions`/`--predictions_dir` instead ([#541](https://github.com/roboflow/trackers/pull/541)).

### 🌱 Changed

- **McByte CMC now defaults to `cmc_downscale=6`** — this aggregate-performance default halves median CMC latency versus factor `2` on the complete 45-clip, 1280x720 SportsMOT validation split and passes the dataset-level mean/median quality criterion. The benchmark used ground-truth detections with masks disabled; 9/45 clips regressed under the previous strict per-clip gate. Pass `cmc_downscale=2` to preserve the previous conservative behavior. Generic `CMCConfig` and `BoTSORTTracker` remain at `2`.
- **Mask stack moved from `trackers.core.mcbyte.masks` to `trackers.core.masks`** — SAM mask generation, Cutie propagation, and `MaskManager` reference no tracker and are not McByte-specific, so they now live beside the trackers rather than inside one. Import from `trackers.core.masks` instead ([#543](https://github.com/roboflow/trackers/pull/543)).

### 🔧 Fixed

- **OC-SORT now returns low-confidence detections with `tracker_id=-1`** — detections below `high_conf_det_threshold` were previously dropped silently instead of being emitted, unlike `SORTTracker`/`ByteTrackTracker`. `update()` now returns one row per input detection, matching the documented contract. Output-contract change: callers may now see additional `tracker_id == -1` rows ([#566](https://github.com/roboflow/trackers/pull/566)).

## [2.6.0] — 2026-08-03

### 🚀 Added

- **`McByteTracker`** — new mask-conditioned ByteTrack tracker combining mask-conditioned association with SAM box-mask generation and Cutie mask propagation; exported from `trackers` (`McByteMaskConfig` also exported) and CLI-discoverable. New `trackers[mask]` extra installs `torch`, `torchvision`, `rf-segment-anything`, and `rf-cutie[inference]` (`rf-cutie` now ships on PyPI instead of git). Device selection defaults to `"auto"` (CUDA → MPS → CPU); Cutie streaming behaviour is tunable via `cutie_max_internal_size` (480), `cutie_mem_every` (10), and `cutie_use_long_term` (`True`), with `cutie_channels_last` / `cutie_compile` and AMP left off by default; `minimum_mask_creation_frames` (3) defers SAM/Cutie encoding for short-lived tracklets. Frames are expected in **RGB** (`BaseTracker`'s other trackers expect BGR). Mask checkpoints load with `weights_only=True` (CWE-502 guard, never shipped unsafe) ([#388](https://github.com/roboflow/trackers/pull/388), [#418](https://github.com/roboflow/trackers/pull/418), [#441](https://github.com/roboflow/trackers/pull/441), [#452](https://github.com/roboflow/trackers/pull/452), [#459](https://github.com/roboflow/trackers/pull/459), [#481](https://github.com/roboflow/trackers/pull/481), [#491](https://github.com/roboflow/trackers/pull/491), [#508](https://github.com/roboflow/trackers/pull/508), [#513](https://github.com/roboflow/trackers/pull/513), [#519](https://github.com/roboflow/trackers/pull/519), [#520](https://github.com/roboflow/trackers/pull/520), [#521](https://github.com/roboflow/trackers/pull/521), [#522](https://github.com/roboflow/trackers/pull/522), [#523](https://github.com/roboflow/trackers/pull/523), [#524](https://github.com/roboflow/trackers/pull/524), [#525](https://github.com/roboflow/trackers/pull/525), [#526](https://github.com/roboflow/trackers/pull/526), [#529](https://github.com/roboflow/trackers/pull/529), [#532](https://github.com/roboflow/trackers/pull/532)).
- **Optional `timestamp=` on `BaseTracker.update()`** — all six trackers convert elapsed wall-clock seconds into Kalman frame units and prune lost tracks on a seconds budget when timestamps are supplied; omitting `timestamp` preserves fixed-rate behaviour ([#446](https://github.com/roboflow/trackers/pull/446)).
- **`KalmanMotionModel`** in `trackers.utils.motion_models` — supplies the Kalman `F` and `Q` for a given `frame_step`; `F` is a trivial constant-velocity matrix (`constant_velocity_F`) while `ScalableProcessNoise` holds the tuned `Q`, used at the nominal step and DWNA-scaled on timestamp gaps.

### ⚠️ Breaking Changes

- **Invalid lost-track buffer settings now raise `ValueError`** — `lost_track_buffer` must be non-negative and `frame_rate` must be finite and positive for `SORTTracker`, `ByteTrackTracker`, `OCSORTTracker`, `BoTSORTTracker`, and `CBIoUTracker` (which forwards its constructor args to `BoTSORTTracker`). Explicit `lost_track_buffer=0` remains valid and means no missed-frame grace period; negative buffers and invalid frame rates previously initialized but produced nonsensical lifecycle behavior ([#420](https://github.com/roboflow/trackers/pull/420)).
- **Confirmed tracks now survive one additional missed frame** — all trackers changed from exclusive (`time_since_update < maximum_frames_without_update`) to inclusive (`<=`) boundary semantics to match OC-SORT's previous behavior. Users comparing metric results across this version should expect small IDSW/HOTA shifts ([#420](https://github.com/roboflow/trackers/pull/420)).

### 🌱 Changed

- **Improved tracker performance** (bit-identical output) — CMC's sparse optical-flow status-mask filtering vectorized (~24x faster for that operation); `KalmanMotionModel` now caches its transition/noise matrices and defers DWNA calibration (~38% lower BoT-SORT/CBIoU predict cost); predicted boxes cached across association stages in BoT-SORT/CBIoU; Kalman-state-estimator copies trimmed and OC-SORT association hygiene improved ([#522](https://github.com/roboflow/trackers/pull/522), [#527](https://github.com/roboflow/trackers/pull/527), [#528](https://github.com/roboflow/trackers/pull/528)).
- Direct dependency constraint `pydeprecate>=0.7.0` raised to `>=0.8.0`; build backend migrated from `hatchling` to `setuptools` (`src`-layout discovery + `py.typed` support) ([#492](https://github.com/roboflow/trackers/pull/492)).

### 🔧 Fixed

- **Positive low-FPS lost-track buffers no longer collapse to zero frames** — all trackers now scale positive `lost_track_buffer` values with `ceil(...)` and keep confirmed tracks alive through exactly the scaled number of missed frames, matching OC-SORT's previous inclusive boundary semantics ([#420](https://github.com/roboflow/trackers/pull/420)).
- **BoT-SORT / CBIoU: instant-activated first-frame tracks no longer dropped on a single miss** — sticky maturity now triggers once a real `tracker_id` is assigned (`tracker_id != -1`), matching ByteTrack; fixes a default-config ID switch when a frame-1 object is missed once ([#478](https://github.com/roboflow/trackers/pull/478) BoT-SORT, [#504](https://github.com/roboflow/trackers/pull/504) CBIoU).
- **ByteTrack / BoT-SORT / CBIoU now return unmatched detections between the two confidence thresholds** — detections below `track_activation_threshold` but above `high_conf_det_threshold` are now emitted with `tracker_id=-1` instead of silently dropped, matching the documented `update()` contract. Output-contract change: callers may now see additional `tracker_id == -1` rows ([#475](https://github.com/roboflow/trackers/pull/475)).
- **Signed IoU variants clamped to `[0, 1]` before score fusion** — the CIoU floor (~-1.5 from the aspect penalty) no longer produces negative fused similarities; the clamp is a no-op for GIoU/DIoU ([#476](https://github.com/roboflow/trackers/pull/476)).
- **OC-SORT: Kalman scale kept positive across frame-step gaps** — `clamp_velocity` no longer allows scale to collapse negative over long timestamp gaps ([#509](https://github.com/roboflow/trackers/pull/509)).
- **CMC: handle mid-stream frame-resolution changes in optical-flow motion estimation** — adds a re-sync/resize path when the input resolution changes between frames ([#505](https://github.com/roboflow/trackers/pull/505)).
- **CMC: fixed `cv2.resize` crash on tiny downscaled images** — dimensions are now clamped to `max(1, ...)`, covering 1x1, 1x3, and 3x1 edge cases ([#488](https://github.com/roboflow/trackers/pull/488)).
- **`xcycsr_to_xyxy`: prevent zero-division on zero-aspect boxes** — an `r1 != 0` guard now yields height `0` instead of `inf` ([#485](https://github.com/roboflow/trackers/pull/485)).
- **Video output resized on mid-stream resolution change** — `VideoOutput` now resizes frames to match the writer's configured size (`INTER_AREA` on downscale only) ([#514](https://github.com/roboflow/trackers/pull/514)).
- **Package builds fixed for `src` layout** — corrected `setuptools` package discovery and typed-package (`py.typed`) support, plus release CI artifact handling and publish triggers ([#492](https://github.com/roboflow/trackers/pull/492)).

### 🔒 Security

- **Hardened ZIP extraction against Zip Slip / path traversal (CWE-22)** — archive members with absolute paths, traversal segments, empty names, or Windows-style backslash paths are now rejected; validated archives are contained so they cannot write outside the destination directory, including path-swap protection ([#495](https://github.com/roboflow/trackers/pull/495)).

## [2.5.0] — 2026-06-22

### 🚀 Added

- **Pluggable IoU variants** — `iou=` parameter on all four trackers (`SORTTracker`, `ByteTrackTracker`, `OCSORTTracker`, `BoTSORTTracker`) accepts any `BaseIoU` subclass. Built-in variants: `IoU` (standard), `GIoU`, `DIoU`, `CIoU`, `BIoU` (Buffered IoU) ([#403](https://github.com/roboflow/trackers/pull/403)).
- **`BaseIoU` ABC** in `trackers.utils.iou` — defines the `compute(boxes_1, boxes_2)` contract; subclass and override `_compute` to implement a custom similarity metric ([#403](https://github.com/roboflow/trackers/pull/403)).
- **`normalize_for_fusion` on `BaseIoU`** — signed variants (GIoU, DIoU, CIoU) override this to shift `[-1, 1]` → `[0, 1]` before BoT-SORT score fusion, preventing ranking inversion ([#403](https://github.com/roboflow/trackers/pull/403)).
- **`CBIoUTracker`** — Cascaded-Buffered IoU tracker (Yang et al., WACV 2023). Two-stage matching with independently tunable `buffer_ratio_first` / `buffer_ratio_second` buffer scales; inherits ByteTrack-style low-confidence second pass from BoT-SORT ([#417](https://github.com/roboflow/trackers/pull/417)).
- **`py.typed` marker** — PEP 561 compliance; IDEs and type checkers now recognise the package as typed without `--ignore-missing-imports`.

### 🔄 Deprecated

- **`SORTTracker.trackers`** — deprecated alias for `.tracks`; emits `FutureWarning` since v2.5, will be removed in v3.0. Use `tracker.tracks` instead.
- **`trackers.core.botsort.cmc` module** — `CMC` moved to `trackers.utils.cmc`; old path re-exports all symbols with `DeprecationWarning` until v3.0. Migrate: `from trackers.utils.cmc import CMC` or `from trackers import CMC` ([#414](https://github.com/roboflow/trackers/pull/414)).
- **`BoTSORTTracker.apply_cmc_batch`** — use `CMC.apply_batch(H, tracker.tracks)` directly. Will be removed in v3.0 ([#414](https://github.com/roboflow/trackers/pull/414)).
- **`CMCTMethod` type alias** — kept as a back-compat alias for `CMCMethod`; will be removed in v3.0. Migrate to `CMCMethod` ([#414](https://github.com/roboflow/trackers/pull/414)).
- **`CMC.apply_to_xyxy` renamed to `CMC.warp_xyxy_corners`** — old name kept as a deprecated wrapper that forwards to the new name; will be removed in v3.0. Update call sites to `CMC.warp_xyxy_corners` ([#414](https://github.com/roboflow/trackers/pull/414)).

### ⚠️ Breaking Changes

- **Internal tracklet ID counters removed** — track IDs are now allocated by each tracker instance instead of the class-level counters on each `*Tracklet` subclass (e.g. `BoTSORTTracklet.get_next_tracker_id()`). Internal tracklet subclassers should allocate IDs in tracker code and assign `tracklet.tracker_id` directly. Use `self._allocate_tracker_id()` (inherited from `BaseTracker`) as the replacement allocator when implementing a custom tracker subclass.

### 🌱 Changed

- **`CMC`, `CMCConfig`, `CMCMethod` moved to `trackers.utils.cmc` and re-exported from top-level `trackers` package** — import directly with `from trackers import CMC`; old `trackers.core.botsort.cmc` path kept as a deprecated shim ([#414](https://github.com/roboflow/trackers/pull/414)).
- **`CMC.warp_xyxy_corners`** — `apply_to_xyxy` renamed to `warp_xyxy_corners`; old name kept as a deprecated wrapper until v3.0 ([#414](https://github.com/roboflow/trackers/pull/414)).
- **`CMC.apply_batch` homogeneity guard** — now raises `TypeError` immediately when the tracklet list contains mixed state-estimator types, preventing silent state corruption ([#414](https://github.com/roboflow/trackers/pull/414)).
- **`BoTSORTTracklet.apply_cmc` delegates to `CMC.apply_batch`** — per-track and batch paths now share identical code; behaviour is unchanged ([#414](https://github.com/roboflow/trackers/pull/414)).
- **`Tuner` gains `enqueue_defaults`, `fixed_params`, `images_dir`, `seed`** — `enqueue_defaults=True` (default) evaluates a baseline trial using each param's `__init__` default before Optuna samples; `fixed_params` holds selected params constant across all trials; `images_dir` enables frame loading for CMC-enabled trackers; `seed` makes TPE sampling reproducible ([#427](https://github.com/roboflow/trackers/pull/427)).

### 🔧 Fixed

- Clarified in docs that `SORTTracker` itself is not deprecated — only the `.trackers` alias is.
- **BoT-SORT score fusion with signed IoU** — `_fuse_score` multiplied raw negative IoU values by confidence, inverting track ranking for GIoU/DIoU/CIoU; `normalize_for_fusion` now normalises similarity before fusion ([#403](https://github.com/roboflow/trackers/pull/403)).
- **Non-finite box coordinates crash `linear_sum_assignment`** — `BaseIoU.compute` now raises `ValueError` with a clear message for NaN/inf inputs instead of propagating invalid entries into SciPy ([#403](https://github.com/roboflow/trackers/pull/403)).
- **OC-SORT Observation-Centric Recovery** now uses standard `IoU` per the paper, independent of the configured `iou=` variant ([#403](https://github.com/roboflow/trackers/pull/403)).
- **Eager division warnings on zero-area boxes** — IoU helper switched from `np.where` (eager) to `np.divide(..., where=...)` (lazy), suppressing `RuntimeWarning` under strict NumPy error settings ([#403](https://github.com/roboflow/trackers/pull/403)).
- **CLI argparse crash on `BaseIoU` parameter** — `iou=` is now excluded from argparse auto-discovery; the variant must be set programmatically ([#403](https://github.com/roboflow/trackers/pull/403)).
- **ByteTrack tracked nothing when detections lacked confidence scores** — the default-fill changed from `np.zeros` to `np.ones`, matching SORT / OC-SORT / BoT-SORT behaviour, so detectors that emit `sv.Detections` without `confidence` now produce tracks instead of empty results ([#415](https://github.com/roboflow/trackers/pull/415)).
- **Tracker instances no longer share track ID counters** — resetting one tracker instance no longer resets another instance's ID allocator, preventing duplicate live IDs in multi-camera, class-specific, or parallel tracker workflows.
- **HOTA per-frame alpha loop vectorized** — removes the inner Python loop; large evaluations run significantly faster with no change to numeric output ([#462](https://github.com/roboflow/trackers/pull/462)).
- **MOT evaluation distractor handling** — ground-truth preprocessing now applies distractor class filtering consistent with TrackEval, correcting reported metrics on MOT17 and similar datasets ([#466](https://github.com/roboflow/trackers/pull/466)).

---

## [2.4.0] — 2026-05-06

### 🚀 Added

- **BoT-SORT tracker** (`BoTSORTTracker`) — new tracker with optional camera motion compensation (CMC), configurable methods (`orb`, `sift`, `sparseOptFlow`, `ecc`), and ByteTrack-style score-fused association ([#386](https://github.com/roboflow/trackers/pull/386)).
- **`tracked_objects` property on `BaseTracker`** and all concrete trackers — exposes every alive track with its Kalman-predicted bounding box, including occluded or detector-missed tracks. `update()` return value is unchanged for backward compatibility ([#373](https://github.com/roboflow/trackers/pull/373), resolves [#105](https://github.com/roboflow/trackers/issues/105)).
- **`Tuner` class** (`trackers.tune.Tuner`) — Optuna-based hyperparameter optimisation driven by each tracker's new `search_space` ClassVar. Supports HOTA / MOTA / IDF1 objectives over MOT-format ground-truth and pre-computed detections ([#301](https://github.com/roboflow/trackers/pull/301)).
- **`trackers tune` CLI subcommand** — wires `Tuner` into the CLI; selects tracker, ground-truth directory, detections directory, objective, and `--n-trials` ([#374](https://github.com/roboflow/trackers/pull/374)).
- **`load_mot_file` is now public** — was `_load_mot_file`. Now exported from `trackers.io.mot` for use in custom tuning and evaluation scripts ([#301](https://github.com/roboflow/trackers/pull/301), [#374](https://github.com/roboflow/trackers/pull/374)).
- **`xyxy_to_xywh` and `xywh_to_xyxy` converters** added to `trackers.utils.converters` for center-width-height format support ([#310](https://github.com/roboflow/trackers/pull/310), [#386](https://github.com/roboflow/trackers/pull/386)).
- **`frame` parameter on `BaseTracker.update()`** — `update(detections, frame=None)`. Required by BoT-SORT when CMC is enabled; ignored (with `UserWarning`) by SORT, ByteTrack, OC-SORT. The `track` CLI passes the current frame automatically ([#386](https://github.com/roboflow/trackers/pull/386)).
- **Swappable Kalman state estimators** — `BaseStateEstimator`, `XCYCSRStateEstimator`, `XCYCWHStateEstimator`, `XYXYStateEstimator` in `trackers.utils.state_representations`; trackers can opt in via `state_estimator_class=` ([#310](https://github.com/roboflow/trackers/pull/310)).
- **`TrackletProtocol`** structural type in `trackers.core.base` — formalises the contract every tracklet stored in `BaseTracker.tracks` must satisfy.
- **`search_space` ClassVar** on every tracker — declarative hyperparameter spaces consumed by `Tuner`, validated for unknown keys and bad types.
- **Modern Python 3.10+ type hints** across the public surface ([#302](https://github.com/roboflow/trackers/pull/302)).
- **Documentation**: `docs/trackers/botsort.md` user guide, `docs/learn/state-estimators.md`, expanded comparison page with DanceTrack section.

### ⚠️ Breaking Changes

- **`SORTTracker.update()` no longer mutates its input `sv.Detections`** — previously assigned `tracker_id` on the caller's object and returned that same instance; now returns a fresh indexed copy, matching ByteTrack and OC-SORT ([#360](https://github.com/roboflow/trackers/pull/360)). Callers that relied on aliasing the input post-update must read `tracker_id` from the returned object.
- **Per-frame spawn order is now deterministic** across SORT, ByteTrack, and OC-SORT — IDs assigned to detections that spawn in the same frame no longer depend on CPython set iteration order ([#361](https://github.com/roboflow/trackers/pull/361)). IDs from a recorded run are reproducible across machines but may differ from a v2.3.0 baseline.
- **Internal tracklet update contract changed** *(subclassers of internal `*Tracklet` classes only — callers of the public `Tracker.update()` API are unaffected)* — internal tracklet classes (notably `OCSORTTracklet`) no longer accept `update(None)` for unmatched tracks; missed-association logic now lives in `predict()` and `_get_alive_tracklets`. Subclasses that overrode tracklet update behaviour must move that logic into `predict()` ([#383](https://github.com/roboflow/trackers/pull/383), follow-up to [#376](https://github.com/roboflow/trackers/pull/376)).

### 🌱 Changed

- **Refactored Kalman filter out of tracklet classes** — every tracker now shares a single Kalman implementation backed by `BaseStateEstimator`. Tracklet classes (`SORTTracklet`, `ByteTrackTracklet`, `OCSORTTracklet`, `BoTSORTTracklet`) handle association and lifecycle only ([#310](https://github.com/roboflow/trackers/pull/310)).
- **ByteTrack tracklets** now count `number_of_successful_consecutive_updates` instead of total `number_of_updates`, matching the original ByteTrack reference ([#310](https://github.com/roboflow/trackers/pull/310)).
- **Eval submodule** uses lazy `__getattr__` for `evaluate_mot_sequence` and `evaluate_mot_sequences` to avoid circular imports.
- Documentation: rewrote landing page, install guide, evaluate guide, ByteTrack page, comparison page; added DanceTrack default tuned numbers.

### 🔧 Fixed

- **ByteTrack: prune unmatched tracks correctly** after the Kalman refactor — `time_since_update` advances on unmatched tracks and `_get_alive_tracklets` expires them after `lost_track_buffer` empty frames ([#376](https://github.com/roboflow/trackers/pull/376)).
- **Documentation index ByteTrack correction** ([#371](https://github.com/roboflow/trackers/pull/371)).

---

## [2.3.0] — 2026-03-16

### 🚀 Added

- **OC-SORT tracker** (`OCSORTTracker`) — complete implementation with swappable state estimators (`XCYCSRStateEstimator`, `XYXYStateEstimator`), direction-consistency batch calculations, full tracklet lifecycle management, API docs, and unit tests; registered in CLI and public API ([#207](https://github.com/roboflow/trackers/pull/207)).
- **`trackers download` CLI subcommand** — downloads MOT17 and SportsMOT benchmark datasets to a persistent local cache (`~/.cache/trackers`) with MD5 verification and Rich-styled progress output; backed by type-safe `Dataset`, `DatasetSplit`, and `DatasetAsset` enums ([#262](https://github.com/roboflow/trackers/pull/262)).
- **Integration tests with TrackEval** — regression tests for SORT, ByteTrack, and OC-SORT against oracle detections from SportsMOT and DanceTrack; evaluates HOTA, MOTA, IDF1, and IDSW in CI ([#298](https://github.com/roboflow/trackers/pull/298)).
- **Parameter-tuned benchmark results** — tracker comparison page redesigned with tabbed Default / Tuned layout; includes grid-search configs for SportsMOT, SoccerNet, MOT17, and DanceTrack as copyable YAML blocks ([#309](https://github.com/roboflow/trackers/pull/309)).
- **DanceTrack default parameters** — SORT and ByteTrack ship tuned defaults for DanceTrack out of the box ([#299](https://github.com/roboflow/trackers/pull/299)).

### 🌱 Changed

- **Coordinate converter hot-path optimisation** — `xcycsr_to_xyxy` and `xyxy_to_xcycsr` restructured for the single-box case, reducing per-frame overhead in tight tracking loops ([#296](https://github.com/roboflow/trackers/pull/296)).
- **Documentation rewrite** — landing page, install guide, and evaluate pages comprehensively rewritten; tracker comparison page expanded with dataset videos, paper links, and a DanceTrack section ([#322](https://github.com/roboflow/trackers/pull/322)).
- **Release and stable branching strategy adopted** — repository now follows a `release/stable` branching model ([#275](https://github.com/roboflow/trackers/pull/275)).

### 🔧 Fixed

- **Evaluation distractor filtering** corrected on the comparison numbers ([#322](https://github.com/roboflow/trackers/pull/322)).
- **PyPI publish action pinned to verified SHA** — `pypa/gh-action-pypi-publish` corrected to the actual v1.13.0 commit SHA ([#294](https://github.com/roboflow/trackers/pull/294)).

---

## [2.2.0] — 2026-02-18

### 🚀 Added

- **Evaluation metrics** — HOTA, CLEAR (MOTA / MOTP / IDSW / MT / PT / ML), and Identity (IDF1 / IDP / IDR) metric implementations in `trackers.eval`; `evaluate_mot_sequence` and `evaluate_mot_sequences` public API ([#210](https://github.com/roboflow/trackers/pull/210), [#212](https://github.com/roboflow/trackers/pull/212), [#223](https://github.com/roboflow/trackers/pull/223), [#224](https://github.com/roboflow/trackers/pull/224), [#226](https://github.com/roboflow/trackers/pull/226)).
- **`trackers eval` CLI subcommand** — runs a tracker over a MOT-format ground-truth directory and prints HOTA / MOTA / IDF1 results; configurable via JSON tracker arguments ([#215](https://github.com/roboflow/trackers/pull/215)).
- **MOT format I/O** — `load_mot_file` and `save_mot_file` in `trackers.io.mot` for reading and writing MOT-format `.txt` annotation files ([#214](https://github.com/roboflow/trackers/pull/214)).
- **`MotionAwareTraceAnnotator` with camera motion compensation** — applies homography-based CMC to keep trace paths stable on moving-camera footage ([#263](https://github.com/roboflow/trackers/pull/263)).
- **Tracker auto-registration** — `BaseTracker.__init_subclass__` now registers every subclass and extracts parameter metadata from `__init__` docstrings, enabling CLI auto-discovery without a hard-coded tracker list (`TrackerInfo`, `ParameterInfo`) ([#230](https://github.com/roboflow/trackers/pull/230)).
- **Benchmark documentation** — evaluation metrics (HOTA, MOTA, IDF1) for SORT and ByteTrack published to the docs site ([#193](https://github.com/roboflow/trackers/pull/193)).
- **Example notebooks** — links to runnable Colab notebooks added to docs index ([#199](https://github.com/roboflow/trackers/pull/199)).

### 🌱 Changed

- **Apache 2.0 license headers** added to all source files via a new pre-commit hook.
- **Ruff security rules migrated to S (bandit)** — `bandit` pre-commit hook replaced by Ruff's built-in `S` rule set ([#188](https://github.com/roboflow/trackers/pull/188)).
- **Dependency trim** — removed unused optional extras; install footprint reduced ([#192](https://github.com/roboflow/trackers/pull/192)).

---

## [2.1.0] — 2026-01-28

### 🚀 Added

- **ByteTrack tracker** (`ByteTrackTracker`) — two-stage low-score / high-score association with a ByteTrack-specific Kalman box tracker; full API docs and unit tests; registered in CLI and public API ([#174](https://github.com/roboflow/trackers/pull/174)).

### ⚠️ Breaking Changes

- **DeepSort removed** — `DeepSortTracker` and all associated REID infrastructure removed from the package, docs, tests, and CI workflows. Projects using DeepSort must pin to `<2.1.0`.
- **Python 3.9 dropped** — minimum supported version is now Python 3.10; type annotations updated to use built-in generics (`list[...]`, `dict[...]`) throughout the public API ([#200](https://github.com/roboflow/trackers/pull/200)).

### 🔧 Fixed

- **Documentation build** — pinned `mkdocstrings-python<2.0.0` to resolve docs generation failure.

<!-- Reference links -->

[2.1.0]: https://github.com/roboflow/trackers/releases/tag/2.1.0
[2.2.0]: https://github.com/roboflow/trackers/compare/2.1.0...2.2.0
[2.3.0]: https://github.com/roboflow/trackers/compare/2.2.0...2.3.0
[2.4.0]: https://github.com/roboflow/trackers/compare/2.3.0...2.4.0
[2.5.0]: https://github.com/roboflow/trackers/compare/2.4.0...2.5.0
[2.6.0]: https://github.com/roboflow/trackers/compare/2.5.0...2.6.0
[unreleased]: https://github.com/roboflow/trackers/compare/2.6.0...HEAD
