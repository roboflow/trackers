# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🔧 Fixed

- **MOT evaluation no longer silently drops ground-truth rows with a class below 1** — MOT15-style files, and the example in the evaluation docs, carry `-1` in column 8, which the loader reads as the class. Those rows matched no scored class, so they were dropped with no warning: predictions identical to an all `-1` file gave MOTA 0 and every prediction a false positive. `evaluate_mot_sequence` and `evaluate_mot_sequences` now raise `ValueError` when a row not marked ignored has a class below 1 that the class config does not name, and `trackers eval` exits 1 with that message. The message names the file and the classes, and gives a `class_config` value that scores those rows alongside the classes already scored. A file mixing class 1 with such rows used to return numbers for the class 1 rows only and now raises too. Classes of 1 and above, ignored rows and tracker files are not checked. `trackers tune` has no `class_config` option, so there the file has to be fixed ([#661](https://github.com/roboflow/trackers/pull/661)).

## [2.7.0] — 2026-11-05

### 🚀 Added

- **Optional BoT-SORT appearance Re-ID** — `BoTSORTTracker(reid_model=..., reid_config=ReIDConfig(...))` accepts an encoder object or model-source string. The `trackers[reid]` extra pins the standalone model package to `reid==0.1.0` and includes Matplotlib. Embeddings augment the first and unconfirmed association stages; low-confidence second-stage matching remains geometry-only. Re-ID is off by default, requires a BGR frame on every `update()`, and is integrated into BoT-SORT only ([#511](https://github.com/roboflow/trackers/pull/511), [#655](https://github.com/roboflow/trackers/pull/655)).

- **Selectable Re-ID fusion** — `ReIDConfig(fusion="botsort")` takes the better geometric/appearance score; opt-in `fusion="adaptive"` adds appearance with a capped best-versus-runner-up margin bonus, adapted from Deep OC-SORT's appearance weighting (not a new Deep OC-SORT tracker). Both methods gate appearance with distance `0.5 * (1 - cosine)` (default `appearance_threshold=0.25`) and standard-IoU proximity (default `proximity_threshold=0.5`). Adaptive defaults are `appearance_weight=0.75`, `adaptive_weight_cap=0.5`; its similarity range can reach 2.25, so first/unconfirmed association thresholds need retuning when switching methods ([#511](https://github.com/roboflow/trackers/pull/511)).

- **Re-ID tuning and analysis** — public `trackers.reid` helpers include `ReIDEncoder`, `FeatureBank`, embedding extraction, appearance similarity, balanced same/different-identity distance sampling, acceptance rates, ROC AUC, frame-gap sweeps and plots. `Tuner` adds per-run `search_space`, `fixed_defaults`, and opt-in `cache_embeddings=False`; conditional nested `reid_config.*` search fields activate with a fixed encoder and require `images_dir`. `fixed_params` overrides `fixed_defaults`; embedding caching trades memory for fewer encoder calls. CLI `track` and `tune` accept `--reid.*`; `track` exposes `--tracker.reid_config.*` ([#511](https://github.com/roboflow/trackers/pull/511)).

- **Grouped CLI and YAML configuration** — `jsonargparse` replaces the argparse entry point, with `--config`, grouped `--detection.*`, `--filters.*`, `--output.*`, `--show.*` and tracker settings. The registry exposes IoU variants, state estimators and McByte mask configuration; hyphens/underscores normalize in option names, and booleans accept explicit values. Python constructor parameter names remain unabbreviated ([#413](https://github.com/roboflow/trackers/pull/413), [#552](https://github.com/roboflow/trackers/pull/552)).

- **Three cookbooks** — Re-ID with appearance-threshold analysis, BoT-SORT camera-motion compensation, and McByte mask tracking are linked from Tutorials. They target 2.7.0 and require the indicated external models, datasets and optional dependencies ([#511](https://github.com/roboflow/trackers/pull/511), [#630](https://github.com/roboflow/trackers/pull/630)).

- **Camera motion compensation for `OCSORTTracker`** — opt-in `enable_cmc` / `cmc_method` / `cmc_downscale`, the parameters BoT-SORT and McByte already expose. With `enable_cmc=True` and the frame passed to `update()`, the shared `CMC` module's estimate warps every track's Kalman state, stored observations (OCM/OCR/velocity) and frozen ORU state into the current frame before association. Off by default, so OC-SORT's defaults and benchmark numbers are unchanged. Contributor-reported validation-split runs with YOLOX detections show gains on MOT17 and SportsMOT and a slight HOTA loss on DanceTrack, so leave it off for static cameras ([#611](https://github.com/roboflow/trackers/pull/611)).

- **`trackers benchmark mcbyte` subcommand** — runs McByte over complete MOT17, DanceTrack, SportsMOT, or SoccerNet-tracking benchmark test sets and writes MOTChallenge-format results, with flags for dataset selection (`--dataset`), per-dataset detection/image roots (`--dataset_roots`), device, output location, CMC method/downscale, isolated mask matching, skip-existing, and partial-result retention ([#541](https://github.com/roboflow/trackers/pull/541), [#543](https://github.com/roboflow/trackers/pull/543)).

- **`trackers inspect` command group** — visual validation commands for the mask stack and the tracker that uses it: `inspect sam` (box-prompted mask generation), `inspect cutie` (mask propagation), `inspect mask-manager` (mask lifecycle over a frame range, driven either from command-line boxes or from a MOT ground-truth file), and `inspect mcbyte` (locked-IoU baseline against full mask-conditioned McByte). Each writes annotated per-frame images into a timestamped run directory ([#543](https://github.com/roboflow/trackers/pull/543)).

- **`BoTSORTTracker.update()` accepts a precomputed camera-motion transform** — the new keyword-only `cmc_transform` argument takes a per-frame 2x3 affine (previous to current frame, the `CMC.estimate()` convention) from external frame registration and applies it in place of the internal frame-based estimate, regardless of `enable_cmc`. Passing both `frame` and `cmc_transform`, or a malformed matrix (non-finite, 3x3 homography, wrong shape), raises `ValueError` before any state change; `CBIoUTracker.update()` accepts `cmc_transform` but ignores it with a `UserWarning` ([#610](https://github.com/roboflow/trackers/pull/610), [#609](https://github.com/roboflow/trackers/issues/609)).

### ⚠️ Compatibility changes

- **CLI Python module path changed** — `trackers.scripts` was removed; use the `trackers` executable or `python -m trackers.cli`. Direct imports from the old implementation modules require migration. The mask-package shims described below do not cover the removed deep modules `trackers.core.mcbyte.masks.{base,sam,cutie,dummy}` ([#413](https://github.com/roboflow/trackers/pull/413), [#543](https://github.com/roboflow/trackers/pull/543)).
- **Dependency floor** — `supervision>=0.31.0` replaces `>=0.26.1`; the base installation also adds `jsonargparse>=4.48.0,<5`. Python remains `>=3.10` ([#413](https://github.com/roboflow/trackers/pull/413), [#639](https://github.com/roboflow/trackers/pull/639)).

### 🔄 Deprecated

- **Legacy CLI spellings** — renamed track/detection/output/show/filter flags, long `minimum_`/`maximum_` CLI parameter prefixes, old list forms and download positional dataset selection warn with removal scheduled for 2.10.0. Use the grouped interface; do not combine a legacy flag with its replacement. `--tracker NAME` remains a supported shorthand for `--tracker.name NAME` ([#413](https://github.com/roboflow/trackers/pull/413), [#552](https://github.com/roboflow/trackers/pull/552)).

- **Legacy mask exports** — the old package-level `MaskGenerator`, `MaskPropagator`, `MaskOutput`, `TrackletSnapshot`, and old `mask_manager.MaskManager` retain deprecation shims until 3.0; import from `trackers.core.masks` or `trackers.core.masks.manager` instead ([#543](https://github.com/roboflow/trackers/pull/543)).

- **`trackers eval --tracker`/`--tracker_dir` renamed to `--predictions`/`--predictions_dir`** — old spellings still parse and emit `FutureWarning`, will be removed in v2.10.0. Use `--predictions`/`--predictions_dir` instead ([#541](https://github.com/roboflow/trackers/pull/541)).

### 🌱 Changed

- **CLI tracking uses source FPS** — a finite positive source frame rate now initializes the tracker unless `--tracker.frame_rate` is explicitly set, correcting buffer/lifetime behavior on non-30-FPS inputs. Pass `--tracker.frame_rate 30` to reproduce the previous CLI setting ([#569](https://github.com/roboflow/trackers/pull/569)).

- **Lower evaluation and prediction overhead** — vectorized Identity ID remapping, reusable contiguous prepared IDs, reduced overlap temporaries and distractor-membership work, direct small diagonal covariance construction, and OC-SORT reuse of the predicted box reduce allocation/computation work. Performance claims remain those of the implementation changes; no fresh benchmark was run for these notes ([#574](https://github.com/roboflow/trackers/pull/574), [#576](https://github.com/roboflow/trackers/pull/576), [#577](https://github.com/roboflow/trackers/pull/577), [#578](https://github.com/roboflow/trackers/pull/578), [#579](https://github.com/roboflow/trackers/pull/579), [#580](https://github.com/roboflow/trackers/pull/580)).

- **McByte skips mask-only association work when mask evidence is unavailable** — clear-match locking and reduced assignment are preserved, while ambiguity and isolated-candidate matrices are no longer built for the default mask-disabled path ([#573](https://github.com/roboflow/trackers/pull/573)).

- **McByte CMC now defaults to `cmc_downscale=6`** — this aggregate-performance default halves median CMC latency versus factor `2` on the complete 45-clip, 1280x720 SportsMOT validation split and passes the dataset-level mean/median quality criterion. The benchmark used ground-truth detections with masks disabled; 9/45 clips regressed under the previous strict per-clip gate. Pass `cmc_downscale=2` to preserve the previous conservative behavior. Generic `CMCConfig` and `BoTSORTTracker` remain at `2`.

- **Mask stack moved from `trackers.core.mcbyte.masks` to `trackers.core.masks`** — SAM mask generation, Cutie propagation, and `MaskManager` reference no tracker and are not McByte-specific, so they now live beside the trackers rather than inside one. Import from `trackers.core.masks` instead ([#543](https://github.com/roboflow/trackers/pull/543)).

- **BoT-SORT / C-BIoU / McByte scale-aware noise refresh avoids full bbox decoding** — the default center-width-height state now reconstructs only the corner pairs needed for width and height while preserving the previous floating-point operation order and output bits.

- **`BoTSORTTracklet` and `McByteTracklet` share their scale-aware noise machinery** — the byte-identical `_SIGMA_*` constants, `_build_*_noise` / `_set_scale_aware_noise` / `_refresh_*_noise_from_state` methods, and the `_clamp_*` helpers now live once in `trackers.utils.scale_aware_tracklet.ScaleAwareNoiseTracklet`, with `_diagonal_matrix` moved to `trackers.utils.matrix`. Both tracklets subclass the new base; no public API or behavior change ([#596](https://github.com/roboflow/trackers/pull/596)).

- **MOT ground-truth class handling is configurable during evaluation preprocessing** — scored and distractor classes now come from a `MOTClassConfig` (the `mot17` default, a `mot20` preset that treats class 6 as a distractor, or a custom config) instead of hard-coded MOT17 constants. Default MOT17 behavior is unchanged. The config is accepted as `class_config` by `evaluate_mot_sequence` and `evaluate_mot_sequences` (default `"mot17"`) and as `--class_config` on `trackers eval`, and a `UserWarning` fires when a MOT20 sequence is scored with the `mot17` preset ([#597](https://github.com/roboflow/trackers/pull/597), [#598](https://github.com/roboflow/trackers/pull/598)).

### 🔧 Fixed

- **Evaluation rejects unknown metric families before file I/O** — use `CLEAR`, `HOTA`, or `Identity`; the single-sequence path no longer silently returns empty metrics for unsupported names ([#598](https://github.com/roboflow/trackers/pull/598)).
- **`evaluate_mot_sequences` no longer discovers hidden directories as MOT-layout sequences** — the hidden-entry filter in sequence discovery tested `p.name`, which is always `gt.txt` for the `<seq>/gt/gt.txt` layout, so it never excluded anything. A hidden directory next to the sequences that contains a `gt/gt.txt`, such as a hidden copy of a sequence left by a backup or sync tool, was evaluated as a sequence, and because no prediction file matches it the whole benchmark aborted with `FileNotFoundError: Tracker file not found: .../.MOT17-02-backup.txt`. The check now tests the sequence directory name, matching the flat layout, which filtered correctly all along. A MOT-layout sequence directory whose own name starts with `.` is therefore no longer discovered; list it in a `seqmap` to evaluate it. Layout detection now skips hidden files too, so a hidden `.txt` file at the root of a MOT-layout `gt_dir` no longer selects the flat layout and fails with `ValueError: No sequences found`. Discovery of non-hidden sequences is unchanged.
- **Empty `update()` results keep the input's `data` keys, `mask` and `metadata`** — every tracker built an empty result from `sv.Detections.empty()`, so on frames without detections, or when all detections were discarded, these fields disappeared: `result.data["class_name"]` raised `KeyError` there while working on every other frame. Empty results are now built like non-empty ones, by indexing the input, so they carry the same fields with zero rows. Empty results also keep the input's dtypes, and an input without `confidence` or `class_id` now gives `None` for them on empty frames too, as non-empty frames already did. Non-empty results are unchanged. ([#638](https://github.com/roboflow/trackers/pull/638))
- **OC-SORT ORU no longer corrupts `XCYCWHStateEstimator` tracks** — when a lost track was re-observed, observation-centric re-update (ORU) replayed its interpolated virtual boxes into the Kalman filter as raw `[x1, y1, x2, y2]`, bypassing `bbox_to_measurement`. That is only correct for `XYXYStateEstimator`; with `OCSORTTracker(state_estimator_class=XCYCWHStateEstimator)` the filter measures `[xc, yc, w, h]`, so every re-acquisition after a miss left the posterior box far from the detection (`[81.9, 38.1, 217.2, 200.0]` instead of `[140, 100, 190, 200]` for a 50x100 box re-observed after a two-frame miss) with no error raised. Virtual boxes are now encoded through the estimator's `bbox_to_measurement` before the update. Default `XCYCSRStateEstimator` and `XYXYStateEstimator` output is bit-identical. The `state_estimator_class` docstring now lists `XCYCWHStateEstimator`; OC-SORT's noise tuning is specific to `XCYCSRStateEstimator`; `XYXYStateEstimator` and `XCYCWHStateEstimator` only get its state-covariance and velocity process-noise scaling.
- **BoT-SORT positional constructor compatibility preserved** — restored the parameter order supported by 2.6.1, correcting a regression introduced during development in #611. Existing positional calls remain supported, including calls combined with keyword Re-ID options ([#658](https://github.com/roboflow/trackers/pull/658)).
- **`update()` rejects non-finite boxes before mutating tracker state** — a detection whose `xyxy` contains NaN or inf raised `ValueError` from the IoU guard only after the tracker had already predicted every track and advanced its timestamp anchor (and, in `ByteTrackTracker`, applied stage-1 matches and allocated IDs), so a caller that caught the error and skipped or retried the frame was left with silently aged or half-updated tracks. `BaseTracker._validate_detections` now runs as the first statement of `update()` in all six trackers, so a failed call leaves tracks, IDs and the timestamp anchor untouched. Low-confidence non-finite rows no longer slip through either: `OCSORTTracker`, `BoTSORTTracker`, `CBIoUTracker` and `McByteTracker` silently ignored them (dropping them from the output as of 2.6.0), and all four now raise.
- **`trackers eval` names a MOT-layout sequence after its directory, not `gt`** — `evaluate_mot_sequence` took the result name from the ground-truth file stem, so `trackers eval --gt <seq>/gt/gt.txt` printed `gt` in the results table, in `SequenceResult.sequence` and in the `"sequence"` field of the `--output` JSON. A `<seq>/gt/gt.txt` file is now named `<seq>` (relative paths are resolved against the working directory) and every other ground-truth file keeps its stem. Scripts that matched the literal `gt` in that output need updating ([#634](https://github.com/roboflow/trackers/issues/634), [#635](https://github.com/roboflow/trackers/pull/635)).

### 📖 Documentation and maintenance

- Reorganized documentation into usage, tuning, developer and evaluation sections; added CLI/migration, inspect and benchmark-runner guides, and exposed the canonical changelog and contribution guidance on the documentation site ([#548](https://github.com/roboflow/trackers/pull/548), [#549](https://github.com/roboflow/trackers/pull/549), [#551](https://github.com/roboflow/trackers/pull/551), [#553](https://github.com/roboflow/trackers/pull/553), [#565](https://github.com/roboflow/trackers/pull/565)).
- Updated development/documentation dependency locks, pre-commit hooks and GitHub Actions; build CI now selects Python 3.12. This is a build-job choice, not a Python minimum increase ([#625](https://github.com/roboflow/trackers/pull/625), [#648](https://github.com/roboflow/trackers/pull/648)). No new security fix is claimed solely from dependency bump subjects.

## [2.6.1]

### 🔧 Fixed

- **`GIoU` no longer scores a malformed box above `0`** — `_compute_iou_and_enclosing` computed box area as `(x2 - x1) * (y2 - y1)` without clamping, so a box inverted on both axes contributed a spuriously positive area: two negative extents multiply to a positive product. That inflated the union past the enclosing area and flipped GIoU's `(enclosing - union) / enclosing` penalty into a bonus, driving scores far outside GIoU's documented `[-1, 1]` range (over `+29` on random malformed input) so that a degenerate box could outrank a genuine match. Box extents are now clamped at zero exactly as the intersection already was. Affects `GIoU` only: `IoU`/`BIoU` do not use this helper, and `DIoU`/`CIoU` are byte-identical because a malformed axis already forces the intersection to zero. Reachable when a tracker is constructed with `iou=GIoU()` — `SORTTracker`'s default `XYXYStateEstimator` has an unconstrained velocity and no downstream box clamp, though every tracker exposing an `iou` parameter defaults to plain `IoU()`, and `CBIoUTracker` exposes none. Results for well-formed boxes are bit-identical.
- **OC-SORT now returns low-confidence detections with `tracker_id=-1`** — detections below `high_conf_det_threshold` were previously dropped silently instead of being emitted, unlike `SORTTracker`/`ByteTrackTracker`. `update()` now returns one row per input detection, matching the documented contract. Output-contract change: callers may now see additional `tracker_id == -1` rows ([#566](https://github.com/roboflow/trackers/pull/566)).
- **McByte mask cleanup survives deferred delivery** — removed tracklet IDs stay queued until the mask pipeline receives them, including across skipped frames, duplicate timestamps, and recoverable CUDA out-of-memory failures; tracks pruned early during timestamp updates are included ([#568](https://github.com/roboflow/trackers/pull/568)).
- **BoT-SORT and McByte refresh only the Kalman noise matrix used by the current step** — prediction refreshes process noise (`Q`) and measurement updates refresh measurement noise (`R`), avoiding stale or unnecessary recalculation ([#567](https://github.com/roboflow/trackers/pull/567)).
- **Batched `xcycsr_to_xyxy` preserves fractional coordinates from integer input** — integer arrays now decode to `float64`; `float32` input remains `float32`. Code depending on the old truncated integer dtype should convert explicitly after decoding ([#557](https://github.com/roboflow/trackers/pull/557)).
- **Image-directory frames use natural filename order** — `2.jpg` is read before `10.jpg`; zero-padded and non-numeric names retain their prior order ([#558](https://github.com/roboflow/trackers/pull/558)).
- **Mike-deployed documentation URLs point at the `/latest/` alias** — canonical, Open Graph, structured-data, and sitemap URLs no longer nest the numbered version beneath `/latest/` ([#582](https://github.com/roboflow/trackers/pull/582)).

### 📖 Documentation

- Updated McByte benchmark and comparison tables, tracker defaults, quickstart and evaluation guidance, dataset details, README terminology, search-result titles, and video poster frames to match this release branch ([#539](https://github.com/roboflow/trackers/pull/539), [#547](https://github.com/roboflow/trackers/pull/547), [#555](https://github.com/roboflow/trackers/pull/555), [#564](https://github.com/roboflow/trackers/pull/564), [#583](https://github.com/roboflow/trackers/pull/583), [#585](https://github.com/roboflow/trackers/pull/585), [#587](https://github.com/roboflow/trackers/pull/587)).

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
[2.6.1]: https://github.com/roboflow/trackers/compare/2.6.0...2.6.1
[2.7.0]: https://github.com/roboflow/trackers/compare/2.6.1...2.7.0
[unreleased]: https://github.com/roboflow/trackers/compare/2.7.0...HEAD
