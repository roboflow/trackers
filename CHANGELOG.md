# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
Every released version is grouped by domain (Trackers, Re-ID, Motion & utilities, Evaluation, Tuning, CLI, I/O & datasets, Packaging & docs).
Inside a domain, bullets are ordered _Breaking:_, _Deprecated:_, untagged (new or changed behaviour), _Fixed:_, _Security:_; only the tagged kinds carry a label.
Each bullet opens with a bold headline, then " — " and the detail.
Every bullet links its PR, or its commit SHA when there was no PR.
[Unreleased] may stay unorganised until the release-prep stage, when it is regrouped the same way.
-->

## [Unreleased]

### 🔧 Fixed

- **MOT and detection files saved with a UTF-8 byte order mark now load** — `load_mot_file` (used by `trackers eval`, `trackers track`, the tuner and ReID calibration) and the detection-file reader behind `trackers benchmark` and `trackers inspect` read the mark that Excel's "CSV UTF-8" export and some Windows editors put at the start of a file as part of the first frame number, so a valid `gt.txt` or `det.txt` failed with `Invalid frame number` or `Could not parse detection line 1`. Both readers now open files as `utf-8-sig`, which drops a leading mark and reads every other UTF-8 file unchanged.

## [2.7.0] — 2026-11-05

### Trackers

- **`OCSORTTracker` camera motion compensation** — opt-in `enable_cmc` / `cmc_method` / `cmc_downscale`, as on BoT-SORT and McByte. With `enable_cmc=True` and the frame passed to `update()`, the shared `CMC` estimate warps every track's Kalman state, stored observations (OCM/OCR/velocity) and frozen ORU state before association. Off by default, so defaults and benchmark numbers are unchanged. Contributor-reported validation runs with YOLOX detections gain on MOT17 and SportsMOT and lose slightly on DanceTrack, so leave it off for static cameras ([#611](https://github.com/roboflow/trackers/pull/611)).
- **`BoTSORTTracker.update()` takes a keyword-only `cmc_transform`** — a per-frame 2x3 affine (previous to current frame, the `CMC.estimate()` convention) from external registration, applied in place of the internal estimate regardless of `enable_cmc`. Passing both `frame` and `cmc_transform`, or a malformed matrix (non-finite, 3x3, wrong shape), raises `ValueError` before any state change. `CBIoUTracker.update()` accepts it but ignores it with a `UserWarning` ([#610](https://github.com/roboflow/trackers/pull/610), [#609](https://github.com/roboflow/trackers/issues/609)).
- **McByte `cmc_downscale` now defaults to `6`** — median CMC latency halves versus `2` on the 45-clip 1280x720 SportsMOT validation split and passes the dataset-level mean/median quality criterion. Ground-truth detections, masks disabled; 9/45 clips regressed under the earlier strict per-clip gate. Pass `cmc_downscale=2` for the previous behavior. `CMCConfig` and `BoTSORTTracker` stay at `2` ([#538](https://github.com/roboflow/trackers/pull/538)).
- **McByte skips mask-only association work when mask evidence is unavailable** — clear-match locking and reduced assignment are kept ([#573](https://github.com/roboflow/trackers/pull/573)).
- **the mask stack (SAM generation, Cutie propagation, `MaskManager`) moved from `trackers.core.mcbyte.masks` to `trackers.core.masks`** — since it is not McByte-specific ([#543](https://github.com/roboflow/trackers/pull/543)).
- **`BoTSORTTracklet` and `McByteTracklet` share scale-aware noise code** in `trackers.utils.scale_aware_tracklet.ScaleAwareNoiseTracklet` (`_diagonal_matrix` moved to `trackers.utils.matrix`), and the BoT-SORT / C-BIoU / McByte noise refresh no longer decodes full boxes. No public API or output change ([#596](https://github.com/roboflow/trackers/pull/596)).
- _Deprecated:_ **package-level `MaskGenerator`, `MaskPropagator`, `MaskOutput`, `TrackletSnapshot` and `mask_manager.MaskManager` keep deprecation shims until 3.0** — import from `trackers.core.masks` or `trackers.core.masks.manager`. Deep modules `trackers.core.mcbyte.masks.{base,sam,cutie,dummy}` are removed without shims ([#543](https://github.com/roboflow/trackers/pull/543)).
- _Fixed:_ **OC-SORT observation-centric re-update (ORU) no longer corrupts `XCYCWHStateEstimator` tracks**. It replayed interpolated boxes into the Kalman filter as raw `[x1, y1, x2, y2]`, bypassing `bbox_to_measurement`, so re-acquisition after a miss left the posterior far from the detection (`[81.9, 38.1, 217.2, 200.0]` instead of `[140, 100, 190, 200]` for a 50x100 box after a two-frame miss) without an error. Virtual boxes are now encoded first. `XCYCSRStateEstimator` and `XYXYStateEstimator` output is bit-identical ([#614](https://github.com/roboflow/trackers/pull/614)).
- _Fixed:_ **`update()` validates detections before mutating state** — in all six trackers. A NaN/inf `xyxy` used to raise only after tracks were predicted and the timestamp anchor advanced (and, in ByteTrack, stage-1 matches applied and IDs allocated), leaving a retrying caller with aged or half-updated tracks. Low-confidence non-finite rows, silently ignored by OC-SORT, BoT-SORT, C-BIoU and McByte, now raise too ([#613](https://github.com/roboflow/trackers/pull/613)).
- _Fixed:_ **empty `update()` results keep the input's `data` keys, `mask`, `metadata` and dtypes** — instead of coming from `sv.Detections.empty()`; `result.data["class_name"]` no longer raises `KeyError` on empty frames. Non-empty results are unchanged ([#638](https://github.com/roboflow/trackers/pull/638)).
- _Fixed:_ **`BoTSORTTracker` positional constructor parameter order from 2.6.1 is restored** — correcting a development regression from #611 ([#658](https://github.com/roboflow/trackers/pull/658)).

### Re-ID

- **optional BoT-SORT appearance Re-ID** — `BoTSORTTracker(reid_model=..., reid_config=ReIDConfig(...))` takes an encoder object or model-source string. The `trackers[reid]` extra pins `reid==0.1.0` and includes Matplotlib. Embeddings augment the first and unconfirmed association stages; the low-confidence second stage stays geometry-only. Off by default, needs a BGR frame on every `update()`, BoT-SORT only ([#511](https://github.com/roboflow/trackers/pull/511), [#655](https://github.com/roboflow/trackers/pull/655)).
- **selectable fusion** — `ReIDConfig(fusion="botsort")` takes the better geometric/appearance score; opt-in `fusion="adaptive"` adds appearance with a capped best-versus-runner-up margin bonus, adapted from Deep OC-SORT's appearance weighting (not a new tracker). Both gate appearance by distance `0.5 * (1 - cosine)` (`appearance_threshold=0.25`) and standard-IoU proximity (`proximity_threshold=0.5`). Adaptive defaults: `appearance_weight=0.75`, `adaptive_weight_cap=0.5`. Its similarity can reach 2.25, so retune first/unconfirmed association thresholds when switching ([#511](https://github.com/roboflow/trackers/pull/511)).
- **tuning and analysis helpers in `trackers.reid`** — `ReIDEncoder`, `FeatureBank`, embedding extraction, appearance similarity, balanced same/different-identity sampling, acceptance rates, ROC AUC, frame-gap sweeps and plots. `Tuner` adds per-run `search_space`, `fixed_defaults` (overridden by `fixed_params`) and opt-in `cache_embeddings=False`; conditional nested `reid_config.*` search fields need a fixed encoder and `images_dir`. CLI `track` and `tune` accept `--reid.*`; `track` also `--tracker.reid_config.*` ([#511](https://github.com/roboflow/trackers/pull/511)).

### Evaluation

- **MOT ground-truth classes come from a `MOTClassConfig`** — the `mot17` default, a `mot20` preset that treats class 6 as a distractor, or a custom config. Accepted as `class_config` by `evaluate_mot_sequence` / `evaluate_mot_sequences` and as `--class_config` on `trackers eval`; a `UserWarning` fires when a MOT20 sequence is scored with `mot17`. Default MOT17 behavior is unchanged ([#597](https://github.com/roboflow/trackers/pull/597), [#598](https://github.com/roboflow/trackers/pull/598)).
- **lower evaluation and prediction overhead** — vectorized Identity ID remapping, fewer overlap temporaries and distractor-membership work, direct small diagonal covariance construction, and OC-SORT reuse of the predicted box. No fresh benchmark was run for these notes ([#574](https://github.com/roboflow/trackers/pull/574), [#576](https://github.com/roboflow/trackers/pull/576), [#577](https://github.com/roboflow/trackers/pull/577), [#578](https://github.com/roboflow/trackers/pull/578), [#579](https://github.com/roboflow/trackers/pull/579), [#580](https://github.com/roboflow/trackers/pull/580)).
- _Fixed:_ **unknown metric families are rejected before file I/O** — use `CLEAR`, `HOTA` or `Identity`. The single-sequence path no longer returns empty metrics for unsupported names ([#598](https://github.com/roboflow/trackers/pull/598)).
- _Fixed:_ **`evaluate_mot_sequences` no longer discovers hidden directories as MOT-layout sequences**. The filter tested `p.name`, always `gt.txt` in `<seq>/gt/gt.txt`, so a hidden backup copy of a sequence was evaluated and aborted the run with `FileNotFoundError: Tracker file not found: .../.MOT17-02-backup.txt`. It now tests the sequence directory name, so a MOT-layout directory whose own name starts with `.` is no longer discovered; list it in a `seqmap`. Layout detection also skips hidden files, so a hidden `.txt` at the root of a MOT-layout `gt_dir` no longer fails with `ValueError: No sequences found` ([#642](https://github.com/roboflow/trackers/pull/642)).
- _Fixed:_ **`trackers eval` names a `<seq>/gt/gt.txt` sequence `<seq>` instead of `gt`** in the results table, `SequenceResult.sequence` and the `--output` JSON (relative paths resolve against the working directory). Other ground-truth files keep their stem. Scripts matching the literal `gt` need updating ([#634](https://github.com/roboflow/trackers/issues/634), [#635](https://github.com/roboflow/trackers/pull/635)).

### CLI

- _Breaking:_ **`trackers.scripts` is removed** — use the `trackers` executable or `python -m trackers.cli`. Direct imports from the old implementation modules need migration ([#413](https://github.com/roboflow/trackers/pull/413), [#543](https://github.com/roboflow/trackers/pull/543)).
- _Deprecated:_ **legacy spellings** (renamed track/detection/output/show/filter flags, long `minimum_`/`maximum_` prefixes, old list forms, positional download dataset selection) warn and are removed in 2.10.0. Use the grouped interface and do not combine a legacy flag with its replacement. `--tracker NAME` stays as shorthand for `--tracker.name NAME` ([#413](https://github.com/roboflow/trackers/pull/413), [#552](https://github.com/roboflow/trackers/pull/552)).
- _Deprecated:_ **`trackers eval --tracker` / `--tracker_dir` are renamed `--predictions` / `--predictions_dir`** — old spellings emit `FutureWarning` and are removed in 2.10.0 ([#541](https://github.com/roboflow/trackers/pull/541)).
- **grouped CLI and YAML configuration** — `jsonargparse` replaces argparse, with `--config` and grouped `--detection.*`, `--filters.*`, `--output.*`, `--show.*` and tracker settings. The registry exposes IoU variants, state estimators and McByte mask configuration; hyphens/underscores normalize in option names and booleans accept explicit values. Python constructor parameter names are unchanged ([#413](https://github.com/roboflow/trackers/pull/413), [#552](https://github.com/roboflow/trackers/pull/552)).
- **`trackers benchmark mcbyte`** runs McByte over complete MOT17, DanceTrack, SportsMOT or SoccerNet-tracking test sets and writes MOTChallenge-format results. Flags cover dataset (`--dataset`), per-dataset detection/image roots (`--dataset_roots`), device, output location, CMC method/downscale, isolated mask matching, skip-existing and partial-result retention ([#541](https://github.com/roboflow/trackers/pull/541), [#543](https://github.com/roboflow/trackers/pull/543)).
- **`trackers inspect` group** for visual validation of the mask stack and McByte: `inspect sam`, `inspect cutie`, `inspect mask-manager` (from command-line boxes or a MOT ground-truth file) and `inspect mcbyte` (locked-IoU baseline against full mask-conditioned McByte). Each writes annotated frames into a timestamped run directory ([#543](https://github.com/roboflow/trackers/pull/543)).
- **CLI tracking initializes the tracker with a finite positive source FPS** unless `--tracker.frame_rate` is set, correcting buffer/lifetime behavior on non-30-FPS input. Pass `--tracker.frame_rate 30` for the previous setting ([#569](https://github.com/roboflow/trackers/pull/569)).

### Packaging & docs

- _Breaking:_ **dependency floor** — `supervision>=0.31.0` replaces `>=0.26.1`, and the base install adds `jsonargparse>=4.48.0,<5`. Python stays `>=3.10` ([#413](https://github.com/roboflow/trackers/pull/413), [#639](https://github.com/roboflow/trackers/pull/639)).
- **three cookbooks** linked from Tutorials: Re-ID with appearance-threshold analysis, BoT-SORT camera-motion compensation and McByte mask tracking. They target 2.7.0 and need the stated external models, datasets and optional dependencies ([#511](https://github.com/roboflow/trackers/pull/511), [#630](https://github.com/roboflow/trackers/pull/630)).
- **documentation reorganized into usage, tuning, developer and evaluation sections** — with CLI/migration, inspect and benchmark-runner guides, and the changelog and contribution guidance exposed on the site ([#548](https://github.com/roboflow/trackers/pull/548), [#549](https://github.com/roboflow/trackers/pull/549), [#551](https://github.com/roboflow/trackers/pull/551), [#553](https://github.com/roboflow/trackers/pull/553), [#565](https://github.com/roboflow/trackers/pull/565)).
- **development/documentation locks, pre-commit hooks and GitHub Actions updated** — build CI selects Python 3.12, a build-job choice and not a Python minimum increase ([#625](https://github.com/roboflow/trackers/pull/625), [#648](https://github.com/roboflow/trackers/pull/648)).

## [2.6.1] — 2026-09-25

### Trackers

- _Fixed:_ **`GIoU` no longer scores a malformed box above `0`**. `_compute_iou_and_enclosing` did not clamp box area, so a box inverted on both axes got a positive area, inflated the union past the enclosing area and turned GIoU's penalty into a bonus (over `+29` on random malformed input, far outside `[-1, 1]`). Extents are now clamped at zero like the intersection. Affects `GIoU` only, reachable with `iou=GIoU()`; `IoU`/`BIoU` do not use the helper and `DIoU`/`CIoU` are byte-identical. Well-formed boxes are bit-identical ([#589](https://github.com/roboflow/trackers/pull/589)).
- _Fixed:_ **OC-SORT returns detections below `high_conf_det_threshold` with `tracker_id=-1`** instead of dropping them, matching `SORTTracker` / `ByteTrackTracker` and the documented one-row-per-detection contract. Callers may see additional `tracker_id == -1` rows ([#566](https://github.com/roboflow/trackers/pull/566)).
- _Fixed:_ **McByte mask cleanup survives deferred delivery** — removed tracklet IDs stay queued until the mask pipeline receives them, across skipped frames, duplicate timestamps and recoverable CUDA out-of-memory failures, including tracks pruned early during timestamp updates ([#568](https://github.com/roboflow/trackers/pull/568)).
- _Fixed:_ **BoT-SORT and McByte refresh only the Kalman noise matrix the current step uses** — `Q` on prediction, `R` on measurement update ([#567](https://github.com/roboflow/trackers/pull/567)).

### Motion & utilities

- _Fixed:_ **batched `xcycsr_to_xyxy` decodes integer input to `float64`** — keeping fractional coordinates; `float32` stays `float32`. Convert explicitly if you relied on the truncated integer dtype ([#557](https://github.com/roboflow/trackers/pull/557)).

### I/O & datasets

- _Fixed:_ **image-directory frames use natural filename order** (`2.jpg` before `10.jpg`); zero-padded and non-numeric names keep their order ([#558](https://github.com/roboflow/trackers/pull/558)).

### Packaging & docs

- _Fixed:_ **Mike-deployed documentation URLs (canonical, Open Graph, structured data, sitemap) point at the `/latest/` alias** without nesting the numbered version ([#582](https://github.com/roboflow/trackers/pull/582)).
- **McByte benchmark and comparison tables** — tracker defaults, quickstart and evaluation guidance, dataset details, README terminology, search-result titles and video poster frames updated for this release ([#539](https://github.com/roboflow/trackers/pull/539), [#547](https://github.com/roboflow/trackers/pull/547), [#555](https://github.com/roboflow/trackers/pull/555), [#564](https://github.com/roboflow/trackers/pull/564), [#583](https://github.com/roboflow/trackers/pull/583), [#585](https://github.com/roboflow/trackers/pull/585), [#587](https://github.com/roboflow/trackers/pull/587)).

## [2.6.0] — 2026-08-06

### Trackers

- _Breaking:_ **invalid lost-track buffer settings raise `ValueError`** — `lost_track_buffer` must be non-negative and `frame_rate` finite and positive for `SORTTracker`, `ByteTrackTracker`, `OCSORTTracker`, `BoTSORTTracker` and `CBIoUTracker`. `lost_track_buffer=0` stays valid (no grace period) ([#420](https://github.com/roboflow/trackers/pull/420)).
- _Breaking:_ **confirmed tracks survive one additional missed frame** — all trackers changed from `time_since_update < maximum_frames_without_update` to `<=`, matching OC-SORT's earlier behavior. Expect small IDSW/HOTA shifts when comparing across this version ([#420](https://github.com/roboflow/trackers/pull/420)).
- **`McByteTracker`** — mask-conditioned ByteTrack with SAM box-mask generation and Cutie propagation, exported with `McByteMaskConfig` and CLI-discoverable. New `trackers[mask]` extra installs `torch`, `torchvision`, `rf-segment-anything` and `rf-cutie[inference]` (now from PyPI, not git). Device defaults to `"auto"` (CUDA → MPS → CPU). Cutie is tunable through `cutie_max_internal_size` (480), `cutie_mem_every` (10) and `cutie_use_long_term` (`True`); `cutie_channels_last`, `cutie_compile` and AMP are off by default; `minimum_mask_creation_frames` (3) defers SAM/Cutie encoding for short-lived tracklets. Frames are **RGB**, unlike the BGR the other trackers expect. Mask checkpoints load with `weights_only=True` (CWE-502 guard) ([#388](https://github.com/roboflow/trackers/pull/388), [#418](https://github.com/roboflow/trackers/pull/418), [#441](https://github.com/roboflow/trackers/pull/441), [#452](https://github.com/roboflow/trackers/pull/452), [#459](https://github.com/roboflow/trackers/pull/459), [#481](https://github.com/roboflow/trackers/pull/481), [#491](https://github.com/roboflow/trackers/pull/491), [#508](https://github.com/roboflow/trackers/pull/508), [#513](https://github.com/roboflow/trackers/pull/513), [#519](https://github.com/roboflow/trackers/pull/519), [#520](https://github.com/roboflow/trackers/pull/520), [#521](https://github.com/roboflow/trackers/pull/521), [#522](https://github.com/roboflow/trackers/pull/522), [#523](https://github.com/roboflow/trackers/pull/523), [#524](https://github.com/roboflow/trackers/pull/524), [#525](https://github.com/roboflow/trackers/pull/525), [#526](https://github.com/roboflow/trackers/pull/526), [#529](https://github.com/roboflow/trackers/pull/529), [#532](https://github.com/roboflow/trackers/pull/532)).
- **optional `timestamp=` on `BaseTracker.update()`** — all six trackers convert elapsed seconds into Kalman frame units and prune lost tracks on a seconds budget; without it, fixed-rate behavior is unchanged ([#446](https://github.com/roboflow/trackers/pull/446)).
- **bit-identical speedups** — ~38% lower BoT-SORT/C-BIoU predict cost from cached transition/noise matrices and deferred DWNA calibration, predicted boxes cached across association stages, fewer Kalman-estimator copies, tidier OC-SORT association ([#522](https://github.com/roboflow/trackers/pull/522), [#527](https://github.com/roboflow/trackers/pull/527), [#528](https://github.com/roboflow/trackers/pull/528)).
- _Fixed:_ **positive low-FPS `lost_track_buffer` values scale with `ceil(...)`** instead of collapsing to zero, and confirmed tracks survive exactly the scaled number of missed frames ([#420](https://github.com/roboflow/trackers/pull/420)).
- _Fixed:_ **BoT-SORT / C-BIoU first-frame tracks activated instantly are no longer dropped on a single miss** — sticky maturity triggers once a real `tracker_id` is assigned, as in ByteTrack, removing a default-config ID switch ([#478](https://github.com/roboflow/trackers/pull/478), [#504](https://github.com/roboflow/trackers/pull/504)).
- _Fixed:_ **ByteTrack / BoT-SORT / C-BIoU return unmatched detections between `high_conf_det_threshold` and `track_activation_threshold` with `tracker_id=-1`** instead of dropping them, matching the `update()` contract. Callers may see additional `tracker_id == -1` rows ([#475](https://github.com/roboflow/trackers/pull/475)).
- _Fixed:_ **signed IoU variants are clamped to `[0, 1]` before score fusion** — so CIoU's ~-1.5 floor no longer yields negative fused similarities; a no-op for GIoU/DIoU ([#476](https://github.com/roboflow/trackers/pull/476)).
- _Fixed:_ **OC-SORT Kalman scale stays positive across frame-step gaps** — `clamp_velocity` no longer lets it collapse negative over long timestamp gaps ([#509](https://github.com/roboflow/trackers/pull/509)).

### Motion & utilities

- **`KalmanMotionModel` in `trackers.utils.motion_models`** supplies the Kalman `F` and `Q` for a given `frame_step`: a constant-velocity `F` (`constant_velocity_F`) and `ScalableProcessNoise` for the tuned `Q`, DWNA-scaled on timestamp gaps ([#446](https://github.com/roboflow/trackers/pull/446)).
- **CMC sparse optical-flow status-mask filtering is vectorized** — ~24x faster for that operation, with bit-identical output ([#522](https://github.com/roboflow/trackers/pull/522)).
- _Fixed:_ **CMC handles mid-stream resolution changes** in optical-flow motion estimation with a re-sync/resize path ([#505](https://github.com/roboflow/trackers/pull/505)).
- _Fixed:_ **CMC no longer crashes in `cv2.resize` on tiny downscaled images** — dimensions clamp to `max(1, ...)`, covering 1x1, 1x3 and 3x1 ([#488](https://github.com/roboflow/trackers/pull/488)).
- _Fixed:_ **`xcycsr_to_xyxy` returns height `0` instead of `inf` on zero-aspect boxes** via an `r1 != 0` guard ([#485](https://github.com/roboflow/trackers/pull/485)).

### I/O & datasets

- _Fixed:_ **`VideoOutput` resizes frames to the writer's configured size** on mid-stream resolution change (`INTER_AREA` on downscale only) ([#514](https://github.com/roboflow/trackers/pull/514)).
- _Security:_ **ZIP extraction is hardened against Zip Slip / path traversal (CWE-22)** — archive members with absolute paths, traversal segments, empty names or Windows backslash paths are rejected, and validated archives cannot write outside the destination, including path-swap protection ([#495](https://github.com/roboflow/trackers/pull/495)).

### Packaging & docs

- **`pydeprecate` constraint raised from `>=0.7.0` to `>=0.8.0`** — build backend migrated from `hatchling` to `setuptools` with `src`-layout discovery and `py.typed` support ([#492](https://github.com/roboflow/trackers/pull/492)).
- _Fixed:_ **package builds for the `src` layout** — corrected `setuptools` discovery and `py.typed` support, plus release CI artifact handling and publish triggers ([#492](https://github.com/roboflow/trackers/pull/492)).

## [2.5.0] — 2026-06-24

### Trackers

- _Breaking:_ **internal tracklet ID counters are removed** — each tracker instance now allocates IDs. Internal `*Tracklet` subclassers should assign `tracklet.tracker_id` directly, using `self._allocate_tracker_id()` from `BaseTracker` in custom trackers ([#419](https://github.com/roboflow/trackers/pull/419)).
- _Deprecated:_ **`SORTTracker.trackers` is an alias for `.tracks`** — emits `FutureWarning` since 2.5 and is removed in 3.0. `SORTTracker` itself is not deprecated ([#460](https://github.com/roboflow/trackers/pull/460)).
- **pluggable IoU** — `iou=` on `SORTTracker`, `ByteTrackTracker`, `OCSORTTracker` and `BoTSORTTracker` accepts any `BaseIoU` subclass (`trackers.utils.iou`), with built-ins `IoU`, `GIoU`, `DIoU`, `CIoU` and `BIoU` (Buffered IoU). Subclass `BaseIoU` and override `_compute` for a custom metric. Signed variants override `normalize_for_fusion` to shift `[-1, 1]` to `[0, 1]` before BoT-SORT score fusion ([#403](https://github.com/roboflow/trackers/pull/403)).
- **`CBIoUTracker`** — Cascaded-Buffered IoU (Yang et al., WACV 2023): two-stage matching with independent `buffer_ratio_first` / `buffer_ratio_second` and BoT-SORT's ByteTrack-style low-confidence second pass ([#417](https://github.com/roboflow/trackers/pull/417)).
- _Fixed:_ **BoT-SORT score fusion multiplied raw negative GIoU/DIoU/CIoU values by confidence and inverted track ranking** — similarity is now normalized first ([#403](https://github.com/roboflow/trackers/pull/403)).
- _Fixed:_ **non-finite box coordinates no longer reach `linear_sum_assignment`** — `BaseIoU.compute` raises `ValueError` for NaN/inf ([#403](https://github.com/roboflow/trackers/pull/403)).
- _Fixed:_ **OC-SORT observation-centric recovery uses standard `IoU` per the paper** — independent of the configured `iou=` ([#403](https://github.com/roboflow/trackers/pull/403)).
- _Fixed:_ **the IoU helper uses lazy `np.divide(..., where=...)`** instead of eager `np.where`, suppressing `RuntimeWarning` on zero-area boxes under strict NumPy error settings ([#403](https://github.com/roboflow/trackers/pull/403)).
- _Fixed:_ **ByteTrack produced no tracks when detections lacked confidence** — the default fill is `np.ones`, as in SORT / OC-SORT / BoT-SORT ([#415](https://github.com/roboflow/trackers/pull/415)).
- _Fixed:_ **tracker instances no longer share ID counters** — so resetting one instance cannot reset another's allocator or duplicate live IDs in multi-camera, class-specific or parallel workflows ([#419](https://github.com/roboflow/trackers/pull/419)).

### Motion & utilities

- **`CMC`, `CMCConfig` and `CMCMethod` moved to `trackers.utils.cmc`** and are re-exported from `trackers` (`from trackers import CMC`). `BoTSORTTracklet.apply_cmc` delegates to `CMC.apply_batch`, so per-track and batch paths share code; behavior is unchanged ([#414](https://github.com/roboflow/trackers/pull/414)).
- **`CMC.apply_batch` raises `TypeError` immediately on a tracklet list with mixed state-estimator types** — preventing silent state corruption ([#414](https://github.com/roboflow/trackers/pull/414)).
- _Deprecated:_ **removed in 3.0** ([#414](https://github.com/roboflow/trackers/pull/414)):
    - `trackers.core.botsort.cmc` re-exports with `DeprecationWarning`; use `trackers.utils.cmc`.
    - `BoTSORTTracker.apply_cmc_batch`; use `CMC.apply_batch(H, tracker.tracks)`.
    - `CMCTMethod`; use `CMCMethod`.
    - `CMC.apply_to_xyxy`, renamed `CMC.warp_xyxy_corners` and kept as a forwarding wrapper.

### Evaluation

- _Fixed:_ **the HOTA per-frame alpha loop is vectorized** — so large evaluations run significantly faster with unchanged numeric output ([#462](https://github.com/roboflow/trackers/pull/462)).
- _Fixed:_ **ground-truth preprocessing applies distractor class filtering consistent with TrackEval** — correcting reported metrics on MOT17 and similar datasets ([#466](https://github.com/roboflow/trackers/pull/466)).

### Tuning

- **`Tuner` gains `enqueue_defaults`** (default `True`, evaluates a baseline trial from each param's `__init__` default before Optuna samples), `fixed_params` (held constant across trials), `images_dir` (frame loading for CMC-enabled trackers) and `seed` (reproducible TPE sampling) ([#427](https://github.com/roboflow/trackers/pull/427)).

### CLI

- _Fixed:_ **`iou=` is excluded from argparse auto-discovery** — so the `BaseIoU` parameter no longer crashes the CLI; set the variant programmatically ([#403](https://github.com/roboflow/trackers/pull/403)).

### Packaging & docs

- **`py.typed` marker (PEP 561)** — so IDEs and type checkers treat the package as typed without `--ignore-missing-imports` ([`f93c0b0a`](https://github.com/roboflow/trackers/commit/f93c0b0a)).

## [2.4.0] — 2026-05-07

### Trackers

- _Breaking:_ **`SORTTracker.update()` no longer mutates its input `sv.Detections`** and returns a fresh indexed copy, like ByteTrack and OC-SORT. Read `tracker_id` from the returned object ([#360](https://github.com/roboflow/trackers/pull/360)).
- _Breaking:_ **per-frame spawn order is deterministic across SORT, ByteTrack and OC-SORT** — IDs for same-frame spawns no longer depend on CPython set order. A recorded run reproduces across machines but may differ from a 2.3.0 baseline ([#361](https://github.com/roboflow/trackers/pull/361)).
- _Breaking:_ **internal tracklets (notably `OCSORTTracklet`) no longer accept `update(None)` for unmatched tracks** — missed-association logic lives in `predict()` and `_get_alive_tracklets`. Affects subclassers of internal `*Tracklet` classes only, not the public `update()` API ([#383](https://github.com/roboflow/trackers/pull/383), [#376](https://github.com/roboflow/trackers/pull/376)).
- **`BoTSORTTracker`** — camera motion compensation (CMC) with `orb`, `sift`, `sparseOptFlow` and `ecc`, and ByteTrack-style score-fused association ([#386](https://github.com/roboflow/trackers/pull/386)).
- **`BaseTracker.update(detections, frame=None)`**. Required by BoT-SORT when CMC is on; ignored with a `UserWarning` by SORT, ByteTrack and OC-SORT. The `track` CLI passes the frame automatically ([#386](https://github.com/roboflow/trackers/pull/386)).
- **`tracked_objects` on `BaseTracker` and all trackers** exposes every alive track with its Kalman-predicted box, including occluded or detector-missed tracks. The `update()` return value is unchanged ([#373](https://github.com/roboflow/trackers/pull/373), [#105](https://github.com/roboflow/trackers/issues/105)).
- **`TrackletProtocol` in `trackers.core.base`** — the structural contract for tracklets stored in `BaseTracker.tracks`, and a `search_space` ClassVar on every tracker consumed by `Tuner` and validated for unknown keys and bad types ([#373](https://github.com/roboflow/trackers/pull/373), [#301](https://github.com/roboflow/trackers/pull/301)).
- **one shared Kalman implementation backed by `BaseStateEstimator`** replaces per-tracklet filters; `SORTTracklet`, `ByteTrackTracklet`, `OCSORTTracklet` and `BoTSORTTracklet` handle association and lifecycle only. ByteTrack tracklets count `number_of_successful_consecutive_updates` instead of `number_of_updates`, as in the reference ([#310](https://github.com/roboflow/trackers/pull/310)).
- _Fixed:_ **ByteTrack prunes unmatched tracks after the Kalman refactor** — `time_since_update` advances and `_get_alive_tracklets` expires them after `lost_track_buffer` empty frames ([#376](https://github.com/roboflow/trackers/pull/376)).

### Motion & utilities

- **swappable Kalman state estimators** in `trackers.utils.state_representations`: `BaseStateEstimator`, `XCYCSRStateEstimator`, `XCYCWHStateEstimator` and `XYXYStateEstimator`, selected with `state_estimator_class=` ([#310](https://github.com/roboflow/trackers/pull/310)).
- **`xyxy_to_xywh` and `xywh_to_xyxy`** in `trackers.utils.converters` for center-width-height format ([#310](https://github.com/roboflow/trackers/pull/310), [#386](https://github.com/roboflow/trackers/pull/386)).

### Evaluation

- **`evaluate_mot_sequence` and `evaluate_mot_sequences` load lazily through `__getattr__`** to avoid circular imports ([#230](https://github.com/roboflow/trackers/pull/230)).

### Tuning

- **`trackers.tune.Tuner`** — Optuna-based hyperparameter optimisation over each tracker's `search_space`, with HOTA / MOTA / IDF1 objectives on MOT-format ground truth and precomputed detections ([#301](https://github.com/roboflow/trackers/pull/301)).
- **`trackers tune` CLI subcommand** to select tracker, ground-truth directory, detections directory, objective and `--n-trials` ([#374](https://github.com/roboflow/trackers/pull/374)).

### I/O & datasets

- **`load_mot_file` (was `_load_mot_file`) is public** in `trackers.io.mot` for custom tuning and evaluation scripts ([#301](https://github.com/roboflow/trackers/pull/301), [#374](https://github.com/roboflow/trackers/pull/374)).

### Packaging & docs

- **modern Python 3.10+ type hints** across the public surface ([#302](https://github.com/roboflow/trackers/pull/302)).
- **BoT-SORT user guide, state-estimator guide and a DanceTrack section** on the comparison page ([#386](https://github.com/roboflow/trackers/pull/386), [#310](https://github.com/roboflow/trackers/pull/310)).
- **landing page, install guide, evaluate guide, ByteTrack page and comparison page rewritten** — with DanceTrack tuned defaults added; documentation index ByteTrack entry corrected ([#371](https://github.com/roboflow/trackers/pull/371)).

## [2.3.0] — 2026-03-16

### Trackers

- **`OCSORTTracker`** with swappable state estimators (`XCYCSRStateEstimator`, `XYXYStateEstimator`), direction-consistency batch calculations and full tracklet lifecycle; registered in the CLI and public API ([#207](https://github.com/roboflow/trackers/pull/207)).
- **SORT and ByteTrack ship tuned DanceTrack defaults** ([#299](https://github.com/roboflow/trackers/pull/299)).

### Motion & utilities

- **`xcycsr_to_xyxy` and `xyxy_to_xcycsr` are restructured for the single-box case** — cutting per-frame overhead in tight tracking loops ([#296](https://github.com/roboflow/trackers/pull/296)).

### Evaluation

- **regression tests against TrackEval** for SORT, ByteTrack and OC-SORT on oracle detections from SportsMOT and DanceTrack, checking HOTA, MOTA, IDF1 and IDSW in CI ([#298](https://github.com/roboflow/trackers/pull/298)).
- _Fixed:_ **distractor filtering corrected** in the comparison numbers ([#322](https://github.com/roboflow/trackers/pull/322)).

### CLI

- **`trackers download`** fetches MOT17 and SportsMOT to a persistent cache (`~/.cache/trackers`) with MD5 verification and Rich progress output, backed by type-safe `Dataset`, `DatasetSplit` and `DatasetAsset` enums ([#262](https://github.com/roboflow/trackers/pull/262)).

### Packaging & docs

- **tracker comparison page with Default / Tuned tabs** and grid-search configs for SportsMOT, SoccerNet, MOT17 and DanceTrack as copyable YAML ([#309](https://github.com/roboflow/trackers/pull/309)).
- **landing page, install guide and evaluate pages rewritten** — comparison page gains dataset videos, paper links and a DanceTrack section ([#322](https://github.com/roboflow/trackers/pull/322)).
- **repository adopts a `release`/`stable` branching model** ([#275](https://github.com/roboflow/trackers/pull/275)).
- _Fixed:_ **`pypa/gh-action-pypi-publish` pinned to the real v1.13.0 commit SHA** ([#294](https://github.com/roboflow/trackers/pull/294)).

## [2.2.0] — 2026-02-18

### Trackers

- **`BaseTracker.__init_subclass__`** registers every subclass and extracts parameter metadata from `__init__` docstrings (`TrackerInfo`, `ParameterInfo`), so the CLI discovers trackers without a hard-coded list ([#230](https://github.com/roboflow/trackers/pull/230)).

### Motion & utilities

- **`MotionAwareTraceAnnotator`** applies homography-based CMC to keep trace paths stable on moving-camera footage ([#263](https://github.com/roboflow/trackers/pull/263)).

### Evaluation

- **HOTA, CLEAR (MOTA / MOTP / IDSW / MT / PT / ML) and Identity (IDF1 / IDP / IDR)** in `trackers.eval`, with public `evaluate_mot_sequence` and `evaluate_mot_sequences` ([#210](https://github.com/roboflow/trackers/pull/210), [#212](https://github.com/roboflow/trackers/pull/212), [#223](https://github.com/roboflow/trackers/pull/223), [#224](https://github.com/roboflow/trackers/pull/224), [#226](https://github.com/roboflow/trackers/pull/226)).

### CLI

- **`trackers eval`** runs a tracker over a MOT-format ground-truth directory and prints HOTA / MOTA / IDF1, with tracker arguments given as JSON ([#215](https://github.com/roboflow/trackers/pull/215)).

### I/O & datasets

- **`load_mot_file` and `save_mot_file`** in `trackers.io.mot` read and write MOT-format `.txt` annotations ([#214](https://github.com/roboflow/trackers/pull/214)).

### Packaging & docs

- **SORT and ByteTrack HOTA / MOTA / IDF1 benchmarks** on the docs site, and runnable Colab notebooks linked from the index ([#193](https://github.com/roboflow/trackers/pull/193), [#199](https://github.com/roboflow/trackers/pull/199)).
- **Apache 2.0 license headers** on all source files via a pre-commit hook ([#282](https://github.com/roboflow/trackers/pull/282)).
- **the `bandit` hook is replaced by Ruff's built-in `S` rule set** ([#188](https://github.com/roboflow/trackers/pull/188)).
- **unused optional extras are removed** — shrinking the install footprint ([#192](https://github.com/roboflow/trackers/pull/192)).

## [2.1.0] — 2026-01-28

### Trackers

- _Breaking:_ **`DeepSortTracker` and all REID infrastructure are removed** from the package, docs, tests and CI. Pin `<2.1.0` to keep using DeepSort ([#161](https://github.com/roboflow/trackers/pull/161)).
- **`ByteTrackTracker`** — two-stage low-score / high-score association with a ByteTrack-specific Kalman box tracker, full API docs and unit tests, registered in the CLI and public API ([#174](https://github.com/roboflow/trackers/pull/174)).

### Packaging & docs

- _Breaking:_ **Python 3.9 is dropped** — the minimum is 3.10 and the public API uses built-in generics (`list[...]`, `dict[...]`) ([#200](https://github.com/roboflow/trackers/pull/200)).
- _Fixed:_ **documentation build pins `mkdocstrings-python<2.0.0`** to fix docs generation ([`0dd757ec`](https://github.com/roboflow/trackers/commit/0dd757ec)).

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
