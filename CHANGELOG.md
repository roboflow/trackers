# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
