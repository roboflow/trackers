#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""optimize_tracking.py — Optuna-based tracker search on MOT17-val.

Usage:
    python optimize_tracking.py bytetrack frcnn         # ByteTrack, FRCNN dets
    python optimize_tracking.py sort frcnn              # SORT baseline
    python optimize_tracking.py ocsort sdp              # OC-SORT, SDP dets
    python optimize_tracking.py bytetrack yolox         # ByteTrack, YOLOX dets
    python optimize_tracking.py bytetrack frcnn --n-trials 200   # Full Optuna study
    python optimize_tracking.py bytetrack yolox --n-trials 50 --fast  # Quick study
    python optimize_tracking.py bytetrack mydet                   # Custom detector

Supported trackers: sort | bytetrack | ocsort
Supported det-sources (MOT17 bundled detectors, no setup needed):
  frcnn   — FRCNN bundled detections from MOT17-{N}-FRCNN/det/det.txt
  sdp     — SDP bundled detections from MOT17-{N}-SDP/det/det.txt
  dpm     — DPM bundled detections from MOT17-{N}-DPM/det/det.txt

Supported det-sources (generated via generate_detections.py):
  yolox   — YOLOX-X CrowdHuman detections from MOT17-{N}-YOLOX/det/det.txt
  rfdetr  — RF-DETR-L detections from MOT17-{N}-RFDETR/det/det.txt

Custom detectors:
  Pass any det_source name not in the list above (e.g. "mydet") — it is
  uppercased automatically to form the directory tag (MOT17-{N}-MYDET/).
  Create the sibling dirs with det/det.txt and gt/, img1/ symlinks.

The search space and tracker construction are updated by the agent as tracker
architectures evolve — add, remove, or retune parameters to match the current code.

Hard boundaries (never change):
  - Evaluation calls go through trackers.eval — do not substitute custom metric code.
  - Ground truth: always gt/gt.txt — never read at inference time.
  - Detections: {seq}-{TAG}/det/det.txt only — never gt/. No ground truth at inference.
  - Output format: __METRICS__ line must remain parseable by the campaign loop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import fire
import numpy as np
import optuna
import supervision as sv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_err = Console(stderr=True)

_STUDY_NAME = "autotrack"

# Maps --det-source values to the sequence directory suffix used in the filesystem.
# generate_detections.py creates MOT17-{N}-{TAG}/ sibling dirs; FRCNN is the original.
_DET_SOURCE_TO_TAG: dict[str, str] = {
    "frcnn": "FRCNN",
    "sdp": "SDP",
    "dpm": "DPM",
    "yolox": "YOLOX",
    "rfdetr": "RFDETR",
}

# Capture the original ByteTrack Kalman init once so repeated patching across
# trials always re-applies from the true original (not a previously patched version).
_ORIG_KALMAN_INIT = None


# ---------------------------------------------------------------------------
# DEFAULTS — loaded from default_config.json; edit that file, not this script
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, dict] = {
    k: v
    for k, v in json.loads(
        (Path(__file__).parent / "default_config.json").read_text()
    ).items()
    if not k.startswith("_")  # strip _comment / _edit_here meta-keys
}


# ---------------------------------------------------------------------------
# SEARCH SPACE — loaded from search_space.json; edit that file, not this script
# ---------------------------------------------------------------------------

_SEARCH_SPACE: dict[str, dict] = {
    k: v
    for k, v in json.loads(
        (Path(__file__).parent / "search_space.json").read_text()
    ).items()
    if not k.startswith("_")  # strip _comment / _edit_here / _types meta-keys
}


def _define_search_space(trial: optuna.Trial, tracker_name: str) -> dict:
    """Sample a parameter dict for this trial from the search_space.json definition.

    To add, remove, or retune a parameter: edit search_space.json — do not
    modify this function.  Supported spec keys per parameter:
      type        "int" | "float" | "categorical"
      low / high  numeric bounds (int and float)
      log         true → log-scale sampling (float only)
      choices     list of values (categorical only)
    """
    space = _SEARCH_SPACE[tracker_name]
    params: dict = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search space type {kind!r} for param {name!r}")
    return params


def _build_tracker(params: dict, tracker_name: str):
    """Instantiate the named tracker from a parameter dict.

    Update the relevant constructor call when new parameters are added or removed.
    """
    if tracker_name == "bytetrack":
        from trackers import ByteTrackTracker

        return ByteTrackTracker(
            lost_track_buffer=params["lost_track_buffer"],
            track_activation_threshold=params["track_activation_threshold"],
            minimum_consecutive_frames=params["minimum_consecutive_frames"],
            minimum_iou_threshold=params["minimum_iou_threshold"],
            high_conf_det_threshold=params["high_conf_det_threshold"],
        )
    if tracker_name == "sort":
        from trackers import SORTTracker

        return SORTTracker(
            lost_track_buffer=params["lost_track_buffer"],
            track_activation_threshold=params["track_activation_threshold"],
            minimum_consecutive_frames=params["minimum_consecutive_frames"],
            minimum_iou_threshold=params["minimum_iou_threshold"],
        )
    if tracker_name == "ocsort":
        from trackers import OCSORTTracker

        return OCSORTTracker(
            lost_track_buffer=params["lost_track_buffer"],
            minimum_consecutive_frames=params["minimum_consecutive_frames"],
            minimum_iou_threshold=params["minimum_iou_threshold"],
            direction_consistency_weight=params["direction_consistency_weight"],
            high_conf_det_threshold=params["high_conf_det_threshold"],
            delta_t=params["delta_t"],
        )
    raise ValueError(
        f"Unknown tracker: {tracker_name!r}. Choose: sort | bytetrack | ocsort"
    )


def _apply_kalman_patch(params: dict, tracker_name: str) -> None:
    """Override ByteTrack Kalman noise matrices from params. No-op for other trackers.

    Remove or replace this function if Kalman scales become constructor args,
    or if the Kalman architecture changes to a point where simple scalar scaling
    no longer makes sense.
    """
    if tracker_name != "bytetrack":
        return

    global _ORIG_KALMAN_INIT
    from trackers.core.bytetrack.kalman import ByteTrackKalmanBoxTracker

    if _ORIG_KALMAN_INIT is None:
        _ORIG_KALMAN_INIT = ByteTrackKalmanBoxTracker._initialize_kalman_filter

    q = params.get("q_scale", 0.01)
    r = params.get("r_scale", 0.1)
    p = params.get("p_scale", 1.0)
    vel_decay = params.get("velocity_decay", 0.95)
    q_miss = params.get("q_miss_alpha", 0.1)

    orig = _ORIG_KALMAN_INIT

    def _patched(self: ByteTrackKalmanBoxTracker) -> None:
        orig(self)
        self.Q = np.eye(self.Q.shape[0], dtype=np.float32) * q
        self.R = np.eye(self.R.shape[0], dtype=np.float32) * r
        self.P = np.eye(self.P.shape[0], dtype=np.float32) * p

    setattr(ByteTrackKalmanBoxTracker, "_initialize_kalman_filter", _patched)
    ByteTrackKalmanBoxTracker.velocity_decay = vel_decay
    ByteTrackKalmanBoxTracker.q_miss_alpha = q_miss
    ByteTrackKalmanBoxTracker.p_reset_threshold = params.get("p_reset_threshold", 5)


# ---------------------------------------------------------------------------
# EVALUATION — do not modify (metrics + dataset integrity)
# ---------------------------------------------------------------------------


def _find_data_dir() -> Path:
    for candidate in [
        Path("./mot17/val"),
        Path("./data/mot17/val"),
        Path.home() / ".cache/trackers/mot17/val",
    ]:
        if candidate.exists() and any(candidate.glob("*/gt/gt.txt")):
            return candidate
    raise FileNotFoundError(
        "MOT17 val data not found. Run:\n"
        "  trackers download mot17 --split val --asset annotations,detections"
    )


def _run_tracker_on_sequence(
    tracker, det_file: Path, output_file: Path, max_interpolation_gap: int = 0
) -> None:
    from trackers.core.sort.utils import interpolate_mot_gaps
    from trackers.io.mot import _load_mot_file, _mot_frame_to_detections

    # reset() clears track list and resets the ID counter for all tracker types
    tracker.reset()

    detections_data = _load_mot_file(det_file)
    if not detections_data:
        return

    max_frame = max(detections_data.keys())
    lines = []
    for frame_idx in range(1, max_frame + 1):
        dets = (
            _mot_frame_to_detections(detections_data[frame_idx])
            if frame_idx in detections_data
            else sv.Detections.empty()
        )
        tracked = tracker.update(dets)
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                if tid < 0:
                    continue
                x1, y1, x2, y2 = tracked.xyxy[i]
                w, h = x2 - x1, y2 - y1
                conf = (
                    float(tracked.confidence[i])
                    if tracked.confidence is not None
                    else 1.0
                )
                lines.append(
                    f"{frame_idx},{tid + 1},{x1:.2f},{y1:.2f},"
                    f"{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"
                )

    if max_interpolation_gap > 0:
        lines = interpolate_mot_gaps(lines, max_gap=max_interpolation_gap)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n" if lines else "")


def _ensure_gt_symlink(data_dir: Path, seq: str) -> None:
    """Create a gt/ symlink inside seq's directory if it is missing.

    Bundled SDP and DPM sequence directories do not include a gt/ subdirectory;
    ground truth lives only in the corresponding FRCNN sibling (which is always
    downloaded alongside the other detectors).  This mirrors what
    ``generate_detections.py`` does for YOLOX/RFDETR directories.

    Args:
        data_dir: MOT17-val root directory.
        seq: Sequence directory name, e.g. ``"MOT17-02-SDP"``.
    """
    gt_dir = data_dir / seq / "gt"
    if gt_dir.exists():
        return
    # Strip the detector suffix to find the FRCNN sibling
    base = seq.rsplit("-", 1)[0]  # "MOT17-02-SDP" → "MOT17-02"
    frcnn_gt = data_dir / f"{base}-FRCNN" / "gt"
    if frcnn_gt.exists():
        gt_dir.symlink_to(f"../{base}-FRCNN/gt")


def _run_eval(
    params: dict,
    sequences: list[str],
    data_dir: Path,
    tracker_name: str,
    det_source: str = "frcnn",
    show_progress: bool = False,
) -> dict:
    """Evaluate tracker on sequences and return metric dict."""
    for seq in sequences:
        _ensure_gt_symlink(data_dir, seq)
    import tempfile

    from trackers.eval import evaluate_mot_sequence, evaluate_mot_sequences

    with tempfile.TemporaryDirectory() as _tmp:
        output_dir = Path(_tmp)

        max_interp = params.get("max_interpolation_gap", 0)
        tracker = _build_tracker(params, tracker_name)

        def _run_seq(seq: str) -> None:
            det_file = data_dir / seq / "det" / "det.txt"
            if det_file.exists():
                _run_tracker_on_sequence(
                    tracker,
                    det_file,
                    output_dir / f"{seq}.txt",
                    max_interpolation_gap=max_interp,
                )

        if show_progress:
            with Progress(
                TextColumn("  {task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                console=_err,
                transient=True,
            ) as prog:
                task_id = prog.add_task("", total=len(sequences))
                for seq in sequences:
                    prog.update(task_id, description=seq)
                    _run_seq(seq)
                    prog.advance(task_id)
        else:
            for seq in sequences:
                _run_seq(seq)

        metrics_list = ["HOTA", "CLEAR", "Identity"]
        try:
            if len(sequences) == 1:
                agg = evaluate_mot_sequence(
                    gt_path=data_dir / sequences[0] / "gt" / "gt.txt",
                    tracker_path=output_dir / f"{sequences[0]}.txt",
                    metrics=metrics_list,
                    threshold=0.5,
                )
            else:
                seqmap = output_dir / "seqmap.txt"
                seqmap.write_text("\n".join(sequences) + "\n")
                agg = evaluate_mot_sequences(
                    gt_dir=data_dir,
                    tracker_dir=output_dir,
                    seqmap=seqmap,
                    metrics=metrics_list,
                    threshold=0.5,
                ).aggregate
        except ValueError as exc:
            # Tracker produced zero tracks (e.g. threshold too aggressive) →
            # empty output file → evaluator raises ValueError.  Return worst-case
            # metrics so Optuna records the trial instead of logging it as failed.
            if "MOT file is empty" in str(exc):
                return {"HOTA": 0.0, "IDF1": 0.0, "MOTA": -100.0, "IDSW": 999999}
            raise

    if not agg.HOTA:
        raise RuntimeError(
            "Evaluation returned no HOTA results — check sequences and gt paths"
        )
    if not agg.Identity:
        raise RuntimeError("Evaluation returned no Identity results")
    if not agg.CLEAR:
        raise RuntimeError("Evaluation returned no CLEAR results")
    return {
        "HOTA": agg.HOTA.HOTA * 100,
        "IDF1": agg.Identity.IDF1 * 100,
        "MOTA": agg.CLEAR.MOTA * 100,
        "IDSW": int(agg.CLEAR.IDSW),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _mp_worker(
    storage_url: str,
    n: int,
    sequences: list[str],
    data_dir_str: str,
    tracker_name: str,
    det_source: str = "frcnn",
) -> None:
    """Multiprocessing worker: loads shared Optuna study and runs N trials.

    Runs in a separate process — module-level state (including the class-level
    Kalman patch) is fully isolated, so concurrent workers never race.

    Args:
        storage_url: SQLAlchemy URL for the shared SQLite study database.
        n: Number of trials this worker should run.
        sequences: MOT17-val sequence names to evaluate.
        data_dir_str: Stringified path to the MOT17-val root (pickling-safe).
        tracker_name: Which tracker to evaluate (sort | bytetrack | ocsort).
        det_source: Detection source — "frcnn", "sdp", "dpm", "yolo", or "yolox".
    """
    _data_dir = Path(data_dir_str)
    _study = optuna.load_study(study_name=_STUDY_NAME, storage=storage_url)

    def _obj(trial: optuna.Trial) -> float:
        params = _define_search_space(trial, tracker_name)
        _apply_kalman_patch(params, tracker_name)
        return _run_eval(
            params=params,
            sequences=sequences,
            data_dir=_data_dir,
            tracker_name=tracker_name,
            det_source=det_source,
        )["HOTA"]

    _study.optimize(_obj, n_trials=n, show_progress_bar=False)


def _validate_args(tracker: str, det_source: str) -> None:
    """Raise ValueError if tracker is unrecognised.

    Any ``det_source`` value is accepted — known sources (frcnn, sdp, dpm, yolox,
    rfdetr) map to their canonical tags via ``_DET_SOURCE_TO_TAG``; unknown values
    are uppercased automatically, enabling custom detectors without any extra flags.
    """
    if tracker not in _DEFAULTS:
        raise ValueError(
            f"Unknown tracker: {tracker!r}. Choose: {' | '.join(_DEFAULTS)}"
        )


def _resolve_sequences(
    data_dir: str | None,
    fast: bool,
    det_source: str,
) -> tuple[Path, list[str]]:
    """Locate MOT17 dir and enumerate sequences for the requested detector.

    Sequences are discovered by the directory suffix that matches the detector tag
    (e.g. ``-FRCNN`` for frcnn, ``-YOLOX`` for yolox).  Each sequence directory must
    have ``det/det.txt``; ground truth is found via ``gt/gt.txt`` (which may be a
    symlink created by ``generate_detections.py``).

    Args:
        data_dir: Explicit MOT17-val path, or None to auto-detect.
        fast: If True, restrict to a single sequence for quick checks.
        det_source: Detection source.  Known values are looked up in
            ``_DET_SOURCE_TO_TAG``; any other value is uppercased to form the tag,
            so ``"mydet"`` resolves to ``MOT17-{N}-MYDET/`` without extra config.

    Returns:
        Tuple of (data_dir Path, sequence name list).
    """
    _data_dir = Path(data_dir) if data_dir else _find_data_dir()
    tag = _DET_SOURCE_TO_TAG.get(det_source, det_source.upper())
    sequences = sorted(
        d.name
        for d in _data_dir.iterdir()
        if d.is_dir()
        and d.name.endswith(f"-{tag}")
        and (d / "det" / "det.txt").exists()
    )
    if not sequences:
        hint = f"  uv run python generate_detections.py --detector-tag {tag}"
        raise FileNotFoundError(
            f"No MOT17-*-{tag}/ sequences found in {_data_dir}.\n"
            f"Generate detections with:\n{hint}"
        )
    if fast:
        fast_seq = [s for s in sequences if "MOT17-04" in s]
        sequences = fast_seq[:1] if fast_seq else sequences[:1]

    return _data_dir, sequences


def _load_warm_start(
    best_config_path: Path,
    tracker: str,
    det_source: str,
    n_trials: int,
) -> dict | None:
    """Load the previous best config as an Optuna warm-start point.

    Args:
        best_config_path: Path to ``best_config.json``.
        tracker: Tracker name (sort | bytetrack | ocsort).
        det_source: Detection source (frcnn | sdp | dpm | yolo | yolox).
        n_trials: Number of trials; warm-start is skipped when n_trials == 1.

    Returns:
        Param dict to enqueue as the first trial, or None if unavailable.
    """
    if n_trials == 1 or not best_config_path.exists():
        return None
    try:
        cfg = json.loads(best_config_path.read_text())
        entry = cfg.get(tracker, {}).get(det_source, {})
        warm = entry.get("config")
        if warm is None:
            _err.print(
                f"[yellow]warn[/yellow] no prior best for {tracker}/{det_source}"
                " in best_config.json — warm-start skipped, starting from defaults"
            )
        return warm
    except json.JSONDecodeError as exc:
        _err.print(
            f"[yellow]warn[/yellow] best_config.json is not valid JSON ({exc})"
            " — warm-start skipped, starting from defaults"
        )
        return None


def _run_optuna_study(
    n_trials: int,
    n_jobs: int,
    sequences: list[str],
    data_dir: Path,
    tracker: str,
    det_source: str,
    defaults: dict,
    warm: dict | None,
) -> tuple[optuna.Study, int]:
    """Create and run an Optuna study, using multiprocessing when n_trials > 1.

    Args:
        n_trials: Total number of trials to run.
        n_jobs: Worker count; -1 uses all CPUs, 1 disables multiprocessing.
        sequences: MOT17-val sequence names to evaluate per trial.
        data_dir: Root directory containing the sequences.
        tracker: Tracker name (sort | bytetrack | ocsort).
        det_source: Detection source (frcnn | sdp | dpm | yolo | yolox).
        defaults: Default parameter dict (from ``default_config.json``) to enqueue.
        warm: Prior best-config dict to warm-start from, or None.

    Returns:
        Tuple of (completed Optuna Study, number of workers used).
    """
    import multiprocessing
    import os
    import tempfile

    cpu_count = os.cpu_count() or 1
    n_workers = (
        1
        if n_trials == 1
        else min(n_trials, cpu_count if n_jobs == -1 else max(1, n_jobs))
    )

    if n_workers > 1:
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        storage_url = f"sqlite:///{db_path}"
        try:
            study = optuna.create_study(
                study_name=_STUDY_NAME,
                storage=storage_url,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.enqueue_trial(warm or defaults)
            base, rem = divmod(n_trials, n_workers)
            counts = [base + (1 if i < rem else 0) for i in range(n_workers)]
            worker_args = [
                (storage_url, c, sequences, str(data_dir), tracker, det_source)
                for c in counts
            ]
            print(f"[→ {n_workers} workers · {n_trials} trials · {cpu_count} cores]")
            with multiprocessing.Pool(n_workers) as pool:
                result = pool.starmap_async(_mp_worker, worker_args)
                with Progress(
                    TextColumn("  {task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeRemainingColumn(),
                    console=_err,
                ) as prog:
                    _prefix = f"{tracker} | {det_source}"
                    tid = prog.add_task(f"{_prefix} | HOTA=?", total=n_trials)
                    while not result.ready():
                        _poll = optuna.load_study(
                            study_name=_STUDY_NAME, storage=storage_url
                        )
                        done = sum(
                            1
                            for t in _poll.trials
                            if t.state == optuna.trial.TrialState.COMPLETE
                        )
                        best = _poll.best_value if done > 0 else 0.0
                        prog.update(
                            tid,
                            completed=done,
                            description=f"{_prefix} | HOTA={best:.3f}",
                        )
                        result.wait(timeout=2)
                    prog.update(tid, completed=n_trials)
                result.get()  # re-raise any worker exceptions
            study = optuna.load_study(study_name=_STUDY_NAME, storage=storage_url)
        finally:
            Path(db_path).unlink(missing_ok=True)
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.enqueue_trial(defaults if n_trials == 1 else (warm or defaults))

        def objective(trial: optuna.Trial) -> float:
            params = _define_search_space(trial, tracker)
            _apply_kalman_patch(params, tracker)
            return _run_eval(
                params=params,
                sequences=sequences,
                data_dir=data_dir,
                tracker_name=tracker,
                det_source=det_source,
            )["HOTA"]

        callbacks: list = []
        _trial_prog: Progress | None = None
        if n_trials > 1:
            _trial_prog = Progress(
                TextColumn("  {task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=_err,
            )
            _prefix = f"{tracker} | {det_source}"
            _tid = _trial_prog.add_task(f"{_prefix} | HOTA=?", total=n_trials)
            _trial_prog.start()

            def _trial_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                best = study.best_value or 0.0
                _trial_prog.update(  # type: ignore[union-attr]
                    _tid, advance=1, description=f"{_prefix} | HOTA={best:.3f}"
                )

            callbacks = [_trial_cb]

        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=False,
                callbacks=callbacks,
            )
        finally:
            if _trial_prog is not None:
                _trial_prog.stop()

    return study, n_workers


def _save_best_if_improved(
    best_config_path: Path,
    tracker: str,
    det_source: str,
    best_metrics: dict,
    best_params: dict,
    n_trials: int,
) -> None:
    """Persist best_params to best_config.json if HOTA improved over the previous run.

    The file is keyed as ``{tracker: {det_source: {hota, config}}}``, matching the
    structure of ``defaults.json`` and ``search_space.json``.

    Args:
        best_config_path: Path to ``best_config.json``.
        tracker: Tracker name (sort | bytetrack | ocsort).
        det_source: Detection source (frcnn | sdp | dpm | yolo | yolox).
        best_metrics: Metric dict from the current run.
        best_params: Corresponding Optuna best params.
        n_trials: Saving is skipped for single-trial baseline evals.
    """
    if n_trials <= 1:
        return
    cfg: dict = {}
    if best_config_path.exists():
        try:
            cfg = json.loads(best_config_path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            _err.print(
                f"[yellow]warn[/yellow] could not read best_config.json ({exc})"
                " — will overwrite with current result if it improves on 0.0"
            )
    prev_hota = float(cfg.get(tracker, {}).get(det_source, {}).get("hota", 0.0))
    if best_metrics["HOTA"] > prev_hota:
        cfg.setdefault(tracker, {})[det_source] = {
            "hota": best_metrics["HOTA"],
            "config": best_params,
        }
        best_config_path.write_text(json.dumps(cfg, indent=4))


def _print_metrics(
    best_metrics: dict,
    best_params: dict,
    elapsed: float,
    n_trials: int,
    n_workers: int,
    tracker: str,
    det_source: str,
    sequences: list[str],
) -> None:
    """Print __METRICS__, __CONFIG__, __ELAPSED__, __TRACKER__ lines to stdout."""
    hota = best_metrics["HOTA"]
    idf1 = best_metrics["IDF1"]
    mota = best_metrics["MOTA"]
    idsw = best_metrics["IDSW"]
    print(f"\n__METRICS__: HOTA={hota:.3f} IDF1={idf1:.3f} MOTA={mota:.3f} IDSW={idsw}")
    print(f"__CONFIG__: {json.dumps(best_params)}")
    print(
        f"__ELAPSED__: {elapsed:.1f}s  __TRIALS__: {n_trials}  __WORKERS__: {n_workers}"
    )
    print(
        f"__TRACKER__: {tracker}  __DET_SOURCE__: {det_source}"
        f"  __SEQUENCES__: {','.join(sequences)}"
    )


def main(
    tracker: str,
    det_source: str,
    n_trials: int = 1,
    fast: bool = False,
    data_dir: str | None = None,
    n_jobs: int = -1,
) -> None:
    """Run Optuna-based tracker search on MOT17-val.

    Args:
        tracker: Which tracker to evaluate. One of: sort | bytetrack | ocsort.
        det_source: Detection source — maps to MOT17-{N}-{TAG}/det/det.txt.
            Bundled (no setup): frcnn, sdp, dpm.
            Generated (run generate_detections.py first): yolox, rfdetr.
            Any other value is uppercased and used directly as the directory tag,
            e.g. ``mydet`` → ``MOT17-{N}-MYDET/``.
        n_trials: Number of Optuna trials. 1 evaluates default params (campaign metric).
        fast: Single sequence only (~3x faster, for development checks).
        data_dir: MOT17 val directory. Auto-detected from standard cache paths if unset.
        n_jobs: Worker processes for parallel trials. -1 uses all CPU cores. 1 disables
            multiprocessing. Ignored when n_trials=1 (single eval needs no parallelism).
    """
    _validate_args(tracker, det_source)

    t0 = time.time()
    _data_dir, sequences = _resolve_sequences(data_dir, fast, det_source)

    best_config_path = Path(__file__).parent / "best_config.json"
    warm = _load_warm_start(best_config_path, tracker, det_source, n_trials)

    study, n_workers = _run_optuna_study(
        n_trials=n_trials,
        n_jobs=n_jobs,
        sequences=sequences,
        data_dir=_data_dir,
        tracker=tracker,
        det_source=det_source,
        defaults=_DEFAULTS[tracker],
        warm=warm,
    )

    best_params = study.best_params
    best_metrics = _run_eval(
        params=best_params,
        sequences=sequences,
        data_dir=_data_dir,
        tracker_name=tracker,
        det_source=det_source,
        show_progress=True,
    )
    elapsed = time.time() - t0

    _save_best_if_improved(
        best_config_path=best_config_path,
        tracker=tracker,
        det_source=det_source,
        best_metrics=best_metrics,
        best_params=best_params,
        n_trials=n_trials,
    )
    _print_metrics(
        best_metrics=best_metrics,
        best_params=best_params,
        elapsed=elapsed,
        n_trials=n_trials,
        n_workers=n_workers,
        tracker=tracker,
        det_source=det_source,
        sequences=sequences,
    )


if __name__ == "__main__":
    fire.Fire(main)
