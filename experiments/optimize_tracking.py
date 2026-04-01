#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""optimize_tracking.py — Optuna-based ByteTrack search on MOT17-val.

Usage:
    python optimize_tracking.py                # 1 trial — evaluate default params
    python optimize_tracking.py --n-trials 200 # full Optuna study
    python optimize_tracking.py --n-trials 50 --fast  # single-sequence quick study

The search space and tracker construction (below the "SEARCH SPACE" comment) are
updated by the agent as the tracker architecture evolves — add, remove, or retune
parameters to match the current code.

Hard boundaries (never change):
  - Evaluation calls go through trackers.eval — do not substitute custom metric code.
  - Data source: MOT17-val, FRCNN public detections from det/det.txt only.
    Never read from gt/ at inference time.
  - Output format: __METRICS__ line must remain parseable by the campaign loop.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import fire
import numpy as np
import optuna
import supervision as sv

optuna.logging.set_verbosity(optuna.logging.WARNING)

_STUDY_NAME = "autotrack"

# Capture the original Kalman init once so repeated patching across trials
# always re-applies from the true original (not a previously patched version).
# Each worker process gets its own copy — class-level patching is safe across processes.
_ORIG_KALMAN_INIT = None


# ---------------------------------------------------------------------------
# SEARCH SPACE — update when tracker architecture changes
# ---------------------------------------------------------------------------


def _define_search_space(trial: optuna.Trial) -> dict:
    """Return a parameter dict sampled by Optuna for this trial.

    Edit this function when:
    - Adding a new constructor parameter to the tracker (add suggest_* call)
    - Removing a parameter (delete the line)
    - Changing a search range after architectural changes
    - Adding new tunable components (e.g., new Kalman matrices, association weights)
    """
    return {
        # ByteTrack constructor params
        "lost_track_buffer": trial.suggest_int("lost_track_buffer", 10, 80),
        "track_activation_threshold": trial.suggest_float(
            "track_activation_threshold", 0.3, 0.9
        ),
        "minimum_consecutive_frames": trial.suggest_int(
            "minimum_consecutive_frames", 1, 5
        ),
        "minimum_iou_threshold": trial.suggest_float(
            "minimum_iou_threshold", 0.05, 0.3
        ),
        "high_conf_det_threshold": trial.suggest_float(
            "high_conf_det_threshold", 0.3, 0.7
        ),
        # Kalman noise scales (applied via patch; remove if integrated into constructor)
        "q_scale": trial.suggest_float("q_scale", 0.001, 0.1, log=True),
        "r_scale": trial.suggest_float("r_scale", 0.01, 1.0, log=True),
        "p_scale": trial.suggest_float("p_scale", 0.1, 10.0, log=True),
        # Velocity decay per lost frame (1.0 = no decay, 0.8 = aggressive)
        "velocity_decay": trial.suggest_float("velocity_decay", 0.80, 1.0),
        # Q inflation rate for lost tracks: Q_eff = Q * (1 + alpha * t_since_update)
        "q_miss_alpha": trial.suggest_float("q_miss_alpha", 0.0, 0.5),
        # Post-processing: fill gaps <= N frames via linear bbox interpolation.
        # Improves AssA by making fragmented tracks continuous.  0 = disabled.
        "max_interpolation_gap": trial.suggest_int("max_interpolation_gap", 0, 30),
        # Reset P to identity after re-detection following >= N lost frames.
        # Clears stale cross-covariances accumulated during Q-inflated occlusion.
        # 0 = disabled.
        "p_reset_threshold": trial.suggest_int("p_reset_threshold", 0, 15),
    }


def _build_tracker(params: dict):
    """Instantiate ByteTrackTracker from a parameter dict.

    Update the constructor call when new parameters are added or removed.
    The signature of this function stays stable — only the body changes.
    """
    from trackers import ByteTrackTracker

    return ByteTrackTracker(
        lost_track_buffer=params["lost_track_buffer"],
        track_activation_threshold=params["track_activation_threshold"],
        minimum_consecutive_frames=params["minimum_consecutive_frames"],
        minimum_iou_threshold=params["minimum_iou_threshold"],
        high_conf_det_threshold=params["high_conf_det_threshold"],
    )


def _apply_kalman_patch(params: dict) -> None:
    """Override Kalman noise matrices from params.

    Remove or replace this function if Kalman scales become constructor args,
    or if the Kalman architecture changes to a point where simple scalar scaling
    no longer makes sense.
    """
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
    # Set velocity decay as class attribute so all new instances pick it up
    ByteTrackKalmanBoxTracker.velocity_decay = vel_decay
    # Set Q inflation rate for lost tracks
    ByteTrackKalmanBoxTracker.q_miss_alpha = q_miss
    # Set P reset threshold for re-detection after long occlusion
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
    print(
        "MOT17 val data not found. Run:\n"
        "  trackers download mot17 --split val --asset annotations,detections",
        file=sys.stderr,
    )
    sys.exit(1)


def _run_tracker_on_sequence(
    tracker, det_file: Path, output_file: Path, max_interpolation_gap: int = 0
) -> None:
    from trackers.core.bytetrack.kalman import ByteTrackKalmanBoxTracker
    from trackers.core.sort.utils import interpolate_mot_gaps
    from trackers.io.mot import _load_mot_file, _mot_frame_to_detections

    tracker.reset()
    ByteTrackKalmanBoxTracker.count_id = 0

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


def _run_eval(params: dict, sequences: list[str], data_dir: Path) -> dict:
    """Evaluate tracker on sequences and return metric dict."""
    import tempfile

    from trackers.eval import evaluate_mot_sequence, evaluate_mot_sequences

    with tempfile.TemporaryDirectory() as _tmp:
        output_dir = Path(_tmp)

        max_interp = params.get("max_interpolation_gap", 0)
        tracker = _build_tracker(params)
        for seq in sequences:
            det_file = data_dir / seq / "det" / "det.txt"
            if det_file.exists():
                _run_tracker_on_sequence(
                    tracker,
                    det_file,
                    output_dir / f"{seq}.txt",
                    max_interpolation_gap=max_interp,
                )

        metrics_list = ["HOTA", "CLEAR", "Identity"]
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

    return {
        "HOTA": agg.HOTA.HOTA * 100 if agg.HOTA else 0.0,
        "IDF1": agg.Identity.IDF1 * 100 if agg.Identity else 0.0,
        "MOTA": agg.CLEAR.MOTA * 100 if agg.CLEAR else 0.0,
        "IDSW": int(agg.CLEAR.IDSW) if agg.CLEAR else 0,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEFAULTS = {
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
    "p_reset_threshold": 5,
}


def _mp_worker(
    storage_url: str, n: int, sequences: list[str], data_dir_str: str
) -> None:
    """Multiprocessing worker: loads shared Optuna study and runs N trials.

    Runs in a separate process — module-level state (including the class-level
    Kalman patch) is fully isolated, so concurrent workers never race.

    Args:
        storage_url: SQLAlchemy URL for the shared SQLite study database.
        n: Number of trials this worker should run.
        sequences: MOT17-val sequence names to evaluate.
        data_dir_str: Stringified path to the MOT17-val root (pickling-safe).
    """
    _data_dir = Path(data_dir_str)
    _study = optuna.load_study(study_name=_STUDY_NAME, storage=storage_url)

    def _obj(trial: optuna.Trial) -> float:
        params = _define_search_space(trial)
        _apply_kalman_patch(params)
        return _run_eval(params, sequences, _data_dir)["HOTA"]

    _study.optimize(_obj, n_trials=n, show_progress_bar=False)


def main(
    n_trials: int = 1,
    fast: bool = False,
    data_dir: str | None = None,
    n_jobs: int = -1,
) -> None:
    """Run Optuna-based ByteTrack search on MOT17-val.

    Args:
        n_trials: Number of Optuna trials. 1 evaluates default params (campaign metric).
        fast: Single sequence only (~3x faster, for development checks).
        data_dir: MOT17 val directory. Auto-detected from standard cache paths if unset.
        n_jobs: Worker processes for parallel trials. -1 uses all CPU cores. 1 disables
            multiprocessing. Ignored when n_trials=1 (single eval needs no parallelism).
    """
    import multiprocessing
    import os
    import tempfile

    t0 = time.time()
    _data_dir = Path(data_dir) if data_dir else _find_data_dir()

    sequences = sorted(
        d.name
        for d in _data_dir.iterdir()
        if d.is_dir() and (d / "gt" / "gt.txt").exists()
    )
    if not sequences:
        print(f"No sequences found in {_data_dir}", file=sys.stderr)
        sys.exit(1)
    if fast:
        fast_seq = [s for s in sequences if "MOT17-04" in s]
        sequences = fast_seq[:1] if fast_seq else sequences[:1]

    best_config_path = Path(__file__).parent / "best_config.json"

    warm: dict | None = None
    if n_trials != 1 and best_config_path.exists():
        try:
            warm = json.loads(best_config_path.read_text()).get("config")
        except (json.JSONDecodeError, KeyError):
            warm = None

    cpu_count = os.cpu_count() or 1
    n_workers = (
        1
        if n_trials == 1
        else min(n_trials, cpu_count if n_jobs == -1 else max(1, n_jobs))
    )

    if n_workers > 1:
        # Multi-process mode: SQLite-backed shared study, isolated per-process state.
        # Requires optuna[rdb] (sqlalchemy). Install via: uv sync --group optimize
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
            study.enqueue_trial(warm or _DEFAULTS)

            base, rem = divmod(n_trials, n_workers)
            counts = [base + (1 if i < rem else 0) for i in range(n_workers)]
            worker_args = [(storage_url, c, sequences, str(_data_dir)) for c in counts]
            print(f"[→ {n_workers} workers · {n_trials} trials · {cpu_count} cores]")
            with multiprocessing.Pool(n_workers) as pool:
                pool.starmap(_mp_worker, worker_args)

            study = optuna.load_study(study_name=_STUDY_NAME, storage=storage_url)
        finally:
            Path(db_path).unlink(missing_ok=True)
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        # Baseline mode (n_trials=1): evaluate defaults for a clean code-change signal.
        study.enqueue_trial(_DEFAULTS if n_trials == 1 else (warm or _DEFAULTS))

        def objective(trial: optuna.Trial) -> float:
            params = _define_search_space(trial)
            _apply_kalman_patch(params)
            return _run_eval(params, sequences, _data_dir)["HOTA"]

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_metrics = _run_eval(best_params, sequences, _data_dir)
    elapsed = time.time() - t0

    if n_trials > 1:
        prev_hota = 0.0
        if best_config_path.exists():
            try:
                prev_hota = json.loads(best_config_path.read_text()).get("hota", 0.0)
            except (json.JSONDecodeError, KeyError):
                prev_hota = 0.0
        if best_metrics["HOTA"] > prev_hota:
            best_config_path.write_text(
                json.dumps(
                    {"hota": best_metrics["HOTA"], "config": best_params}, indent=2
                )
            )

    hota, idf1, mota, idsw = (
        best_metrics["HOTA"],
        best_metrics["IDF1"],
        best_metrics["MOTA"],
        best_metrics["IDSW"],
    )
    print(f"\n__METRICS__: HOTA={hota:.3f} IDF1={idf1:.3f} MOTA={mota:.3f} IDSW={idsw}")
    print(f"__CONFIG__: {json.dumps(best_params)}")
    print(
        f"__ELAPSED__: {elapsed:.1f}s  __TRIALS__: {n_trials}  __WORKERS__: {n_workers}"
    )
    print(f"__SEQUENCES__: {','.join(sequences)}")


if __name__ == "__main__":
    fire.Fire(main)
