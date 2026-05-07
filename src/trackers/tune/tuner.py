# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Optuna-based hyperparameter tuner for registered MOT trackers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import supervision as sv

from trackers.core.base import BaseTracker
from trackers.eval.evaluate import evaluate_mot_sequences
from trackers.eval.results import BenchmarkResult
from trackers.io.mot import _mot_frame_to_detections, _MOTOutput, load_mot_file

if TYPE_CHECKING:
    import optuna


class Tuner:
    """Wraps Optuna to tune hyperparameters of a registered MOT tracker.

    Uses the tracker's ``search_space`` ClassVar to define the Optuna parameter
    distributions. For each trial the tuner instantiates the tracker with the
    sampled parameters, runs it frame-by-frame over every sequence using
    pre-computed detections from ``detections_dir``, and evaluates the
    predictions with ``evaluate_mot_sequences``.

    Args:
        tracker_id: Registered tracker identifier (e.g. ``"bytetrack"``).
        gt_dir: Directory of ground-truth MOT files.
        detections_dir: Directory of pre-computed detection files in MOT
            format — one ``{seq}.txt`` per sequence, where each line must
            contain at least ``<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,
            <bb_height>`` (6 comma-separated columns, with 1-based frame
            index). Use ``id=-1`` for detections (no pre-assigned ID).
            Optional trailing MOT columns such as ``<conf>`` and additional
            class / visibility fields may also be present; when omitted,
            ``load_mot_file()`` applies its default values.
        metrics: Metric families to compute. Supported values are
            ``["CLEAR", "HOTA", "Identity"]``. Defaults to ``["CLEAR"]``.
            The family required by ``objective`` is added automatically if
            missing (e.g. ``objective="HOTA"`` adds ``"HOTA"`` to metrics).
        objective: Scalar metric field to maximise (e.g. ``"MOTA"``,
            ``"HOTA"``, ``"IDF1"``). Case-insensitive. Defaults to
            ``"MOTA"``.
        n_trials: Number of Optuna trials to run. Defaults to ``100``.
        threshold: IoU threshold forwarded to ``evaluate_mot_sequences``.
            Defaults to ``0.5``.
        seqmap: Optional path to a sequence map file. When provided only the
            listed sequences are evaluated.

    Examples:
        Tune ByteTrack hyperparameters on a local dataset::

            from trackers.tune import Tuner

            tuner = Tuner(
                tracker_id="bytetrack",
                gt_dir="data/gt/",
                detections_dir="data/det/",
                n_trials=50,
            )
            best_params = tuner.run()
    """

    def __init__(
        self,
        tracker_id: str,
        gt_dir: str | Path,
        detections_dir: str | Path,
        metrics: list[str] | None = None,
        objective: str = "MOTA",
        n_trials: int = 100,
        threshold: float = 0.5,
        seqmap: str | Path | None = None,
    ) -> None:
        try:
            import optuna as _optuna

            self._optuna = _optuna
        except ImportError as exc:
            raise ImportError(
                "Error: optuna is required for hyperparameter tuning. "
                "Install it with: pip install 'trackers[tune]'"
            ) from exc

        tracker_info = BaseTracker._lookup_tracker(tracker_id)
        if tracker_info is None:
            raise ValueError(
                f"Tracker {tracker_id!r} is not registered. "
                f"Available trackers: {BaseTracker._registered_trackers()}"
            )

        search_space = tracker_info.tracker_class.search_space
        if not search_space:
            raise ValueError(
                f"Tracker {tracker_id!r} does not define a search_space. "
                "Add a search_space ClassVar to enable tuning."
            )

        self._tracker_id = tracker_id
        self._tracker_info = tracker_info
        self._search_space: dict[str, dict] = search_space
        self._gt_dir = Path(gt_dir)
        self._detections_dir = Path(detections_dir)
        self._metrics = list(metrics) if metrics else ["CLEAR"]
        self._objective_metric = objective.upper()
        self._n_trials = n_trials

        # Auto-add the metric family required by the chosen objective so
        # callers don't need to remember the mapping themselves.
        _objective_to_family = {
            "MOTA": "CLEAR",
            "HOTA": "HOTA",
            "IDF1": "Identity",
        }
        required_family = _objective_to_family.get(self._objective_metric)
        if required_family and required_family not in self._metrics:
            self._metrics = [*self._metrics, required_family]
        self._threshold = threshold
        self._sequences = _discover_sequences(self._detections_dir, seqmap)
        self.study: optuna.Study | None = None

        if not self._sequences:
            raise ValueError(f"No sequences found in {self._detections_dir}")

        self._validate_sequence_files()

    def _validate_sequence_files(self) -> None:
        """Validate that every selected sequence has required MOT files.

        This performs eager filesystem validation so configuration errors are
        reported during tuner initialization rather than later during trial
        execution.
        """
        missing_detection_files = [
            str(self._detections_dir / f"{seq_name}.txt")
            for seq_name in self._sequences
            if not (self._detections_dir / f"{seq_name}.txt").is_file()
        ]
        if missing_detection_files:
            raise FileNotFoundError(
                "Missing detection files for selected sequences: "
                + ", ".join(missing_detection_files)
            )

        missing_gt_files = [
            str(self._gt_dir / f"{seq_name}.txt")
            for seq_name in self._sequences
            if not (self._gt_dir / f"{seq_name}.txt").is_file()
        ]
        if missing_gt_files:
            raise FileNotFoundError(
                "Missing ground-truth files for selected sequences: "
                + ", ".join(missing_gt_files)
            )

    def _objective(self, trial: optuna.Trial) -> float:
        """Sample hyperparameters, run tracker over all sequences, return metric.

        Args:
            trial: Optuna trial used to sample parameter values.

        Returns:
            Scalar metric value for this trial.
        """
        params: dict[str, Any] = {}
        for name, spec in self._search_space.items():
            stype = spec["type"]
            if stype == "randint":
                low, high = spec["range"]
                params[name] = trial.suggest_int(name, low, high)
            elif stype == "uniform":
                low, high = spec["range"]
                params[name] = trial.suggest_float(name, low, high)
            elif stype == "choice":
                params[name] = trial.suggest_categorical(name, spec["options"])
            else:
                raise ValueError(
                    f"Unknown search_space type: {stype!r}. "
                    "Valid types: 'randint', 'uniform', 'choice'"
                )

        # Pass only sampled parameters so tracker __init__ defaults apply naturally.
        tracker = self._tracker_info.tracker_class(**params)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            for seq_name in self._sequences:
                tracker.reset()
                det_path = self._detections_dir / f"{seq_name}.txt"
                pred_path = output_dir / f"{seq_name}.txt"
                _run_tracker_on_detections(tracker, det_path, pred_path)

            seqmap = getattr(self, "_seqmap", None)
            if seqmap is None:
                seqmap = output_dir / "seqmap.txt"
                seqmap.write_text(
                    "\n".join(self._sequences) + "\n",
                    encoding="utf-8",
                )

            result: BenchmarkResult = evaluate_mot_sequences(
                gt_dir=self._gt_dir,
                tracker_dir=output_dir,
                metrics=self._metrics,
                threshold=self._threshold,
                seqmap=seqmap,
            )

        return _extract_metric(result, self._objective_metric)

    def run(self) -> dict[str, Any]:
        """Create an Optuna study, run trials, and return the best parameters.

        Returns:
            Dictionary mapping each ``search_space`` parameter name to its
            best value found across all trials.
        """
        self.study = self._optuna.create_study(
            direction="maximize",
            study_name=f"trackers-tune-{self._tracker_id}",
        )
        self.study.optimize(self._objective, n_trials=self._n_trials)
        return dict(self.study.best_params)


def _discover_sequences(
    detections_dir: str | Path,
    seqmap: str | Path | None,
) -> list[str]:
    """Return the list of sequence names to tune over.

    Reads sequence names from ``seqmap`` when provided, otherwise discovers
    them by globbing ``*.txt`` files in ``detections_dir``.

    Args:
        detections_dir: Directory containing ``{seq}.txt`` detection files.
        seqmap: Optional sequence map file. Each non-comment, non-empty line
            is treated as a sequence name.

    Returns:
        Sorted list of sequence names.
    """
    detections_dir = Path(detections_dir)
    if seqmap is not None:
        lines = Path(seqmap).read_text().splitlines()
        return [
            ln.strip()
            for ln in lines
            if ln.strip() and not ln.startswith("#") and ln.strip().lower() != "name"
        ]
    return sorted(p.stem for p in detections_dir.glob("*.txt"))


def _run_tracker_on_detections(
    tracker: BaseTracker,
    det_path: Path,
    pred_path: Path,
) -> None:
    """Run a tracker on a MOT detection file and write predictions.

    Iterates every frame from 1 to the last frame in the detection file,
    feeding ``sv.Detections.empty()`` for frames with no detections so the
    tracker can age and prune its internal state correctly.

    Args:
        tracker: Tracker instance already reset for this sequence.
        det_path: Path to the MOT-format detection file.
        pred_path: Destination path for the MOT-format prediction file.
    """
    det_data = load_mot_file(det_path)
    max_frame = max(det_data.keys())

    with _MOTOutput(pred_path) as mot_out:
        for frame_idx in range(1, max_frame + 1):
            if frame_idx in det_data:
                dets = _mot_frame_to_detections(det_data[frame_idx])
            else:
                dets = sv.Detections.empty()
            # TODO: Add frame reading to tuner class
            tracked = tracker.update(dets)
            mot_out.write(frame_idx, tracked)


def _extract_metric(result: BenchmarkResult, metric: str) -> float:
    """Extract a scalar metric value from ``BenchmarkResult.aggregate``.

    Searches CLEAR, HOTA, and Identity metrics in order.

    Args:
        result: Benchmark result returned by ``evaluate_mot_sequences``.
        metric: Field name to extract (e.g. ``"MOTA"``, ``"HOTA"``,
            ``"IDF1"``).

    Returns:
        The metric value as a float.
    """
    agg = result.aggregate
    for metrics_obj in (agg.CLEAR, agg.HOTA, agg.Identity):
        if metrics_obj is None:
            continue
        value = metrics_obj.to_dict().get(metric)
        if value is not None:
            return float(value)

    raise ValueError(
        f"Metric {metric!r} not found in BenchmarkResult.aggregate. "
        "Ensure the corresponding metric family is included in `metrics`."
    )
