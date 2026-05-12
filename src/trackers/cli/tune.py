# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.command("tune")
@click.option(
    "--tracker", "tracker_id", required=True, metavar="ID", help="Tracker ID to tune (e.g. bytetrack, sort, ocsort)."
)
@click.option(
    "--gt-dir",
    type=click.Path(path_type=Path),
    required=True,
    metavar="DIR",
    help="Directory containing ground-truth MOT files.",
)
@click.option(
    "--detections-dir",
    type=click.Path(path_type=Path),
    required=True,
    metavar="DIR",
    help="Directory containing pre-computed detection files in MOT flat format (one {seq}.txt per sequence).",
)
@click.option(
    "--objective",
    default="HOTA",
    type=click.Choice(["MOTA", "HOTA", "IDF1"]),
    help="Scalar metric to maximise. Default: HOTA.",
)
@click.option(
    "--n-trials", "n_trials", type=int, default=100, metavar="N", help="Number of Optuna trials to run. Default: 100."
)
@click.option(
    "--metrics",
    multiple=True,
    default=("CLEAR",),
    type=click.Choice(["CLEAR", "HOTA", "Identity"]),
    help="Metric families to compute. Default: CLEAR.",
)
@click.option(
    "--threshold", type=float, default=0.5, help="IoU threshold for CLEAR and Identity matching. Default: 0.5."
)
@click.option(
    "--seqmap",
    type=click.Path(path_type=Path),
    default=None,
    metavar="PATH",
    help="Sequence map file listing sequences to evaluate.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    metavar="PATH",
    help="Output file for best parameters (JSON format).",
)
def tune_command(
    tracker_id: str,
    gt_dir: Path,
    detections_dir: Path,
    objective: str,
    n_trials: int,
    metrics: tuple[str, ...],
    threshold: float,
    seqmap: Path | None,
    output: Path | None,
) -> None:
    """Tune tracker hyperparameters via Optuna."""
    rc = tune(
        tracker=tracker_id,
        gt_dir=gt_dir,
        detections_dir=detections_dir,
        objective=objective,
        n_trials=n_trials,
        metrics=list(metrics),
        threshold=threshold,
        seqmap=seqmap,
        output=output,
    )
    if rc != 0:
        sys.exit(rc)


def tune(
    tracker: str,
    gt_dir: Path,
    detections_dir: Path,
    objective: str = "HOTA",
    n_trials: int = 100,
    metrics: list[str] | None = None,
    threshold: float = 0.5,
    seqmap: Path | None = None,
    output: Path | None = None,
) -> int:
    """Tune tracker hyperparameters using Optuna.

    Args:
        tracker: Tracker ID to tune (e.g. bytetrack, sort).
        gt_dir: Directory of ground-truth MOT files.
        detections_dir: Directory of pre-computed detection files in MOT flat
            format (one {seq}.txt per sequence).
        objective: Scalar metric to maximise. Options: MOTA, HOTA, IDF1.
        n_trials: Number of Optuna trials to run.
        metrics: Metric families to compute. Options: CLEAR, HOTA, Identity.
            Default: CLEAR.
        threshold: IoU threshold for CLEAR and Identity matching.
        seqmap: Sequence map file listing sequences to evaluate.
        output: Output file path for best parameters (JSON format).

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    if metrics is None:
        metrics = ["CLEAR"]

    from trackers.tune import Tuner

    try:
        tuner = Tuner(
            tracker_id=tracker,
            gt_dir=gt_dir,
            detections_dir=detections_dir,
            metrics=metrics,
            objective=objective,
            n_trials=n_trials,
            threshold=threshold,
            seqmap=seqmap,
        )
    except (ValueError, ImportError, FileNotFoundError) as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        best_params = tuner.run()
    except Exception as e:
        print(f"Error during tuning: {e}", file=sys.stderr)
        return 1

    print(f"\nBest parameters for {tracker}:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")
    if tuner.study is not None:
        print(f"\nBest {objective}: {tuner.study.best_value:.4f}")

    if output:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(best_params, indent=2))
        except OSError as e:
            print(f"Error writing output: {e}", file=sys.stderr)
            return 1
        print(f"\nResults saved to: {output}")

    return 0
