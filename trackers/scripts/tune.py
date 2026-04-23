#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
import sys
from pathlib import Path


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
    """
    if metrics is None:
        metrics = ["CLEAR"]

    # Normalize objective to uppercase so metric lookup is case-insensitive
    objective = objective.upper()

    # Auto-add metric family required by the chosen objective
    OBJECTIVE_TO_FAMILY = {
        "MOTA": "CLEAR",
        "HOTA": "HOTA",
        "IDF1": "Identity",
    }
    required_family = OBJECTIVE_TO_FAMILY.get(objective)
    if required_family and required_family not in metrics:
        metrics = [*list(metrics), required_family]

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
    except (ValueError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
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
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(best_params, indent=2))
        print(f"\nResults saved to: {output}")

    return 0
