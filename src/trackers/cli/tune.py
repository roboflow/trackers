#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""``trackers tune`` subcommand — Optuna hyperparameter optimisation."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrackerSelection:
    """Tracking algorithm selection for ``tune``.

    A minimal, one-field counterpart to
    :class:`trackers.cli.track.TrackerOptions` — tune has no per-parameter CLI
    surface (that is what ``fixed_params`` and ``search_space`` are for), so
    only the algorithm selector needs a nested group.

    Attributes:
        name: Tracker ID to tune (e.g. ``bytetrack``, ``sort``, ``ocsort``).
            Discoverable via ``BaseTracker._registered_trackers()``.
            ``--tracker <id>`` is the shorthand spelling of
            ``--tracker.name <id>``, matching ``trackers track``.
    """

    name: str


def tune_command(
    tracker: TrackerSelection,
    gt_dir: Path,
    detections_dir: Path,
    objective: str = "HOTA",
    n_trials: int = 100,
    metrics: list[str] | None = None,
    threshold: float = 0.5,
    seqmap: Path | None = None,
    fixed_params: dict | None = None,
    images_dir: Path | None = None,
    enqueue_defaults: bool = True,
    seed: int | None = None,
    output: Path | None = None,
) -> int:
    """Tune tracker hyperparameters using Optuna.

    Args:
        tracker: Tracking algorithm selection.
        gt_dir: Directory of ground-truth MOT files.
        detections_dir: Directory of pre-computed detection files in MOT flat
            format (one ``{seq}.txt`` per sequence).
        objective: Scalar metric to maximise. Options: ``MOTA``, ``HOTA``,
            ``IDF1``.
        n_trials: Number of Optuna trials to run.
        metrics: Metric families to compute. Options: ``CLEAR``, ``HOTA``,
            ``Identity``. Default: ``["CLEAR"]``. The family required by
            ``objective`` is added automatically if missing.
        threshold: IoU threshold for CLEAR and Identity matching.
        seqmap: Sequence map file listing sequences to evaluate.
        fixed_params: Tracker ``__init__`` kwargs held constant across all
            trials (e.g. ``{"enable_cmc": False}``).
        images_dir: MOT-style image root for frame-based features such as CMC.
            Frames are read from ``{images_dir}/{sequence}/img1/``.
        enqueue_defaults: When ``True`` (default), the first trial evaluates
            the tracker's default parameters before Optuna sampling begins.
        seed: Random seed for Optuna's TPE sampler for reproducible runs.
        output: Output JSON file for best parameters.

    Returns:
        Exit code: ``0`` on success, ``1`` on error.
    """
    if metrics is None:
        metrics = ["CLEAR"]

    from trackers.tune import Tuner

    try:
        tuner = Tuner(
            tracker_id=tracker.name,
            gt_dir=gt_dir,
            detections_dir=detections_dir,
            metrics=metrics,
            objective=objective,
            n_trials=n_trials,
            threshold=threshold,
            seqmap=seqmap,
            fixed_params=fixed_params,
            images_dir=images_dir,
            enqueue_defaults=enqueue_defaults,
            seed=seed,
        )
    except (ValueError, ImportError, FileNotFoundError) as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        best_params = tuner.run()
    except Exception as e:
        print(f"Error during tuning: {e}", file=sys.stderr)
        return 1

    print(f"\nBest parameters for {tracker.name}:")
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
