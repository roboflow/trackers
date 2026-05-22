#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_tune_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add the tune subcommand to the argument parser."""
    parser = subparsers.add_parser(
        "tune",
        help="Tune tracker hyperparameters via Optuna.",
        description=(
            "Run Optuna-based hyperparameter optimisation for a registered "
            "tracker using pre-computed detections and ground-truth MOT files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--tracker",
        required=True,
        metavar="ID",
        help="Tracker ID to tune (e.g. bytetrack, sort, ocsort).",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing ground-truth MOT files.",
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help=("Directory containing pre-computed detection files in MOT flat format (one {seq}.txt per sequence)."),
    )
    parser.add_argument(
        "--objective",
        default="HOTA",
        choices=["MOTA", "HOTA", "IDF1"],
        help="Scalar metric to maximise. Default: HOTA.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        metavar="N",
        help="Number of Optuna trials to run. Default: 100.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["CLEAR"],
        choices=["CLEAR", "HOTA", "Identity"],
        help=(
            "Metric families to compute. Default: CLEAR. The family required "
            "by --objective is added automatically if missing."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="IoU threshold for CLEAR and Identity matching. Default: 0.5.",
    )
    parser.add_argument(
        "--seqmap",
        type=Path,
        metavar="PATH",
        help="Sequence map file listing sequences to evaluate.",
    )
    parser.add_argument(
        "--fixed-params",
        type=str,
        metavar="JSON",
        help=("JSON object of tracker kwargs held fixed for every trial (e.g. '{\"enable_cmc\": false}')."),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        metavar="DIR",
        help="MOT image root ({sequence}/img1/) for trackers that need frames (e.g. BoTSORT CMC).",
    )
    parser.add_argument(
        "--no-enqueue-defaults",
        action="store_true",
        help="Skip the baseline trial that uses tracker/search_space defaults.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for Optuna sampling (reproducible hyperparameter trials).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        metavar="PATH",
        help="Output file for best parameters (JSON format).",
    )

    parser.set_defaults(func=run_tune)


def run_tune(args: argparse.Namespace) -> int:
    """Execute the tune command."""
    fixed_params = None
    if args.fixed_params is not None:
        try:
            fixed_params = json.loads(args.fixed_params)
        except json.JSONDecodeError as e:
            print(f"Invalid --fixed-params JSON: {e}", file=sys.stderr)
            return 1
        if not isinstance(fixed_params, dict):
            print("--fixed-params must be a JSON object", file=sys.stderr)
            return 1

    return tune(
        tracker=args.tracker,
        gt_dir=args.gt_dir,
        detections_dir=args.detections_dir,
        objective=args.objective,
        n_trials=args.n_trials,
        metrics=args.metrics,
        threshold=args.threshold,
        seqmap=args.seqmap,
        fixed_params=fixed_params,
        images_dir=args.images_dir,
        enqueue_defaults=not args.no_enqueue_defaults,
        seed=args.seed,
        output=args.output,
    )


def tune(
    tracker: str,
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
        enqueue_defaults: Whether to run a baseline trial before sampling.
        fixed_params: Tracker kwargs held constant for every trial.
        images_dir: MOT image root for frame-based features (e.g. CMC).
        seed: Random seed for Optuna's TPE sampler.
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
            enqueue_defaults=enqueue_defaults,
            fixed_params=fixed_params,
            images_dir=images_dir,
            seed=seed,
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
