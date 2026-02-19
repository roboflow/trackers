#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import logging
import sys
from pathlib import Path


def evaluate(
    gt: Path | None = None,
    tracker: Path | None = None,
    gt_dir: Path | None = None,
    tracker_dir: Path | None = None,
    seqmap: Path | None = None,
    metrics: list[str] | None = None,
    threshold: float = 0.5,
    columns: list[str] | None = None,
    output: Path | None = None,
) -> int:
    """Evaluate tracker predictions against ground truth.

    Args:
        gt: Path to ground truth file (MOT format).
        tracker: Path to tracker predictions file (MOT format).
        gt_dir: Directory containing ground truth files.
        tracker_dir: Directory containing tracker prediction files.
        seqmap: Sequence map file listing sequences to evaluate.
        metrics: Metrics to compute. Options: CLEAR, HOTA, Identity.
            Default: CLEAR.
        threshold: IoU threshold for CLEAR and Identity matching.
        columns: Metric columns to display. Default: auto-selected based on
            metrics. CLEAR: MOTA, MOTP, MODA, CLR_Re, CLR_Pr, MTR, PTR, MLR,
            sMOTA, CLR_TP, CLR_FN, CLR_FP, IDSW, MT, PT, ML, Frag. HOTA:
            HOTA, DetA, AssA, DetRe, DetPr, AssRe, AssPr, LocA. Identity:
            IDF1, IDR, IDP, IDTP, IDFN, IDFP.
        output: Output file for results (JSON format).
    """
    # Configure logging to show detection info
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    if metrics is None:
        metrics = ["CLEAR"]

    # Validate arguments
    single_mode = gt is not None and tracker is not None
    benchmark_mode = gt_dir is not None and tracker_dir is not None

    if not single_mode and not benchmark_mode:
        print(
            "Error: Must specify either --gt/--tracker or --gt_dir/--tracker_dir",
            file=sys.stderr,
        )
        return 1

    if single_mode and benchmark_mode:
        print(
            "Error: Cannot use both single sequence and benchmark mode",
            file=sys.stderr,
        )
        return 1

    # Import evaluation functions
    from trackers.eval import evaluate_mot_sequence, evaluate_mot_sequences

    try:
        if single_mode:
            seq_result = evaluate_mot_sequence(
                gt_path=gt, tracker_path=tracker, metrics=metrics, threshold=threshold
            )
            print(seq_result.table(columns=columns))

            # Save results if output specified
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(seq_result.json())
                print(f"\nResults saved to: {output}")
        else:
            bench_result = evaluate_mot_sequences(
                gt_dir=gt_dir,
                tracker_dir=tracker_dir,
                seqmap=seqmap,
                metrics=metrics,
                threshold=threshold,
            )
            print(bench_result.table(columns=columns))

            # Save results if output specified
            if output:
                bench_result.save(output)
                print(f"\nResults saved to: {output}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0
