#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""``trackers eval`` subcommand — evaluate tracker predictions against ground truth."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def eval_cmd(
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
    """Evaluate tracker predictions against ground-truth MOT files.

    Two modes:

    - Single sequence: pass ``gt`` and ``tracker``.
    - Benchmark: pass ``gt_dir`` and ``tracker_dir`` (with optional ``seqmap``).

    Args:
        gt: Ground-truth file (MOT format) for single-sequence mode.
        tracker: Tracker predictions file (MOT format) for single-sequence mode.
        gt_dir: Directory of ground-truth files for benchmark mode.
        tracker_dir: Directory of tracker prediction files for benchmark mode.
        seqmap: Sequence map listing sequences to evaluate.
        metrics: Metrics to compute. Options: ``CLEAR``, ``HOTA``, ``Identity``.
            Defaults to ``["CLEAR"]``.
        threshold: IoU threshold for CLEAR and Identity matching.
        columns: Metric columns to display. ``None`` auto-selects from
            available metrics.
        output: Output JSON file for results.

    Returns:
        Exit code: ``0`` on success, ``1`` on error.
    """
    metrics = metrics or ["CLEAR"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    single_mode = gt is not None and tracker is not None
    benchmark_mode = gt_dir is not None and tracker_dir is not None

    if not single_mode and not benchmark_mode:
        print("Error: Must specify either --gt/--tracker or --gt_dir/--tracker_dir", file=sys.stderr)
        return 1

    if single_mode and benchmark_mode:
        print("Error: Cannot use both single sequence and benchmark mode", file=sys.stderr)
        return 1

    from trackers.eval import evaluate_mot_sequence, evaluate_mot_sequences

    try:
        if single_mode:
            assert gt is not None and tracker is not None  # noqa: S101 — narrows for type checker
            seq_result = evaluate_mot_sequence(
                gt_path=gt,
                tracker_path=tracker,
                metrics=metrics,
                threshold=threshold,
            )
            print(seq_result.table(columns=columns))
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(seq_result.json())
                print(f"\nResults saved to: {output}")
        else:
            assert gt_dir is not None and tracker_dir is not None  # noqa: S101 — narrows for type checker
            bench_result = evaluate_mot_sequences(
                gt_dir=gt_dir,
                tracker_dir=tracker_dir,
                seqmap=seqmap,
                metrics=metrics,
                threshold=threshold,
            )
            print(bench_result.table(columns=columns))
            if output:
                bench_result.save(output)
                print(f"\nResults saved to: {output}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0
