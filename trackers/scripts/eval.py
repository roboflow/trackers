#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def add_eval_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add the eval subcommand to the argument parser."""
    parser = subparsers.add_parser(
        "eval",
        help="Evaluate tracker predictions against ground truth.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Single sequence mode
    single_group = parser.add_argument_group("single sequence evaluation")
    single_group.add_argument(
        "--gt",
        type=Path,
        metavar="PATH",
        help="Path to ground truth file (MOT format).",
    )
    single_group.add_argument(
        "--tracker",
        type=Path,
        metavar="PATH",
        help="Path to tracker predictions file (MOT format).",
    )

    # Benchmark mode
    bench_group = parser.add_argument_group("benchmark evaluation")
    bench_group.add_argument(
        "--gt-dir",
        type=Path,
        metavar="DIR",
        help="Directory containing ground truth files.",
    )
    bench_group.add_argument(
        "--tracker-dir",
        type=Path,
        metavar="DIR",
        help="Directory containing tracker prediction files.",
    )
    bench_group.add_argument(
        "--seqmap",
        type=Path,
        metavar="PATH",
        help="Sequence map file listing sequences to evaluate.",
    )
    bench_group.add_argument(
        "--benchmark",
        type=str,
        metavar="NAME",
        help="Override auto-detected benchmark name (e.g., MOT17, SportsMOT).",
    )
    bench_group.add_argument(
        "--split",
        type=str,
        metavar="NAME",
        help="Override auto-detected split name (e.g., train, val, test).",
    )
    bench_group.add_argument(
        "--tracker-name",
        type=str,
        metavar="NAME",
        help="Override auto-detected tracker name.",
    )

    # Common options
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["CLEAR"],
        choices=["CLEAR", "HOTA"],
        help="Metrics to compute. Default: CLEAR. Options: CLEAR, HOTA",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="IoU threshold for CLEAR matching. Default: 0.5",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        metavar="COL",
        help=(
            "Metric columns to display. Default: auto-selected based on metrics. "
            "CLEAR: MOTA, MOTP, MODA, CLR_Re, CLR_Pr, MTR, PTR, MLR, sMOTA, "
            "CLR_TP, CLR_FN, CLR_FP, IDSW, MT, PT, ML, Frag. "
            "HOTA: HOTA, DetA, AssA, DetRe, DetPr, AssRe, AssPr, LocA"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        metavar="PATH",
        help="Output file for results (JSON format).",
    )

    parser.set_defaults(func=run_eval)


def run_eval(args: argparse.Namespace) -> int:
    """Execute the eval command."""
    # Configure logging to show detection info
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    # Validate arguments
    single_mode = args.gt is not None and args.tracker is not None
    benchmark_mode = args.gt_dir is not None and args.tracker_dir is not None

    if not single_mode and not benchmark_mode:
        print(
            "Error: Must specify either --gt/--tracker or --gt-dir/--tracker-dir",
            file=sys.stderr,
        )
        return 1

    if single_mode and benchmark_mode:
        print(
            "Error: Cannot use both single sequence and benchmark mode",
            file=sys.stderr,
        )
        return 1

    # Columns: None means auto-select based on available metrics
    columns = args.columns

    # Import evaluation functions
    from trackers.eval import evaluate_benchmark, evaluate_mot_sequence

    try:
        if single_mode:
            seq_result = evaluate_mot_sequence(
                gt_path=args.gt,
                tracker_path=args.tracker,
                metrics=args.metrics,
                threshold=args.threshold,
            )
            print(seq_result.table(columns=columns))

            # Save results if output specified
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(seq_result.json())
                print(f"\nResults saved to: {args.output}")
        else:
            bench_result = evaluate_benchmark(
                gt_dir=args.gt_dir,
                tracker_dir=args.tracker_dir,
                seqmap=args.seqmap,
                metrics=args.metrics,
                threshold=args.threshold,
                benchmark=args.benchmark,
                split=args.split,
                tracker_name=args.tracker_name,
            )
            print(bench_result.table(columns=columns))

            # Save results if output specified
            if args.output:
                bench_result.save(args.output)
                print(f"\nResults saved to: {args.output}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0
