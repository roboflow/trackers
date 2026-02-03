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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trackers.eval.results import BenchmarkResult, SequenceResult

# Check for optional rich dependency
try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# TrackEval summary field order
CLEAR_FLOAT_FIELDS = [
    "MOTA",
    "MOTP",
    "MODA",
    "CLR_Re",
    "CLR_Pr",
    "MTR",
    "PTR",
    "MLR",
    "sMOTA",
]
CLEAR_INT_FIELDS = [
    "CLR_TP",
    "CLR_FN",
    "CLR_FP",
    "IDSW",
    "MT",
    "PT",
    "ML",
    "Frag",
]
DEFAULT_COLUMNS = CLEAR_FLOAT_FIELDS + CLEAR_INT_FIELDS


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
        choices=["CLEAR"],
        help="Metrics to compute. Default: CLEAR",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching. Default: 0.5",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        metavar="COL",
        help=(
            "Metric columns to display. Default: all. "
            "Available: MOTA, MOTP, MODA, CLR_Re, CLR_Pr, MTR, PTR, MLR, "
            "sMOTA, CLR_TP, CLR_FN, CLR_FP, IDSW, MT, PT, ML, Frag"
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

    # Check for rich and show hint if not available
    if not RICH_AVAILABLE:
        print(
            "Tip: Install 'rich' for better output formatting: "
            "pip install trackers[cli]",
            file=sys.stderr,
        )
        print(file=sys.stderr)

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

    # Determine columns to display
    columns = args.columns if args.columns else DEFAULT_COLUMNS

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
            _print_single_results(seq_result, columns)

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
            _print_benchmark_results(bench_result, columns)

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


def _format_value(value: float | int, metric: str) -> str:
    """Format a metric value for display (TrackEval format).

    Float metrics are displayed as XX.XXX (percentages without % symbol).
    Integer metrics are displayed as-is.
    """
    if metric in CLEAR_FLOAT_FIELDS:
        # Display as percentage with 3 decimal places (TrackEval format)
        return f"{value * 100:.3f}"
    return str(value)


def _print_single_results(result: SequenceResult, columns: list[str]) -> None:
    """Print results for a single sequence."""
    if RICH_AVAILABLE:
        _print_single_results_rich(result, columns)
    else:
        _print_single_results_plain(result, columns)


def _print_single_results_rich(result: SequenceResult, columns: list[str]) -> None:
    """Print single sequence results using rich."""
    console = Console()
    console.print(f"\nResults for {result.sequence}\n")

    table = Table(title="CLEAR Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    metrics_dict = result.CLEAR.to_dict()
    for col in columns:
        if col in metrics_dict:
            value = metrics_dict[col]
            formatted = _format_value(value, col)
            table.add_row(col, formatted)

    console.print(table)


def _print_single_results_plain(result: SequenceResult, columns: list[str]) -> None:
    """Print single sequence results in plain text."""
    print(f"\nResults for {result.sequence}")
    print("=" * 40)
    print("\nCLEAR Metrics:")

    metrics_dict = result.CLEAR.to_dict()
    for col in columns:
        if col in metrics_dict:
            value = metrics_dict[col]
            formatted = _format_value(value, col)
            print(f"  {col}: {formatted}")


def _print_benchmark_results(result: BenchmarkResult, columns: list[str]) -> None:
    """Print results for benchmark evaluation."""
    if RICH_AVAILABLE:
        _print_benchmark_results_rich(result, columns)
    else:
        _print_benchmark_results_plain(result, columns)


def _print_benchmark_results_rich(result: BenchmarkResult, columns: list[str]) -> None:
    """Print benchmark results using rich."""
    console = Console()
    console.print("\nBenchmark Results\n")

    table = Table(title="CLEAR Metrics by Sequence")
    table.add_column("Sequence", style="cyan")

    # Determine column widths based on max value length
    for col in columns:
        justify = "right"
        table.add_column(col, justify=justify)

    # Add sequence rows
    for seq_name in sorted(result.sequences.keys()):
        seq_result = result.sequences[seq_name]
        metrics_dict = seq_result.CLEAR.to_dict()

        row_values = [seq_name]
        for col in columns:
            value = metrics_dict.get(col, 0)
            formatted = _format_value(value, col)
            row_values.append(formatted)

        table.add_row(*row_values)

    # Add separator and aggregate row
    table.add_section()

    agg_dict = result.aggregate.CLEAR.to_dict()
    agg_values = ["[bold]COMBINED[/bold]"]
    for col in columns:
        value = agg_dict.get(col, 0)
        formatted = _format_value(value, col)
        agg_values.append(f"[bold]{formatted}[/bold]")

    table.add_row(*agg_values)

    console.print(table)


def _print_benchmark_results_plain(result: BenchmarkResult, columns: list[str]) -> None:
    """Print benchmark results in plain text."""
    print("\nBenchmark Results")
    print("=" * 120)

    # Determine column widths
    col_widths = {"Sequence": 30}
    for col in columns:
        # Use max of header length and typical value length
        col_widths[col] = max(len(col), 10)

    # Print header
    header = "Sequence".ljust(col_widths["Sequence"])
    for col in columns:
        header += col.rjust(col_widths[col]) + "  "
    print(header)
    print("-" * len(header))

    # Print sequence rows
    for seq_name in sorted(result.sequences.keys()):
        seq_result = result.sequences[seq_name]
        metrics_dict = seq_result.CLEAR.to_dict()

        row = seq_name.ljust(col_widths["Sequence"])
        for col in columns:
            value = metrics_dict.get(col, 0)
            formatted = _format_value(value, col)
            row += formatted.rjust(col_widths[col]) + "  "
        print(row)

    # Print aggregate row
    print("-" * len(header))
    agg_dict = result.aggregate.CLEAR.to_dict()
    row = "COMBINED".ljust(col_widths["Sequence"])
    for col in columns:
        value = agg_dict.get(col, 0)
        formatted = _format_value(value, col)
        row += formatted.rjust(col_widths[col]) + "  "
    print(row)
