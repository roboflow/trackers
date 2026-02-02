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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Check for optional rich dependency
try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


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
        "--output",
        "-o",
        type=Path,
        metavar="PATH",
        help="Output file for results (JSON format).",
    )

    parser.set_defaults(func=run_eval)


def run_eval(args: argparse.Namespace) -> int:
    """Execute the eval command."""
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

    # Import evaluation functions
    from trackers.eval import evaluate_benchmark, evaluate_mot_sequence

    try:
        if single_mode:
            results = evaluate_mot_sequence(
                gt_path=args.gt,
                tracker_path=args.tracker,
                metrics=args.metrics,
                threshold=args.threshold,
            )
            _print_single_results(results, args.gt.stem)
        else:
            results = evaluate_benchmark(
                gt_dir=args.gt_dir,
                tracker_dir=args.tracker_dir,
                seqmap=args.seqmap,
                metrics=args.metrics,
                threshold=args.threshold,
            )
            _print_benchmark_results(results)

        # Save results if output specified
        if args.output:
            _save_results(results, args.output)
            print(f"\nResults saved to: {args.output}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def _print_single_results(results: dict[str, Any], sequence_name: str) -> None:
    """Print results for a single sequence."""
    if RICH_AVAILABLE:
        _print_single_results_rich(results, sequence_name)
    else:
        _print_single_results_plain(results, sequence_name)


def _print_single_results_rich(results: dict[str, Any], sequence_name: str) -> None:
    """Print single sequence results using rich."""
    console = Console()
    console.print(f"\n[bold]Results for {sequence_name}[/bold]\n")

    if "CLEAR" in results:
        clear = results["CLEAR"]
        table = Table(title="CLEAR Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        # Main metrics
        table.add_row("MOTA", f"{clear['MOTA']:.1%}")
        table.add_row("MOTP", f"{clear['MOTP']:.3f}")
        table.add_row("IDs", f"{clear['IDSW']}")
        table.add_section()

        # Detection metrics
        table.add_row("TP", f"{clear['CLR_TP']}")
        table.add_row("FP", f"{clear['CLR_FP']}")
        table.add_row("FN", f"{clear['CLR_FN']}")
        table.add_section()

        # Track quality
        table.add_row("MT", f"{clear['MT']} ({clear['MTR']:.1%})")
        table.add_row("PT", f"{clear['PT']} ({clear['PTR']:.1%})")
        table.add_row("ML", f"{clear['ML']} ({clear['MLR']:.1%})")
        table.add_row("Frag", f"{clear['Frag']}")

        console.print(table)


def _print_single_results_plain(results: dict[str, Any], sequence_name: str) -> None:
    """Print single sequence results in plain text."""
    print(f"\nResults for {sequence_name}")
    print("=" * 40)

    if "CLEAR" in results:
        clear = results["CLEAR"]
        print("\nCLEAR Metrics:")
        print(f"  MOTA: {clear['MOTA']:.1%}")
        print(f"  MOTP: {clear['MOTP']:.3f}")
        print(f"  IDs:  {clear['IDSW']}")
        print(f"  TP:   {clear['CLR_TP']}")
        print(f"  FP:   {clear['CLR_FP']}")
        print(f"  FN:   {clear['CLR_FN']}")
        print(f"  MT:   {clear['MT']} ({clear['MTR']:.1%})")
        print(f"  PT:   {clear['PT']} ({clear['PTR']:.1%})")
        print(f"  ML:   {clear['ML']} ({clear['MLR']:.1%})")
        print(f"  Frag: {clear['Frag']}")


def _print_benchmark_results(results: dict[str, Any]) -> None:
    """Print results for benchmark evaluation."""
    if RICH_AVAILABLE:
        _print_benchmark_results_rich(results)
    else:
        _print_benchmark_results_plain(results)


def _print_benchmark_results_rich(results: dict[str, Any]) -> None:
    """Print benchmark results using rich."""
    console = Console()
    sequences = results["sequences"]
    aggregate = results["aggregate"]

    console.print("\n[bold]Benchmark Results[/bold]\n")

    if "CLEAR" in aggregate:
        table = Table(title="CLEAR Metrics by Sequence")
        table.add_column("Sequence", style="cyan")
        table.add_column("MOTA", justify="right")
        table.add_column("MOTP", justify="right")
        table.add_column("IDs", justify="right")
        table.add_column("MT", justify="right")
        table.add_column("ML", justify="right")

        for seq_name, seq_results in sorted(sequences.items()):
            clear = seq_results["CLEAR"]
            table.add_row(
                seq_name,
                f"{clear['MOTA']:.1%}",
                f"{clear['MOTP']:.3f}",
                str(clear["IDSW"]),
                str(clear["MT"]),
                str(clear["ML"]),
            )

        # Add aggregate row
        table.add_section()
        agg = aggregate["CLEAR"]
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{agg['MOTA']:.1%}[/bold]",
            f"[bold]{agg['MOTP']:.3f}[/bold]",
            f"[bold]{agg['IDSW']}[/bold]",
            f"[bold]{agg['MT']}[/bold]",
            f"[bold]{agg['ML']}[/bold]",
        )

        console.print(table)


def _print_benchmark_results_plain(results: dict[str, Any]) -> None:
    """Print benchmark results in plain text."""
    sequences = results["sequences"]
    aggregate = results["aggregate"]

    print("\nBenchmark Results")
    print("=" * 60)

    if "CLEAR" in aggregate:
        print(
            f"\n{'Sequence':<20} {'MOTA':>8} {'MOTP':>8} {'IDs':>6} {'MT':>4} {'ML':>4}"
        )
        print("-" * 60)

        for seq_name, seq_results in sorted(sequences.items()):
            clear = seq_results["CLEAR"]
            print(
                f"{seq_name:<20} {clear['MOTA']:>7.1%} {clear['MOTP']:>8.3f} "
                f"{clear['IDSW']:>6} {clear['MT']:>4} {clear['ML']:>4}"
            )

        print("-" * 60)
        agg = aggregate["CLEAR"]
        print(
            f"{'TOTAL':<20} {agg['MOTA']:>7.1%} {agg['MOTP']:>8.3f} "
            f"{agg['IDSW']:>6} {agg['MT']:>4} {agg['ML']:>4}"
        )


def _save_results(results: dict[str, Any], output_path: Path) -> None:
    """Save results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
