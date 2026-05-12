# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import cast

import click

from trackers.cli._options import metrics_option, output_option, seqmap_option, threshold_option


@click.command("eval")
@click.option(
    "--gt",
    type=click.Path(path_type=Path),
    default=None,
    metavar="PATH",
    help="Path to ground truth file (MOT format).",
)
@click.option(
    "--tracker",
    "tracker_path",
    type=click.Path(path_type=Path),
    default=None,
    metavar="PATH",
    help="Path to tracker predictions file (MOT format).",
)
@click.option(
    "--gt-dir",
    type=click.Path(path_type=Path),
    default=None,
    metavar="DIR",
    help="Directory containing ground truth files.",
)
@click.option(
    "--tracker-dir",
    type=click.Path(path_type=Path),
    default=None,
    metavar="DIR",
    help="Directory containing tracker prediction files.",
)
@seqmap_option
@metrics_option
@threshold_option
@click.option(
    "--columns", multiple=True, default=(), metavar="COL", help="Metric columns to display. Default: auto-selected."
)
@output_option("Output file for results (JSON format).")
def eval_command(
    gt: Path | None,
    tracker_path: Path | None,
    gt_dir: Path | None,
    tracker_dir: Path | None,
    seqmap: Path | None,
    metrics: tuple[str, ...],
    threshold: float,
    columns: tuple[str, ...],
    output: Path | None,
) -> None:
    """Evaluate tracker predictions against ground truth."""
    single_mode = gt is not None and tracker_path is not None
    benchmark_mode = gt_dir is not None and tracker_dir is not None

    if not single_mode and not benchmark_mode:
        raise click.UsageError("Must specify either --gt/--tracker or --gt-dir/--tracker-dir")

    if single_mode and benchmark_mode:
        raise click.UsageError("Cannot use both single sequence and benchmark mode")

    columns_list: list[str] | None = list(columns) if columns else None
    metrics_list = list(metrics)

    from trackers.eval import evaluate_mot_sequence, evaluate_mot_sequences

    try:
        if single_mode:
            seq_result = evaluate_mot_sequence(
                gt_path=cast(Path, gt),
                tracker_path=cast(Path, tracker_path),
                metrics=metrics_list,
                threshold=threshold,
            )
            print(seq_result.table(columns=columns_list))

            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(seq_result.json())
                print(f"\nResults saved to: {output}")
        else:
            bench_result = evaluate_mot_sequences(
                gt_dir=cast(Path, gt_dir),
                tracker_dir=cast(Path, tracker_dir),
                seqmap=seqmap,
                metrics=metrics_list,
                threshold=threshold,
            )
            print(bench_result.table(columns=columns_list))

            if output:
                bench_result.save(output)
                print(f"\nResults saved to: {output}")

    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    except ValueError as e:
        raise click.ClickException(str(e)) from e
