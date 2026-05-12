# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import click

F = TypeVar("F", bound=Callable)

METRIC_CHOICES: list[str] = ["CLEAR", "HOTA", "Identity"]


def metrics_option(f: F) -> F:
    """Shared --metrics option for eval and tune commands.

    Examples:
        >>> import click
        >>> @click.command()
        ... @metrics_option
        ... def cmd(metrics): pass
        >>> cmd.params[0].name
        'metrics'
    """
    return click.option(
        "--metrics",
        multiple=True,
        default=("CLEAR",),
        type=click.Choice(METRIC_CHOICES),
        help="Metrics to compute. Repeat flag for multiple: --metrics CLEAR --metrics HOTA. Default: CLEAR.",
    )(f)


def threshold_option(f: F) -> F:
    """Shared --threshold option for eval and tune commands.

    Examples:
        >>> import click
        >>> @click.command()
        ... @threshold_option
        ... def cmd(threshold): pass
        >>> cmd.params[0].name
        'threshold'
    """
    return click.option(
        "--threshold",
        type=float,
        default=0.5,
        help="IoU threshold for CLEAR and Identity matching. Default: 0.5",
    )(f)


def seqmap_option(f: F) -> F:
    """Shared --seqmap option for eval and tune commands.

    Examples:
        >>> import click
        >>> @click.command()
        ... @seqmap_option
        ... def cmd(seqmap): pass
        >>> cmd.params[0].name
        'seqmap'
    """
    return click.option(
        "--seqmap",
        type=click.Path(path_type=Path),
        default=None,
        metavar="PATH",
        help="Sequence map file listing sequences to evaluate.",
    )(f)


def output_option(help_text: str = "Output file path.") -> Callable[[F], F]:
    """Shared -o/--output option factory.

    Args:
        help_text: Help text for the option.

    Returns:
        Decorator that adds the output option to a command.

    Examples:
        >>> import click
        >>> @click.command()
        ... @output_option("Output JSON file.")
        ... def cmd(output): pass
        >>> cmd.params[0].name
        'output'
    """

    def decorator(f: F) -> F:
        return click.option(
            "-o",
            "--output",
            type=click.Path(path_type=Path),
            default=None,
            metavar="PATH",
            help=help_text,
        )(f)

    return decorator
