# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import warnings

import click

from trackers.cli.download import download_command
from trackers.cli.eval import eval_command
from trackers.cli.track import track_command
from trackers.cli.tune import tune_command


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(package_name="trackers", prog_name="trackers")
def cli() -> None:
    """Command-line tools for multi-object tracking."""


cli.add_command(track_command, "track")
cli.add_command(eval_command, "eval")
cli.add_command(download_command, "download")
cli.add_command(tune_command, "tune")


def main() -> None:
    """Main entry point for the trackers CLI."""
    warnings.warn(
        "The trackers CLI is in beta. APIs may change in future releases.",
        UserWarning,
        stacklevel=2,
    )
    cli()


if __name__ == "__main__":
    main()
