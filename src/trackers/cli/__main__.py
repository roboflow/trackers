#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Command-line entry point for the trackers package.

The parsing conventions every command shares live in
:mod:`trackers.cli._parser`, and the transitional spellings kept alive for one
release cycle live in :mod:`trackers.cli._legacy`. Both are re-exported here
because this module is the CLI's public-facing name; a subcommand module must
import them from their own modules instead, since importing this one pulls in
every subcommand.
"""

from __future__ import annotations

import sys
import warnings
from importlib.metadata import version

from jsonargparse import CLI

from trackers.cli._legacy import _translate_legacy_args
from trackers.cli._parser import _CLIParser
from trackers.cli._parser import _normalise_option as _normalise_option  # re-export
from trackers.cli.download import download_command
from trackers.cli.eval import eval_command
from trackers.cli.mcbyte import benchmark_command
from trackers.cli.track import track_command
from trackers.cli.tune import tune_command

__all__ = ["main"]


def main() -> int:
    """Dispatch to track / eval / tune / download / mcbyte via jsonargparse CLI."""
    warnings.warn(
        "The trackers CLI is in beta. APIs may change in future releases.",
        UserWarning,
        stacklevel=2,
    )
    try:
        args = _translate_legacy_args(sys.argv[1:])
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    if args == ["--version"]:
        print(f"trackers {version('trackers')}")
        return 0
    rc = CLI(
        {
            "track": track_command,
            "eval": eval_command,
            "tune": tune_command,
            "download": download_command,
            "mcbyte": benchmark_command,
        },
        args=args,
        as_positional=False,
        prog="trackers",
        description="Command-line tools for multi-object tracking.",
        parser_class=_CLIParser,
    )
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
