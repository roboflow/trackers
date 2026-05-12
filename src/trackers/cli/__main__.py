#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import sys
import warnings

import jsonargparse


def main() -> int:
    """Main entry point for the trackers CLI."""
    # Beta warning
    warnings.warn(
        "The trackers CLI is in beta. APIs may change in future releases.",
        UserWarning,
        stacklevel=2,
    )

    parser = jsonargparse.ArgumentParser(
        prog="trackers",
        description="Command-line tools for multi-object tracking.",
        epilog="For more information, visit: https://github.com/roboflow/trackers",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit.",
    )
    parser.add_argument(
        "--config",
        action="config",
        help="Path to a YAML/JSON config file with default argument values.",
    )

    subparsers = parser.add_subparsers(  # type: ignore[var-annotated]
        dest="command",
        title="commands",
        description="Available commands:",
    )

    # Import and register subcommands
    from trackers.cli.download import add_download_subparser
    from trackers.cli.eval import add_eval_subparser
    from trackers.cli.track import add_track_subparser
    from trackers.cli.tune import add_tune_subparser

    add_download_subparser(subparsers)
    add_eval_subparser(subparsers)
    add_track_subparser(subparsers)
    add_tune_subparser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    if args.version:
        from importlib.metadata import version

        print(f"trackers {version('trackers')}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    # Execute the command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
