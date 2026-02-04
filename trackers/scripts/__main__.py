#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
import warnings


def main() -> int:
    """Main entry point for the trackers CLI."""
    # Beta warning
    warnings.warn(
        "The trackers CLI is in beta. APIs may change in future releases.",
        UserWarning,
        stacklevel=2,
    )

    parser = argparse.ArgumentParser(
        prog="trackers",
        description="Command-line tools for multi-object tracking.",
        epilog="For more information, visit: https://github.com/roboflow/trackers",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands:",
    )

    # Import and register subcommands
    from trackers.scripts.eval import add_eval_subparser

    add_eval_subparser(subparsers)

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
