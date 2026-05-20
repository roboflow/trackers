#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Command-line entry point for the trackers package."""

from __future__ import annotations

import sys
import warnings

from jsonargparse import CLI, ActionYesNo, ArgumentParser

from trackers.cli.download import download
from trackers.cli.eval import eval_cmd
from trackers.cli.track import track
from trackers.cli.tune import tune


class _BoolFlagParser(ArgumentParser):
    """Render plain ``bool`` fields as ``--flag`` / ``--no-flag`` pairs."""

    def add_argument(self, *args, **kwargs):  # type: ignore[override]
        if kwargs.get("type") is bool:
            kwargs.pop("type")
            kwargs["action"] = ActionYesNo(yes_prefix="", no_prefix="no-")
        return super().add_argument(*args, **kwargs)


def main() -> int:
    """Dispatch to track / eval / tune / download via jsonargparse CLI."""
    warnings.warn(
        "The trackers CLI is in beta. APIs may change in future releases.",
        UserWarning,
        stacklevel=2,
    )
    rc = CLI(
        {"track": track, "eval": eval_cmd, "tune": tune, "download": download},
        as_positional=False,
        prog="trackers",
        description="Command-line tools for multi-object tracking.",
        parser_class=_BoolFlagParser,
    )
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
