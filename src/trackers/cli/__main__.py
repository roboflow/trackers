#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import sys
import warnings


def main() -> int:
    """Main entry point for the trackers CLI."""
    warnings.warn(
        "The trackers CLI is in beta. APIs may change in future releases.",
        UserWarning,
        stacklevel=2,
    )

    from importlib.metadata import version

    import defopt

    from trackers.cli.download import download
    from trackers.cli.eval import eval_cmd
    from trackers.cli.track import track
    from trackers.cli.tune import tune

    result = defopt.run(
        {"track": track, "eval": eval_cmd, "tune": tune, "download": download},
        argv=sys.argv[1:],
        cli_options="all",
        version=version("trackers"),
        short={"output": "o"},
    )
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
