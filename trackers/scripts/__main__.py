#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import warnings


def main() -> None:
    """Main entry point for the trackers CLI."""
    warnings.warn(
        "The trackers CLI is in beta. APIs may change in future releases.",
        UserWarning,
        stacklevel=2,
    )

    from jsonargparse import auto_cli, set_parsing_settings

    from trackers.scripts.eval import evaluate
    from trackers.scripts.track import track

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli([track, evaluate], as_positional=False)


if __name__ == "__main__":
    main()
