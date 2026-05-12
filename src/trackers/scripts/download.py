# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# Backward-compatibility shim — trackers.scripts is deprecated; use trackers.cli
from trackers.cli.download import _print_available, download_command  # noqa: F401
