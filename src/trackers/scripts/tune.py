# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# Backward-compatibility shim — trackers.scripts is deprecated; use trackers.cli
from trackers.cli.tune import tune, tune_command  # noqa: F401
