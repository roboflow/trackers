#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# Backward-compat shim — use trackers.cli.track instead.
from trackers.cli.track import (  # noqa: F401
    _format_labels,
    _init_annotators,
    _init_model,
    _init_tracker,
    _resolve_class_filter,
    _resolve_track_id_filter,
    _run_frameless,
    _run_model,
    _run_with_source,
    track,
)
