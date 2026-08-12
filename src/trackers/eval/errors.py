# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Typed exceptions shared across evaluation modules."""


class AggregationIncompatibleError(ValueError):
    """Raised when metric payloads cannot be aggregated as requested."""
