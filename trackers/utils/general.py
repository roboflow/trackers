# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Any


def _normalize_list(value: Any | list[Any] | None) -> list[str] | None:
    """Wrap a scalar value in a list, pass lists through, and return None as-is.

    Enum members are converted to their ``.value`` so callers always
    receive plain strings.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [str(getattr(v, "value", v)) for v in value]
    return [str(getattr(value, "value", value))]
