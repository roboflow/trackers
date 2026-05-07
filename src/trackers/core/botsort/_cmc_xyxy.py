# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared CMC helper for XYXY-state Kalman filters."""

from __future__ import annotations

import numpy as np


def _xyxy_corner_min_max(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform four box corners via R/t and return the enclosing min/max.

    Works for both batched inputs (shape ``(N,)``) and scalar inputs (shape
    ``()`` or plain ``float``). The four corners are ``(x1,y1)``, ``(x2,y1)``,
    ``(x2,y2)``, ``(x1,y2)``; after the affine transform the axis-aligned
    bounding box of the transformed corners is returned.

    Args:
        x1: Left edge coordinate(s).
        y1: Top edge coordinate(s).
        x2: Right edge coordinate(s).
        y2: Bottom edge coordinate(s).
        R: 2x2 rotation/shear sub-matrix of the affine transform.
        t: Optional 2-element translation vector.

    Returns:
        Tuple ``(new_x1, new_y1, new_x2, new_y2)`` — per-axis min and max of
        the four transformed corners.
    """
    corners = np.stack(
        [
            np.stack([x1, y1], axis=-1),
            np.stack([x2, y1], axis=-1),
            np.stack([x2, y2], axis=-1),
            np.stack([x1, y2], axis=-1),
        ],
        axis=-2,
    )  # (..., 4, 2)
    out = corners @ R.T
    if t is not None:
        out = out + t
    lo = out.min(axis=-2)
    hi = out.max(axis=-2)
    return lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1]
