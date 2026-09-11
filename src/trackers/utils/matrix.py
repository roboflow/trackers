# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np


def _diagonal_matrix(values: list[float]) -> np.ndarray:
    """Construct a diagonal matrix directly from known values.

    Construct-only: always returns float64, regardless of the input
    values' type. Unlike ``np.diag``, does not extract a diagonal from
    a 2-D input.

    Args:
        values: Diagonal entries, ordered from the top-left corner.

    Returns:
        A ``(len(values), len(values))`` float64 array carrying ``values``
        on the main diagonal and zeros elsewhere.
    """
    size = len(values)
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix.flat[:: size + 1] = values
    return matrix
