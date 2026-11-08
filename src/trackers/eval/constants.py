# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

# Epsilon for floating point comparisons.
# Must match TrackEval exactly for numerical parity.
# References:
#   - trackeval/metrics/clear.py:82,86 (threshold comparisons)
#   - trackeval/metrics/hota.py:59,92 (similarity masking)
#   - trackeval/datasets/_base_dataset.py:274-285 (IoU computation)
EPS = np.finfo("float").eps


def is_zero_based_contiguous(unique_ids: np.ndarray) -> bool:
    """Check whether sorted unique IDs are already usable as array indices.

    `np.unique` returns sorted, distinct values. When those values are
    integer-dtype and span exactly `0..len(unique_ids)-1`, each per-frame ID
    already equals its own index into `unique_ids`, so callers can skip
    `np.searchsorted` and use the raw ID array as an index array directly.

    Args:
        unique_ids: Sorted, distinct ID array, e.g. from `np.unique`. Must be
            non-empty — callers check this after the empty-sequence early
            returns, where `unique_ids[0]`/`unique_ids[-1]` are safe to read.

    Returns:
        `True` if `unique_ids` is integer-dtype and equals `{0, ..., n-1}`.

    Examples:
        >>> import numpy as np
        >>> from trackers.eval.constants import is_zero_based_contiguous
        >>> is_zero_based_contiguous(np.array([0, 1, 2]))
        True
        >>> is_zero_based_contiguous(np.array([0, 1, 3]))
        False
        >>> is_zero_based_contiguous(np.array([0.0, 1.0, 2.0]))
        False
    """
    return bool(unique_ids.dtype.kind in "iu" and unique_ids[0] == 0 and unique_ids[-1] == len(unique_ids) - 1)
