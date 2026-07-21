# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Per-track exponential moving average feature bank."""

from __future__ import annotations

import numpy as np


class FeatureBank:
    """Per-track EMA appearance embedding.

    Args:
        alpha: EMA momentum in ``[0, 1]``.
    """

    def __init__(self, alpha: float = 0.9) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha
        self._feature: np.ndarray | None = None

    @property
    def feature(self) -> np.ndarray | None:
        """Current stored embedding, or ``None`` if never updated."""
        return None if self._feature is None else self._feature.copy()

    @property
    def is_initialized(self) -> bool:
        """``True`` after the first update."""
        return self._feature is not None

    def update(self, embedding: np.ndarray) -> None:
        """Blend an embedding into the stored feature."""
        cleaned = _require_embedding(embedding)

        if self._feature is None:
            self._feature = cleaned.copy()
            return

        if self._feature.shape != cleaned.shape:
            raise ValueError(
                f"embedding shape {cleaned.shape} does not match stored feature shape {self._feature.shape}"
            )

        self._feature = (self._alpha * self._feature + (1.0 - self._alpha) * cleaned).astype(np.float32)

    def reset(self) -> None:
        """Clear the stored feature."""
        self._feature = None


def _require_embedding(embedding: np.ndarray) -> np.ndarray:
    """Return a finite 1-D float32 vector."""
    flat = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise ValueError("embedding must be non-empty")
    if not np.all(np.isfinite(flat)):
        raise ValueError("embedding must contain only finite values")
    return flat
