# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Per-track exponential moving average feature bank."""

from __future__ import annotations

import numpy as np


class FeatureBank:
    """EMA-smoothed appearance embedding for a single track.

    Args:
        alpha: EMA momentum in ``[0, 1]`` (``0.9`` default).
    """

    def __init__(self, alpha: float = 0.9) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha
        self._feature: np.ndarray | None = None

    @property
    def feature(self) -> np.ndarray | None:
        """Current L2-normalised embedding, or ``None`` if never updated."""
        return self._feature

    @property
    def is_initialized(self) -> bool:
        """``True`` once at least one embedding has been ingested."""
        return self._feature is not None

    def update(self, embedding: np.ndarray) -> None:
        """Blend *embedding* into the stored feature (L2-normalised)."""
        if self._feature is None:
            self._feature = embedding.copy()
        else:
            self._feature = self._alpha * self._feature + (1.0 - self._alpha) * embedding
            norm = float(np.linalg.norm(self._feature))
            if norm > 1e-8:
                self._feature /= norm

    def reset(self) -> None:
        """Clear the stored feature."""
        self._feature = None
