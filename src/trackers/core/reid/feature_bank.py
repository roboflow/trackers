# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Per-track exponential moving average feature bank."""

from __future__ import annotations

import numpy as np

_NORM_EPS = 1e-12


class FeatureBank:
    """EMA-smoothed appearance embedding for a single track.

    Every ingested vector is L2-normalised (``eps`` floor on the norm, same
    idea as ``torch.nn.functional.normalize``).

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
        return None if self._feature is None else self._feature.copy()

    @property
    def is_initialized(self) -> bool:
        """``True`` once at least one embedding has been ingested."""
        return self._feature is not None

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """Return an L2-normalised 1-D vector."""
        flat = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            raise ValueError("embedding must be non-empty")
        if not np.all(np.isfinite(flat)):
            raise ValueError("embedding must contain only finite values")
        norm = float(np.linalg.norm(flat))
        return (flat / max(norm, _NORM_EPS)).astype(np.float32)

    def update(self, embedding: np.ndarray) -> None:
        """Blend a normalised *embedding* into the stored feature."""
        normalized = self.normalize_embedding(embedding)

        if self._feature is None:
            self._feature = normalized.copy()
            return

        if self._feature.shape != normalized.shape:
            raise ValueError(
                f"embedding shape {normalized.shape} does not match "
                f"stored feature shape {self._feature.shape}"
            )

        blended = self._alpha * self._feature + (1.0 - self._alpha) * normalized
        self._feature = self.normalize_embedding(blended)

    def reset(self) -> None:
        """Clear the stored feature."""
        self._feature = None
