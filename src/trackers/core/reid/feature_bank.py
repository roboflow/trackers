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

    Every ingested vector is L2-normalised. Non-finite, zero-norm, or
    incompatible-shape inputs are ignored: the bank keeps its previous state
    (or stays uninitialized).

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
        """``True`` once at least one valid embedding has been ingested."""
        return self._feature is not None

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray | None:
        """Return an L2-normalised 1-D vector, or ``None`` when unusable."""
        flat = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if flat.size == 0 or not np.all(np.isfinite(flat)):
            return None
        norm = float(np.linalg.norm(flat))
        if norm < 1e-8:
            return None
        return (flat / norm).astype(np.float32)

    def update(self, embedding: np.ndarray) -> bool:
        """Blend a normalised *embedding* into the stored feature.

        Returns:
            ``True`` when the bank accepted the vector; ``False`` when the
            input was skipped (non-finite, zero norm, incompatible shape, etc.).
        """
        normalized = self.normalize_embedding(embedding)
        if normalized is None:
            return False

        if self._feature is None:
            self._feature = normalized.copy()
            return True

        if self._feature.shape != normalized.shape:
            return False

        blended = self._alpha * self._feature + (1.0 - self._alpha) * normalized
        blended_norm = self.normalize_embedding(blended)
        if blended_norm is None:
            return False
        self._feature = blended_norm
        return True

    def reset(self) -> None:
        """Clear the stored feature."""
        self._feature = None
