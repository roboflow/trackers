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
    """Per-track EMA appearance embedding, kept on the unit hypersphere.

    Matches BoT-SORT's ``STrack.update_features``
    (https://github.com/NirAharon/BoT-SORT/blob/main/tracker/bot_sort.py):
    L2-normalize the incoming embedding, blend with EMA momentum ``alpha``,
    then L2-normalize the result again so the stored template stays unit-norm.

    That is tracker association policy, not the standalone ``reid`` package.
    ``reid.ReIDModel.extract_features`` returns raw embeddings; gallery eval in
    ``reid`` L2-normalizes only when computing cosine distance. Here the bank
    normalizes on update so EMA is taken on the unit sphere, as in BoT-SORT.

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
        """Current stored unit embedding, or ``None`` if never updated."""
        return None if self._feature is None else self._feature.copy()

    @property
    def is_initialized(self) -> bool:
        """``True`` after the first update."""
        return self._feature is not None

    def update(self, embedding: np.ndarray) -> None:
        """Blend an L2-normalized embedding into the stored unit feature."""
        cleaned = _l2_normalize(_require_embedding(embedding))

        if self._feature is None:
            self._feature = cleaned
            return

        if self._feature.shape != cleaned.shape:
            raise ValueError(
                f"embedding shape {cleaned.shape} does not match stored feature shape {self._feature.shape}"
            )

        blended = self._alpha * self._feature + (1.0 - self._alpha) * cleaned
        self._feature = _l2_normalize(blended)

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


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return a unit-norm float32 vector (zero vectors are returned unchanged)."""
    norm = float(np.linalg.norm(vec))
    return (vec / max(norm, _NORM_EPS)).astype(np.float32)
