# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Per-track exponential moving average feature bank."""

from __future__ import annotations

import numpy as np

from trackers.core.reid.appearance import _l2_normalize


class FeatureBank:
    """Per-track EMA unit embedding (L2 before and after blend).

    Follows the per-track feature update in BoT-SORT
    (https://github.com/NirAharon/BoT-SORT/blob/main/tracker/bot_sort.py).
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

    def update(self, embedding: np.ndarray, *, normalized: bool = False) -> None:
        """Blend an embedding into the stored unit feature.

        Args:
            embedding: Raw embedding, or a unit embedding when ``normalized`` is
                true.
            normalized: Skip input normalization for validated output from
                ``extract_detection_embeddings``.

        Raises:
            ValueError: If the embedding is empty, non-finite, or changes shape.
        """
        cleaned = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if cleaned.size == 0:
            raise ValueError("embedding must be non-empty")
        if not normalized:
            cleaned = _l2_normalize(cleaned)

        if self._feature is None:
            self._feature = cleaned.copy()
            return

        if self._feature.shape != cleaned.shape:
            raise ValueError(
                f"embedding shape {cleaned.shape} does not match stored feature shape {self._feature.shape}"
            )

        blended = self._alpha * self._feature + (1.0 - self._alpha) * cleaned
        self._feature = _l2_normalize(blended)
