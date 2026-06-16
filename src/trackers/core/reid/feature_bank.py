# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Per-track exponential moving average feature bank."""

from __future__ import annotations

import numpy as np


class FeatureBank:
    """Maintains an EMA-smoothed appearance embedding for a single track.

    Each time a track is successfully matched to a detection, the new
    detection embedding is blended into the stored feature using an
    exponential moving average and the result is L2-normalised. This
    smooths out per-frame noise while keeping the representation
    up-to-date with gradual appearance changes.

    The bank is intentionally **pure-numpy** and has no dependency on
    ``torch``, so it adds zero overhead when ``[reid]`` is installed but
    the :class:`~trackers.core.reid.model.ReIDModel` is not in use.

    Args:
        alpha: EMA momentum.  ``alpha=1.0`` keeps only the most recent
            embedding (no smoothing); ``alpha=0.0`` keeps only the very
            first embedding (frozen).  Default ``0.9`` is a good starting
            point for MOT scenes.

    Examples:
        >>> import numpy as np
        >>> bank = FeatureBank(alpha=0.9)
        >>> bank.is_initialized
        False
        >>> e1 = np.array([1.0, 0.0, 0.0])
        >>> bank.update(e1)
        >>> bank.is_initialized
        True
        >>> np.allclose(bank.feature, e1)
        True
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
        """Blend *embedding* into the stored feature using EMA.

        The result is always L2-normalised so that cosine similarity
        remains a valid metric.

        Args:
            embedding: L2-normalised embedding vector, shape ``(D,)``.
                Must be the same dimensionality as all previous updates.
        """
        if self._feature is None:
            self._feature = embedding.copy()
        else:
            self._feature = self._alpha * self._feature + (1.0 - self._alpha) * embedding
            norm = float(np.linalg.norm(self._feature))
            if norm > 1e-8:
                self._feature /= norm

    def reset(self) -> None:
        """Clear the stored feature, returning the bank to uninitialised state.

        Examples:
            >>> import numpy as np
            >>> bank = FeatureBank()
            >>> bank.update(np.array([1.0, 0.0]))
            >>> bank.reset()
            >>> bank.is_initialized
            False
        """
        self._feature = None
