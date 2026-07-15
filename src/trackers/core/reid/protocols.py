# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Lightweight typing protocols for optional Re-ID integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import supervision as sv


class ReIDEncoder(Protocol):
    """Minimal encoder surface for tracker appearance association.

    Any object implementing :meth:`extract_features` may be passed as
    ``reid_model`` to :class:`~trackers.core.botsort.tracker.BoTSORTTracker`.
    The concrete :class:`~trackers.core.reid.model.ReIDModel` satisfies this
    protocol but is not required for tests or custom encoders.
    """

    def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        """Return L2-normalised embeddings for each detection box."""
        ...


class ReIDPathEncoder(Protocol):
    """Minimal encoder surface for dataset / retrieval evaluation.

    :class:`~trackers.core.reid.eval.evaluator.ReIDEvaluator` accepts any object
    that can embed image paths in batches. :class:`~trackers.core.reid.model.ReIDModel`
    implements this protocol.
    """

    def extract_features_from_paths(
        self,
        image_paths: Sequence[str],
        *,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed images read from disk."""
        ...
