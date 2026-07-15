# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Encoder protocols for tracker association and gallery evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import supervision as sv


class ReIDEncoder(Protocol):
    """Protocol for tracker appearance association (``extract_features``)."""

    def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        """Return appearance embeddings for each detection box.

        Args:
            detections: Boxes to embed (``xyxy``).
            frame: BGR frame the detections were produced on.

        Returns:
            Float32 array of shape ``(N, D)``, or ``(0, 0)`` when empty.
        """
        ...


class ReIDPathEncoder(Protocol):
    """Protocol for gallery evaluation (``extract_features_from_paths``)."""

    def extract_features_from_paths(
        self,
        image_paths: Sequence[str],
        *,
        batch_size: int = 64,
        normalize: bool = False,
    ) -> np.ndarray:
        """Embed images read from disk.

        Args:
            image_paths: Paths to crop images.
            batch_size: Images per forward pass.
            normalize: L2-normalise embeddings when ``True``.

        Returns:
            Float32 array of shape ``(N, D)``.
        """
        ...
