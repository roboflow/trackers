# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Encoder protocol for ReID association and gallery evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import supervision as sv


class ReIDEncoder(Protocol):
    """Appearance encoder used for tracking association and gallery evaluation.

    ``ReIDModel`` is the concrete encoder (load/save/preprocess plus both
    methods). Trackers use ``extract_features``; gallery eval uses
    ``extract_features_from_paths``. Custom or test encoders may implement this
    protocol without depending on the full model stack.
    """

    def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        """Return appearance embeddings for each detection box.

        Args:
            detections: Boxes to embed (``xyxy``).
            frame: BGR frame the detections were produced on.

        Returns:
            Float32 array of shape ``(N, D)``, or ``(0, 0)`` when empty.
        """
        ...

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
