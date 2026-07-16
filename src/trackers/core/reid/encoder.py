# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Encoder protocol for gallery evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class ReIDPathEncoder(Protocol):
    """Gallery-evaluation interface for embedding image paths.

    ``ReIDEvaluator`` accepts any object with ``extract_features_from_paths``.
    ``ReIDModel`` implements this protocol.
    """

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
