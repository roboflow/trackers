# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Encoder protocol for ReID association."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import supervision as sv


class ReIDEncoder(Protocol):
    """Appearance encoder used for tracking association.

    Trackers only depend on ``extract_features``. ``reid.ReIDModel`` structurally
    satisfies this protocol, and custom or test encoders may implement it without
    depending on the full model stack.
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
