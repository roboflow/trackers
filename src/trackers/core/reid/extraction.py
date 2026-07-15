# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Detection embedding extraction for tracker association."""

from __future__ import annotations

import numpy as np
import supervision as sv

from trackers.core.reid.protocols import ReIDEncoder


def extract_detection_embeddings(
    model: ReIDEncoder,
    frame: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    """Extract L2-normalised appearance embeddings for detection boxes.

    Args:
        model: Re-ID encoder used to embed each crop.
        frame: BGR video frame containing the detections.
        boxes: Detection bounding boxes, shape ``(N, 4)`` in ``xyxy`` format.

    Returns:
        Embedding matrix of shape ``(N, D)``.  Returns ``(0, 0)`` when
        ``boxes`` is empty.
    """
    if len(boxes) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return model.extract_features(sv.Detections(xyxy=boxes), frame)
