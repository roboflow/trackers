# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import supervision as sv


def default_confidences(detections: sv.Detections) -> np.ndarray:
    """Return per-detection confidence, defaulting to all-ones when absent.

    When ``detections.confidence`` is ``None``, every detection is treated as
    fully confident (``1.0``). All four trackers (SORT, ByteTrack, OC-SORT,
    BoT-SORT) share this convention so callers that emit
    ``sv.Detections`` without a ``confidence`` field still produce tracks
    instead of empty results.

    Args:
        detections: ``sv.Detections`` whose ``confidence`` is either a
            ``float32`` array of shape ``(N,)`` or ``None``.

    Returns:
        ``np.ndarray`` of shape ``(len(detections),)`` containing the
        per-detection confidence used by tracker association. The fallback
        array uses ``dtype=np.float32`` to match the ``supervision``
        convention for the confidence field.

    Examples:
        >>> import numpy as np
        >>> import supervision as sv
        >>> from trackers.utils.detections import default_confidences
        >>> det = sv.Detections(xyxy=np.array([[0.0, 0.0, 10.0, 10.0]]))
        >>> default_confidences(det).tolist()
        [1.0]
        >>> det.confidence = np.array([0.5], dtype=np.float32)
        >>> default_confidences(det).tolist()
        [0.5]
    """
    if detections.confidence is not None:
        return detections.confidence
    return np.ones(len(detections), dtype=np.float32)
