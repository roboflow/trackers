# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tracker-agnostic configuration for appearance ReID association."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReIDConfig:
    """Settings for appearance ReID association.

    Holds only the values that steer fusion, so any tracker that accepts a ReID
    encoder can reuse it. The encoder itself is a separate argument, since it is
    an object rather than a setting.

    Attributes:
        ema_alpha: EMA momentum for track appearance features. Default ``0.9``.
        appearance_threshold: Appearance distance gate. Drops the appearance term
            when the halved cosine distance ``0.5 * (1 - cos_sim)`` exceeds this
            value, leaving the pair scored on geometry alone. Default ``0.25``
            (BoT-SORT ``appearance_thresh``).
        proximity_threshold: IoU distance gate applied before appearance is
            used. Default ``0.5`` (BoT-SORT ``proximity_thresh``; requires
            ``IoU >= 1 - proximity_threshold``).

    Raises:
        ValueError: If any field falls outside ``[0, 1]``.

    Example:
        >>> ReIDConfig(appearance_threshold=0.3)
        ReIDConfig(ema_alpha=0.9, appearance_threshold=0.3, proximity_threshold=0.5)
    """

    ema_alpha: float = 0.9
    appearance_threshold: float = 0.25
    proximity_threshold: float = 0.5

    def __post_init__(self) -> None:
        for name in ("ema_alpha", "appearance_threshold", "proximity_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
