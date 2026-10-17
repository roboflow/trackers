# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tracker-agnostic configuration for appearance ReID association."""

from __future__ import annotations

import math
from dataclasses import dataclass

from trackers.core.reid.fusion import ReidFusionMethod


@dataclass(frozen=True)
class ReIDConfig:
    """Settings for appearance ReID association.

    Holds only the values that steer fusion, so any tracker that accepts a ReID
    encoder can reuse it. The encoder itself is a separate argument, since it is
    an object rather than a setting.

    Attributes:
        ema_alpha: EMA momentum for track appearance features. Default ``0.9``.
        appearance_threshold: Maximum appearance distance used by ``"botsort"``.
            Default ``0.25`` (BoT-SORT ``appearance_thresh``). Ignored by
            ``"adaptive"``.
        proximity_threshold: IoU distance gate applied before appearance is
            used. Default ``0.5`` (BoT-SORT ``proximity_thresh``; requires
            ``IoU >= 1 - proximity_threshold``).
        fusion: How appearance combines with geometry. ``"botsort"`` takes the
            minimum cost; ``"adaptive"`` adds weighted appearance similarity.
        appearance_weight: Base appearance weight for adaptive fusion.
        adaptive_weight_cap: Maximum adaptive appearance bonus.
        appearance_floor: Minimum cosine similarity for appearance to
            contribute with ``"adaptive"``. Default ``0.0`` (disabled).

    Raises:
        ValueError: If a setting is outside its valid range.

    Example:
        >>> ReIDConfig(appearance_threshold=0.3).appearance_threshold
        0.3
    """

    ema_alpha: float = 0.9
    appearance_threshold: float = 0.25
    proximity_threshold: float = 0.5
    fusion: ReidFusionMethod = "botsort"
    appearance_weight: float = 0.75
    adaptive_weight_cap: float = 0.5
    appearance_floor: float = 0.0

    def __post_init__(self) -> None:
        for name in ("ema_alpha", "appearance_threshold", "proximity_threshold", "appearance_floor"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.fusion not in ("botsort", "adaptive"):
            raise ValueError(f"Unknown fusion {self.fusion!r}. Valid options are: ('botsort', 'adaptive').")
        for name in ("appearance_weight", "adaptive_weight_cap"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
