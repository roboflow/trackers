# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Timing payload for one tracklet predict step."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictTiming:
    """Kalman frame step and optional wall-clock elapsed time for one predict.

    Attributes:
        frame_step: Kalman ``F`` / ``Q`` step in **frame units** (``1.0`` = one
            nominal frame at the tuned reference).
        elapsed_seconds: ``None`` in fixed-rate mode; otherwise seconds since the
            last processed timestamp (for ``time_since_update_seconds``).
    """

    frame_step: float
    elapsed_seconds: float | None

    @property
    def skip_predict(self) -> bool:
        """True when predict should be skipped (non-monotonic timestamp)."""
        return self.frame_step <= 0.0

    @property
    def uses_elapsed_time(self) -> bool:
        """True when this step should accumulate elapsed seconds on tracklets."""
        return self.elapsed_seconds is not None


FIXED_RATE_TIMING = PredictTiming(frame_step=1.0, elapsed_seconds=None)
