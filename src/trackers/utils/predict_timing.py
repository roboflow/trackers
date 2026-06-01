# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Timing for one Kalman predict step.

Stores how large the predict step is (in frame units) and how many seconds passed since the
last step. Two fields are used because Kalman ``F``/``Q`` scale in frame units,
while timestamped updates also need real elapsed time between calls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictTiming:
    """Predict step size and elapsed time since the last step."""

    frame_step: float
    elapsed_seconds: float | None

    @property
    def skip_predict(self) -> bool:
        """Return whether predict should be skipped."""
        return self.frame_step <= 0.0

    @property
    def uses_elapsed_time(self) -> bool:
        """Return whether elapsed wall-clock time is available."""
        return self.elapsed_seconds is not None


# One frame per step; elapsed time not tracked.
FIXED_RATE_TIMING = PredictTiming(frame_step=1.0, elapsed_seconds=None)
