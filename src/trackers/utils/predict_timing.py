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
    """Predict step size and elapsed time since the last step.

    Attributes:
        frame_step (float): Elapsed time in frame units for this predict step.
            Used to scale Kalman ``F`` and ``Q``. ``1.0`` means one nominal
            frame; larger values indicate a gap between updates.
        elapsed_seconds (float | None): Wall-clock seconds elapsed since the
            last ``update()`` call. ``None`` in fixed-rate mode (no timestamp
            was passed); non-``None`` in dynamic-rate mode.
        skip_update (bool): When ``True`` the caller should skip the entire
            measurement update step (e.g. backwards or non-finite timestamp).
    """

    frame_step: float
    elapsed_seconds: float | None
    skip_update: bool = False

    @property
    def skip_predict(self) -> bool:
        """Return True when the Kalman predict step should be skipped.

        Returns:
            ``True`` if ``frame_step <= 0.0`` (duplicate or invalid timestamp),
            ``False`` otherwise.
        """
        return self.frame_step <= 0.0

    @property
    def uses_elapsed_time(self) -> bool:
        """Return True when wall-clock elapsed time is available.

        Returns:
            ``True`` if ``elapsed_seconds`` is not ``None`` (dynamic-rate mode),
            ``False`` in fixed-rate mode.
        """
        return self.elapsed_seconds is not None


# One frame per step; elapsed time not tracked.
FIXED_RATE_TIMING = PredictTiming(frame_step=1.0, elapsed_seconds=None)
