# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from trackers.utils.predict_timing import FIXED_RATE_TIMING, PredictTiming
from trackers.utils.state_representations import BaseStateEstimator


class BaseTracklet(ABC):
    """
    Abstract base class for all tracker-specific tracklets.
    Provides common interface and attributes for tracklet management.
    """

    def __init__(self, bbox: np.ndarray, state_estimator_class: type[BaseStateEstimator]) -> None:
        self.age = 0
        self.state_estimator: BaseStateEstimator = state_estimator_class(bbox)

        self.tracker_id = -1
        self.time_since_update = 0
        self.time_since_update_seconds: float = 0.0
        self.number_of_successful_consecutive_updates = 0

    @abstractmethod
    def update(self, bbox: np.ndarray) -> None:
        """Update tracklet state with a new bounding-box observation.

        Called only when the track is matched to a detection. Missed frames
        are handled exclusively by `predict()` — `None` is not accepted.

        Args:
            bbox: Bounding box `[x1, y1, x2, y2]`.
        """
        pass

    def _advance_miss_clocks(self, timing: PredictTiming) -> None:
        """Advance miss counters by one step.

        When ``elapsed_seconds`` is ``None`` (fixed-rate mode), the seconds
        counter is reset to zero so stale values from a prior timestamp-mode
        stretch cannot influence later seconds-budget pruning.
        """
        self.time_since_update += 1
        if timing.elapsed_seconds is not None:
            self.time_since_update_seconds += timing.elapsed_seconds
        else:
            self.time_since_update_seconds = 0.0
        self.age += 1

    @staticmethod
    def within_lost_track_budget(
        tracklet: BaseTracklet,
        *,
        maximum_frames_without_update: int,
        maximum_time_without_update: float | None = None,
    ) -> bool:
        """Return whether a tracklet is still within its lost-track budget.

        When ``maximum_time_without_update`` is provided (dynamic-rate mode),
        the check uses ``tracklet.time_since_update_seconds``; otherwise it
        falls back to the frame count ``tracklet.time_since_update``.

        Args:
            tracklet: The tracklet to evaluate.
            maximum_frames_without_update: Maximum number of frames a track may
                go unmatched before being pruned (used in fixed-rate mode).
            maximum_time_without_update: Maximum wall-clock seconds a track may
                go unmatched before being pruned. ``None`` means no time budget
                is applied and the frame count is used instead.

        Returns:
            ``True`` if the tracklet is within its allowed budget, ``False`` if
            it should be pruned.
        """
        if maximum_time_without_update is not None:
            return tracklet.time_since_update_seconds < maximum_time_without_update
        return tracklet.time_since_update < maximum_frames_without_update

    @abstractmethod
    def predict(self, timing: PredictTiming = FIXED_RATE_TIMING) -> np.ndarray:
        """Predict next bounding box position and advance missed-frame state.

        Propagates the Kalman filter and increments ``time_since_update`` (and
        ``age``) on every call — matched or unmatched.

        Args:
            timing: ``PredictTiming`` carrying ``frame_step`` (Kalman step size
                in frame units) and ``elapsed_seconds`` (wall-clock gap since
                the last update, or ``None`` in fixed-rate mode).

        Returns:
            Predicted bounding box ``[x1, y1, x2, y2]``.
        """
        pass

    @abstractmethod
    def get_state_bbox(self) -> np.ndarray:
        """Get current bounding box estimate from the filter/state."""
        pass
