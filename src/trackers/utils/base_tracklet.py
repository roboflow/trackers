# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod

import numpy as np

from trackers.utils.predict_timing import FIXED_RATE_TIMING, PredictTiming
from trackers.utils.state_representations import BaseStateEstimator


class BaseTracklet(ABC):
    """
    Abstract base class for all tracker-specific tracklets.
    Provides common interface and attributes for tracklet management.
    """

    count_id: int = 0

    def __init__(self, bbox: np.ndarray, state_estimator_class: type[BaseStateEstimator]) -> None:
        self.age = 0
        self.state_estimator: BaseStateEstimator = state_estimator_class(bbox)

        self.tracker_id = -1
        self.time_since_update = 0
        self.time_since_update_seconds: float = 0.0
        self.number_of_successful_consecutive_updates = 0

    @classmethod
    def get_next_tracker_id(cls) -> int:
        next_id = cls.count_id
        cls.count_id += 1
        return next_id

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
        """Advance frame and optional wall-clock miss counters after predict."""
        self.time_since_update += 1
        if timing.elapsed_seconds is not None:
            self.time_since_update_seconds += timing.elapsed_seconds
        self.age += 1

    @abstractmethod
    def predict(self, timing: PredictTiming = FIXED_RATE_TIMING) -> np.ndarray:
        """Predict next bounding box position and advance missed-frame state.

        Propagates the Kalman filter and increments `time_since_update` (and
        `age`) on every call — matched or unmatched.

        Args:
            timing: Kalman frame step and optional elapsed seconds for this
                update. Defaults to one fixed-rate frame step.

        Returns:
            Predicted bounding box `[x1, y1, x2, y2]`.
        """
        pass

    @abstractmethod
    def get_state_bbox(self) -> np.ndarray:
        """Get current bounding box estimate from the filter/state."""
        pass
