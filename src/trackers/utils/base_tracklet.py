# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod

import numpy as np

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

    @abstractmethod
    def predict(self) -> np.ndarray:
        """Predict next bounding box position and advance missed-frame state.

        Propagates the Kalman filter and increments `time_since_update` (and
        `age`) on every call — matched or unmatched. Subclasses that track
        consecutive-update counters must also reset them here when
        `time_since_update > 0` before incrementing.

        Returns:
            Predicted bounding box `[x1, y1, x2, y2]`.
        """
        pass

    @abstractmethod
    def get_state_bbox(self) -> np.ndarray:
        """Get current bounding box estimate from the filter/state."""
        pass
