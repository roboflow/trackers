# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np

from trackers.utils.base_tracklet import BaseTracklet
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XYXYStateEstimator,
)


class ByteTrackTracklet(BaseTracklet):
    count_id: int = 0

    def __init__(
        self,
        initial_bbox: np.ndarray,
        state_estimator_class: type[BaseStateEstimator] = XYXYStateEstimator,
    ) -> None:
        super().__init__(initial_bbox, state_estimator_class)
        self._configure_noise()
        # Count initial bbox as first successful update (matches original
        # ByteTrackKalmanBoxTracker behavior where hits started at 1)
        self.number_of_successful_consecutive_updates = 1

    def update(self, bbox: np.ndarray | None) -> None:
        """Update tracklet with new observation or None if missed."""
        if bbox is not None:
            self.state_estimator.update(bbox)
            self.time_since_update = 0
            self.number_of_successful_consecutive_updates += 1
        else:
            self.state_estimator.update(None)
            self.time_since_update += 1
            self.number_of_successful_consecutive_updates = 0

    def predict(self) -> np.ndarray:
        """Predict next bounding box position."""
        self.state_estimator.predict()
        self.age += 1
        return self.state_estimator.state_to_bbox()

    def get_state_bbox(self) -> np.ndarray:
        """Get current bounding box estimate from the filter/state."""
        return self.state_estimator.state_to_bbox()

    def _configure_noise(self) -> None:
        """Configure Kalman filter noise (original ByteTrack tuning)."""
        kf = self.state_estimator.kf
        self.state_estimator.set_kf_covariances(
            R=kf.R * 0.1,
            Q=kf.Q * 0.01,
        )
