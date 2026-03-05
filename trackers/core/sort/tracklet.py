# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np

from trackers.utils.base_tracklet import BaseTracklet
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCSRStateEstimator,
)


class SORTTracklet(BaseTracklet):
    count_id: int = 0

    def __init__(
        self,
        initial_bbox: np.ndarray,
        state_estimator_class: type[BaseStateEstimator] = XCYCSRStateEstimator,
    ) -> None:
        super().__init__(initial_bbox, state_estimator_class)
        self._configure_noise()

    def update(self, bbox: np.ndarray | None) -> None:
        """Update tracklet with new observation or None if missed."""
        if bbox is not None:
            self.kalman_filter.update(bbox)
            self.time_since_update = 0
            self.number_of_successful_consecutive_updates += 1
        else:
            self.kalman_filter.update(None)
            self.time_since_update += 1
            self.number_of_successful_consecutive_updates = 0

    def predict(self) -> np.ndarray:
        """Predict next bounding box position."""
        self.kalman_filter.predict()
        self.age += 1
        return self.kalman_filter.state_to_bbox()

    def get_state_bbox(self) -> np.ndarray:
        """Get current bounding box estimate from the filter/state."""
        return self.kalman_filter.state_to_bbox()

    def _configure_noise(self) -> None:
        """Configure Kalman filter noise matrices (OC-SORT paper behaviour)."""
        kf = self.kalman_filter.kf
        R = kf.R
        P = kf.P
        Q = kf.Q
        if isinstance(self.kalman_filter, XCYCSRStateEstimator):
            R[2:, 2:] *= 10.0
            P[4:, 4:] *= 1000.0
            P *= 10.0
            Q[-1, -1] *= 0.01
            Q[4:, 4:] *= 0.01
        else:
            # XYXY: same velocity uncertainty scaling
            P[4:, 4:] *= 1000.0
            P *= 10.0
            Q[4:, 4:] *= 0.01
        self.kalman_filter.set_kf_covariances(R=R, Q=Q, P=P)
