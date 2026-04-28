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
    XYXYStateEstimator,
)


class SORTTracklet(BaseTracklet):
    count_id: int = 0

    def __init__(
        self,
        initial_bbox: np.ndarray,
        state_estimator_class: type[BaseStateEstimator] = XYXYStateEstimator,
    ) -> None:
        super().__init__(initial_bbox, state_estimator_class)
        self._configure_noise()
        # SORTKalmanBoxTracker behavior where hits started at 1)
        self.number_of_successful_updates = (
            1  # SORT doesn't use number_of_successful_consecutive_updates
        )

    def update(self, bbox: np.ndarray) -> None:
        """Update tracklet state with a new bounding-box observation.

        Args:
            bbox: Bounding box `[x1, y1, x2, y2]`.
        """
        self.state_estimator.update(bbox)
        self.time_since_update = 0
        self.number_of_successful_updates += 1

    def predict(self) -> np.ndarray:
        """Predict next bounding box position and advance missed-frame clock.

        Propagates the Kalman filter and increments `time_since_update` and
        `age`. Called for every live track each frame regardless of match
        status — unmatched tracks advance their clock automatically here
        without any separate miss notification.

        Returns:
            Predicted bounding box `[x1, y1, x2, y2]`.
        """
        self.state_estimator.predict()
        self.time_since_update += 1
        self.age += 1
        return self.state_estimator.state_to_bbox()

    def get_state_bbox(self) -> np.ndarray:
        """Get current bounding box estimate from the filter/state."""
        return self.state_estimator.state_to_bbox()

    def _configure_noise(self) -> None:
        """Configure Kalman filter noise matrices (OC-SORT paper behaviour) and SORT
        behaviour for XYXY coordinates."""
        kf = self.state_estimator.kf
        R = kf.R
        P = kf.P
        Q = kf.Q
        if isinstance(self.state_estimator, XCYCSRStateEstimator):
            R[2:, 2:] *= 10.0
            P[4:, 4:] *= 1000.0
            P *= 10.0
            Q[-1, -1] *= 0.01
            Q[4:, 4:] *= 0.01
        else:
            # Process covariance matrix (Q)
            Q = np.eye(8, dtype=np.float64) * 0.01

            # Measurement covariance (R): noise in detection
            R = np.eye(4, dtype=np.float64) * 0.1

            # Error covariance matrix (P)
            P = np.eye(8, dtype=np.float64)

        self.state_estimator.set_kf_covariances(R=R, Q=Q, P=P)
