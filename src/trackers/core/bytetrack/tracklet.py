# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np

from trackers.utils.base_tracklet import BaseTracklet
from trackers.utils.predict_timing import FIXED_RATE_TIMING, PredictTiming
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XYXYStateEstimator,
)


class ByteTrackTracklet(BaseTracklet):
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

    def update(self, bbox: np.ndarray) -> None:
        """Update tracklet state with a new bounding-box observation.

        Args:
            bbox: Bounding box `[x1, y1, x2, y2]`.
        """
        self.state_estimator.update(bbox)
        self.time_since_update = 0
        self.time_since_update_seconds = 0.0
        self.number_of_successful_consecutive_updates += 1

    def predict(self, timing: PredictTiming = FIXED_RATE_TIMING) -> np.ndarray:
        """Predict next bounding box position and advance missed-frame clock.

        Propagates the Kalman filter and advances `time_since_update`, `age`,
        and the consecutive-update counter. On missed frames (when
        `time_since_update > 0` at call time), resets
        `number_of_successful_consecutive_updates` to 0 so the counter
        reflects only truly consecutive observations.

        Returns:
            Predicted bounding box `[x1, y1, x2, y2]`.
        """
        self.state_estimator.predict(timing.frame_step)

        if self.time_since_update > 0:
            self.number_of_successful_consecutive_updates = 0
        self._advance_miss_clocks(timing)
        return self.state_estimator.state_to_bbox()

    def get_state_bbox(self) -> np.ndarray:
        """Get current bounding box estimate from the filter/state."""
        return self.state_estimator.state_to_bbox()

    def _configure_noise(self) -> None:
        """Configure Kalman filter noise (original ByteTrack tuning)."""
        kf = self.state_estimator.kf
        self.state_estimator.set_kf_covariances(
            measurement_noise=kf.measurement_noise * 0.1,
            process_noise=kf.process_noise * 0.01,
        )
