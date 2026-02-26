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
