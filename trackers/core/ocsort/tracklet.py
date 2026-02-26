# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.utils.converters import (
    xyxy_to_xcycsr,
)
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCSRStateEstimator,
)


class OCSORTTracklet:
    """Tracklet for OC-SORT tracker with ORU (Observation-centric Re-Update).

    Manages a single tracked object with Kalman filter state estimation.
    Implements OC-SORT specific features: freeze/unfreeze for saving state
    before track is lost, virtual trajectory generation (ORU) for recovering
    lost tracks, and configurable state representation (XCYCSR or XYXY).

    Attributes:
        age: Age of the tracklet in frames.
        kalman_filter: The Kalman filter wrapping the state representation.
        tracker_id: Unique identifier (-1 until track is mature).
        number_of_successful_consecutive_updates: Consecutive successful updates.
        time_since_update: Frames since last observation.
        last_observation: Last observed bounding box `[x1, y1, x2, y2]`.
        previous_to_last_observation: Second-to-last observation for velocity.
        observations: Dict mapping age to observed bbox for delta_t lookback.
        velocity: Normalized direction vector computed with delta_t lookback.
        delta_t: Number of timesteps back to look for velocity estimation.
    """

    count_id: int = 0

    def __init__(
        self,
        initial_bbox: np.ndarray,
        state_estimator_class: type[BaseStateEstimator] = XCYCSRStateEstimator,
        delta_t: int = 3,
    ) -> None:
        """Initialize tracklet with first detection.

        Args:
            initial_bbox: Initial bounding box `[x1, y1, x2, y2]`.
            kalman_filter_class: Kalman filter class to use. Instantiated
                with *initial_bbox*. Defaults to
                `XCYCSRKalmanFilter`.
            delta_t: Number of timesteps back to look for velocity estimation.
                Higher values use observations further in the past to estimate
                motion direction, providing more stable velocity estimates.
        """
        self.age = 0

        # Initialize state estimator (wraps KalmanFilter + state repr)
        self.kalman_filter: BaseStateEstimator = state_estimator_class(initial_bbox)

        # Observation history for ORU and delta_t
        self.delta_t = delta_t
        self.last_observation = initial_bbox
        self.previous_to_last_observation: np.ndarray | None = None
        self.observations: dict[int, np.ndarray] = {}
        self.velocity: np.ndarray | None = None

        # Track ID can be initialized before mature in oc-sort
        # it is assigned if the frame number is less than minimum_consecutive_frames
        self.tracker_id = -1

        # Tracking counters
        self.number_of_successful_consecutive_updates = 0
        self.time_since_update = 0

        # ORU: saved state for freeze/unfreeze
        self._frozen_state: dict | None = None
        self._observed = True

    @classmethod
    def get_next_tracker_id(cls) -> int:
        """Get next available tracker ID."""
        next_id = cls.count_id
        cls.count_id += 1
        return next_id

    def _freeze(self) -> None:
        """Save Kalman filter state before track is lost (ORU mechanism)."""
        self._frozen_state = self.kalman_filter.get_state()

    def _unfreeze(self, new_bbox: np.ndarray) -> None:
        """Restore state and apply virtual trajectory (ORU mechanism).

        Generates linear interpolation between last observation and new
        detection, then re-updates the Kalman filter through this virtual
        trajectory.

        Args:
            new_bbox: New observation bounding box `[x1, y1, x2, y2]`.
        """
        if self._frozen_state is None:
            return

        # Restore to frozen state
        self.kalman_filter.set_state(self._frozen_state)

        time_gap = self.time_since_update
        # this is oc-sort specific
        if isinstance(self.kalman_filter, XCYCSRStateEstimator):
            self._unfreeze_xcycsr(new_bbox, time_gap)
        else:
            self._unfreeze_xyxy(new_bbox, time_gap)

        self._frozen_state = None

    def _unfreeze_xcycsr(self, new_bbox: np.ndarray, time_gap: int) -> None:
        """ORU interpolation for XCYCSR representation.

        Generates time_gap predict+update cycles with virtual observations
        interpolated from the last observation to the new bbox. The interpolation
        factors go from 0 to (time_gap-1)/time_gap. The caller is responsible
        for the final real update at factor 1.0.
        """
        # Convert to (x, y, s, r) format
        last_xcycsr = xyxy_to_xcycsr(self.last_observation)
        new_xcycsr = xyxy_to_xcycsr(new_bbox)

        # Convert s, r back to w, h for interpolation
        x1, y1, s1, r1 = last_xcycsr
        w1 = np.sqrt(s1 * r1)
        h1 = np.sqrt(s1 / r1)

        x2, y2, s2, r2 = new_xcycsr
        w2 = np.sqrt(s2 * r2)
        h2 = np.sqrt(s2 / r2)

        # Linear interpolation deltas
        dx = (x2 - x1) / time_gap
        dy = (y2 - y1) / time_gap
        dw = (w2 - w1) / time_gap
        dh = (h2 - h1) / time_gap

        for i in range(time_gap):
            x = x1 + (i + 1) * dx
            y = y1 + (i + 1) * dy
            w = w1 + (i + 1) * dw
            h = h1 + (i + 1) * dh

            # Convert back to (x, y, s, r)
            s = w * h
            r = w / h
            virtual_obs = np.array([x, y, s, r]).reshape((4, 1))

            self.kalman_filter.kf.update(virtual_obs)
            if i < time_gap - 1:
                self.kalman_filter.kf.predict()

    def _unfreeze_xyxy(self, new_bbox: np.ndarray, time_gap: int) -> None:
        """ORU interpolation for XYXY representation.

        Same pattern as XCYCSR: time_gap predict+update cycles with factors
        0 to (time_gap-1)/time_gap. Caller does the final real update.
        """
        last_xyxy = self.last_observation
        new_xyxy = new_bbox

        # Linear interpolation deltas for each coordinate
        delta = (new_xyxy - last_xyxy) / time_gap

        for i in range(time_gap):
            virtual_obs = (last_xyxy + (i + 1) * delta).reshape((4, 1))

            self.kalman_filter.kf.update(virtual_obs)
            if i < time_gap - 1:
                self.kalman_filter.kf.predict()

    def get_k_previous_obs(self) -> np.ndarray | None:
        """Get observation from delta_t steps ago.

        Looks back up to delta_t timesteps in the observation history.
        Falls back to the most recent observation if none found in the window.

        Returns:
            The observation from delta_t steps ago, or most recent if not found,
            or None if no observations exist.
        """
        if len(self.observations) == 0:
            return None
        for i in range(self.delta_t):
            dt = self.delta_t - i
            if self.age - dt in self.observations:
                return self.observations[self.age - dt]
        max_age = max(self.observations.keys())
        return self.observations[max_age]

    @staticmethod
    def _compute_velocity(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
        """Compute normalized direction vector between two bounding box centers.

        Args:
            bbox1: First bounding box `[x1, y1, x2, y2]`.
            bbox2: Second bounding box `[x1, y1, x2, y2]`.

        Returns:
            Normalized direction vector [dy, dx].
        """
        cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
        cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
        speed = np.array([cy2 - cy1, cx2 - cx1])
        norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
        return speed / norm

    def update(self, bbox: np.ndarray | None) -> None:
        """Update tracklet with new observation.

        Handles ORU: if track was lost and now observed again,
        generates virtual trajectory to smooth the transition.
        Computes velocity using observation from delta_t steps ago.

        Args:
            bbox: Bounding box `[x1, y1, x2, y2]` or None for no observation.
        """
        if bbox is not None:
            # Compute velocity only after the track has been observed at least once
            # (matches original OC-SORT: velocity is None until 2nd match)

            previous_box = self.get_k_previous_obs()
            if previous_box is not None:
                self.velocity = self._compute_velocity(previous_box, bbox)

            # Check if we need to unfreeze (was lost, now observed)
            if not self._observed and self._frozen_state is not None:
                self._unfreeze(bbox)

            # Update KF with the real observation
            # (after ORU this is the final update at the correct time step;
            #  without ORU this is the normal measurement update)
            self.kalman_filter.update(bbox)

            self._observed = True
            self.time_since_update = 0
            self.number_of_successful_consecutive_updates += 1
            self.previous_to_last_observation = self.last_observation
            self.last_observation = bbox
            self.observations[self.age] = bbox
        else:
            # No observation - freeze state if this is first miss
            if self._observed:
                self._freeze()
            self._observed = False
            self.kalman_filter.update(None)

    def predict(self) -> np.ndarray:
        """Predict next bounding box position.

        Returns:
            Predicted bounding box `[x1, y1, x2, y2]`.
        """
        self.kalman_filter.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.number_of_successful_consecutive_updates = 0

        self.time_since_update += 1
        return self.kalman_filter.state_to_bbox()

    def get_state_bbox(self) -> np.ndarray:
        """Get current bounding box estimate from Kalman filter.

        Returns:
            Current bounding box estimate `[x1, y1, x2, y2]`.
        """
        return self.kalman_filter.state_to_bbox()

    def resolve_tracker_id(
        self,
        minimum_consecutive_frames: int,
        frame_count: int,
    ) -> int:
        """Resolve the tracker ID for the current tracklet state.

        Assigns a new unique ID if the tracklet is mature but hasn't been
        assigned one yet. Returns -1 for immature tracklets.

        Args:
            minimum_consecutive_frames: Frames required for track maturity.
            frame_count: Current frame number in tracking process.

        Returns:
            Integer tracker ID, or -1 for immature tracks.
        """
        is_mature = (
            self.number_of_successful_consecutive_updates >= minimum_consecutive_frames
        )
        if frame_count <= minimum_consecutive_frames:
            if self.time_since_update == 0:
                if self.tracker_id == -1:
                    self.tracker_id = OCSORTTracklet.get_next_tracker_id()
                return self.tracker_id
        else:
            if is_mature:
                if self.tracker_id == -1:
                    self.tracker_id = OCSORTTracklet.get_next_tracker_id()
                return self.tracker_id
        return -1
