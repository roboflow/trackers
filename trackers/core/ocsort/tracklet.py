from __future__ import annotations

from enum import Enum

import numpy as np

from trackers.utils.converters import (
    xcycsr_to_xyxy,
    xyxy_to_xcycsr,
)
from trackers.utils.kalman_filter import KalmanFilter


class StateRepresentation(Enum):
    """Kalman filter state representation for bounding boxes.

    XCYCSR: Center-based (x_center, y_center, scale, aspect_ratio, vx, vy, vs)
        - 7 state variables, aspect ratio is constant (no velocity)
        - Used in original SORT/OC-SORT papers

    XYXY: Corner-based (x1, y1, x2, y2, vx1, vy1, vx2, vy2)
        - 8 state variables, all coordinates have velocities
        - More direct representation, potentially better for non-rigid objects
    """

    XCYCSR = "xcycsr"
    XYXY = "xyxy"


class OCSORTTracklet:
    """Tracklet for OC-SORT tracker with ORU (Observation-centric Re-Update).

    Manages a single tracked object with Kalman filter state estimation.
    Implements OC-SORT specific features:
    - Freeze/unfreeze for saving state before track is lost
    - Virtual trajectory generation (ORU) for recovering lost tracks
    - Configurable state representation (XCYCSR or XYXY)

    Attributes:
        age: Age of the tracklet in frames.
        kalman_filter: The Kalman filter instance for state estimation.
        tracker_id: Unique identifier (-1 until track is mature).
        number_of_successful_consecutive_updates: Consecutive successful updates.
        time_since_update: Frames since last observation.
        last_observation: Last observed bounding box [x1, y1, x2, y2].
        previous_to_last_observation: Second-to-last observation for velocity.
        state_repr: The state representation being used.
    """

    count_id: int = 0

    def __init__(
        self,
        initial_bbox: np.ndarray,
        state_repr: StateRepresentation = StateRepresentation.XCYCSR,
    ) -> None:
        """Initialize tracklet with first detection.

        Args:
            initial_bbox: Initial bounding box [x1, y1, x2, y2].
            state_repr: State representation to use (XCYCSR or XYXY).
        """
        self.age = 0
        self.state_repr = state_repr

        # Initialize Kalman filter based on state representation
        if state_repr == StateRepresentation.XCYCSR:
            self._init_xcycsr_filter(initial_bbox)
        else:
            self._init_xyxy_filter(initial_bbox)

        # Observation history for ORU
        self.last_observation = initial_bbox
        self.previous_to_last_observation: np.ndarray | None = None

        # Track ID can be initialized before mature in oc-sort
        # it is assigned if the frame number is less than minimum_consecutive_frames
        self.tracker_id = -1

        # Tracking counters
        self.number_of_successful_consecutive_updates = 1
        self.time_since_update = 0

        # ORU: saved state for freeze/unfreeze
        self._frozen_state: dict | None = None
        self._observed = True

    def _init_xcycsr_filter(self, initial_bbox: np.ndarray) -> None:
        """Initialize Kalman filter with XCYCSR state representation.

        State: [x_center, y_center, scale, aspect_ratio, vx, vy, vs]
        Measurement: [x_center, y_center, scale, aspect_ratio]
        """
        self.kalman_filter = KalmanFilter(dim_x=7, dim_z=4)

        # State transition: constant velocity model
        self.kalman_filter.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],  # aspect ratio: no velocity
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Measurement function: observe (x, y, s, r) from state
        self.kalman_filter.H = np.eye(4, 7, dtype=np.float64)

        # Noise tuning (from OC-SORT paper)
        self.kalman_filter.R[2:, 2:] *= 10.0
        self.kalman_filter.P[4:, 4:] *= 1000.0  # high uncertainty for velocities
        self.kalman_filter.P *= 10.0
        self.kalman_filter.Q[-1, -1] *= 0.01
        self.kalman_filter.Q[4:, 4:] *= 0.01

        # Initialize state with first observation
        self.kalman_filter.x[:4] = xyxy_to_xcycsr(initial_bbox).reshape((4, 1))

    def _init_xyxy_filter(self, initial_bbox: np.ndarray) -> None:
        """Initialize Kalman filter with XYXY state representation.

        State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        Measurement: [x1, y1, x2, y2]
        """
        self.kalman_filter = KalmanFilter(dim_x=8, dim_z=4)

        # State transition: constant velocity model for all coordinates
        self.kalman_filter.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],  # x1 += vx1
                [0, 1, 0, 0, 0, 1, 0, 0],  # y1 += vy1
                [0, 0, 1, 0, 0, 0, 1, 0],  # x2 += vx2
                [0, 0, 0, 1, 0, 0, 0, 1],  # y2 += vy2
                [0, 0, 0, 0, 1, 0, 0, 0],  # vx1
                [0, 0, 0, 0, 0, 1, 0, 0],  # vy1
                [0, 0, 0, 0, 0, 0, 1, 0],  # vx2
                [0, 0, 0, 0, 0, 0, 0, 1],  # vy2
            ],
            dtype=np.float64,
        )

        # Measurement function: observe (x1, y1, x2, y2) from state
        self.kalman_filter.H = np.eye(4, 8, dtype=np.float64)

        # Noise tuning (similar scaling to XCYCSR version)
        self.kalman_filter.R *= 1.0  # measurement noise
        self.kalman_filter.P[4:, 4:] *= 1000.0  # high uncertainty for velocities
        self.kalman_filter.P *= 10.0
        self.kalman_filter.Q[4:, 4:] *= 0.01

        # Initialize state with first observation (direct XYXY)
        self.kalman_filter.x[:4] = initial_bbox.reshape((4, 1))

    def _bbox_to_measurement(self, bbox: np.ndarray) -> np.ndarray:
        """Convert bbox to measurement based on state representation."""
        if self.state_repr == StateRepresentation.XCYCSR:
            return xyxy_to_xcycsr(bbox)
        else:
            return bbox

    def _state_to_bbox(self) -> np.ndarray:
        """Convert current state to xyxy bbox."""
        if self.state_repr == StateRepresentation.XCYCSR:
            return xcycsr_to_xyxy(self.kalman_filter.x[:4].reshape((4,)))
        else:
            return self.kalman_filter.x[:4].reshape((4,))

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
            new_bbox: New observation bounding box [x1, y1, x2, y2].
        """
        if self._frozen_state is None:
            return

        # Restore to frozen state
        self.kalman_filter.set_state(self._frozen_state)

        time_gap = self.time_since_update

        if self.state_repr == StateRepresentation.XCYCSR:
            self._unfreeze_xcycsr(new_bbox, time_gap)
        else:
            self._unfreeze_xyxy(new_bbox, time_gap)

        self._frozen_state = None

    def _unfreeze_xcycsr(self, new_bbox: np.ndarray, time_gap: int) -> None:
        """ORU interpolation for XCYCSR representation."""
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

        for i in range(1, time_gap + 1):
            x = x1 + i * dx
            y = y1 + i * dy
            w = w1 + i * dw
            h = h1 + i * dh

            # Convert back to (x, y, s, r)
            s = w * h
            r = w / h
            virtual_obs = np.array([x, y, s, r]).reshape((4, 1))

            if i < time_gap:
                self.kalman_filter.predict()
            self.kalman_filter.update(virtual_obs)

    def _unfreeze_xyxy(self, new_bbox: np.ndarray, time_gap: int) -> None:
        """ORU interpolation for XYXY representation."""
        last_xyxy = self.last_observation
        new_xyxy = new_bbox

        # Linear interpolation deltas for each coordinate
        delta = (new_xyxy - last_xyxy) / time_gap

        for i in range(1, time_gap + 1):
            virtual_obs = (last_xyxy + i * delta).reshape((4, 1))

            if i < time_gap:
                self.kalman_filter.predict()
            self.kalman_filter.update(virtual_obs)

    def update(self, bbox: np.ndarray | None) -> None:
        """Update tracklet with new observation.

        Handles ORU: if track was lost and now observed again,
        generates virtual trajectory to smooth the transition.

        Args:
            bbox: Bounding box [x1, y1, x2, y2] or None for no observation.
        """
        if bbox is not None:
            # Check if we need to unfreeze (was lost, now observed)
            if not self._observed and self._frozen_state is not None:
                self._unfreeze(bbox)
            else:
                measurement = self._bbox_to_measurement(bbox)
                self.kalman_filter.update(measurement)

            self._observed = True
            self.time_since_update = 0
            self.number_of_successful_consecutive_updates += 1
            self.previous_to_last_observation = self.last_observation
            self.last_observation = bbox
        else:
            # No observation - freeze state if this is first miss
            if self._observed:
                self._freeze()
            self._observed = False
            self.kalman_filter.update(None)

    def predict(self) -> np.ndarray:
        """Predict next bounding box position.

        Returns:
            Predicted bounding box [x1, y1, x2, y2].
        """
        self.kalman_filter.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.number_of_successful_consecutive_updates = 0

        self.time_since_update += 1
        return self._state_to_bbox()

    def is_lost(self) -> bool:
        """Check if tracklet is considered lost."""
        return self.time_since_update > 1

    def get_state_bbox(self) -> np.ndarray:
        """Get current bounding box estimate from Kalman filter.

        Returns:
            Current bounding box estimate [x1, y1, x2, y2].
        """
        return self._state_to_bbox()
