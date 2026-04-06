# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
from numpy.typing import NDArray


class SORTKalmanBoxTracker:
    """
    The `SORTKalmanBoxTracker` class represents the internals of a single
    tracked object (bounding box), with a Kalman filter to predict and update
    its position.

    Attributes:
        tracker_id: Unique identifier for the tracker.
        number_of_successful_updates: Number of times the object has been
            updated successfully.
        time_since_update: Number of frames since the last update.
        state: State vector of the bounding box.
        F: State transition matrix.
        H: Measurement matrix.
        Q: Process noise covariance matrix.
        R: Measurement noise covariance matrix.
        P: Error covariance matrix.
        count_id: Class variable to assign unique IDs to each tracker.

    Args:
        bbox: Initial bounding box in the form [x1, y1, x2, y2].
    """

    count_id: int = 0
    state: NDArray[np.float32]
    F: NDArray[np.float32]
    H: NDArray[np.float32]
    Q: NDArray[np.float32]
    R: NDArray[np.float32]
    P: NDArray[np.float32]

    @classmethod
    def get_next_tracker_id(cls) -> int:
        next_id = cls.count_id
        cls.count_id += 1
        return next_id

    def __init__(
        self,
        bbox: NDArray[np.float64],
        velocity_decay: float = 0.95,
        q_miss_alpha: float = 0.0,
        p_reset_threshold: int = 0,
        oru_threshold: int = 0,
    ) -> None:
        # Initialize with a temporary ID of -1
        # Will be assigned a real ID when the track is considered mature
        self.tracker_id = -1

        # Number of hits indicates how many times the object has been
        # updated successfully
        self.number_of_successful_updates = 1
        # Number of frames since the last update
        self.time_since_update = 0

        # Kalman dynamics hyper-parameters
        # velocity_decay: shrinks velocity components each missed frame to prevent
        # runaway linear extrapolation during occlusions (technique from OC-SORT).
        self.velocity_decay = velocity_decay
        # q_miss_alpha: multiplicative Q-inflation rate for missed frames —
        # widens predicted covariance so the filter trusts fresh measurements
        # more on re-detection. Orthogonal to velocity_decay.
        self.q_miss_alpha = q_miss_alpha
        # p_reset_threshold: if a track was lost for >= this many frames before
        # re-detection, reset P to identity after the update step so stale
        # uncertainty is discarded. 0 disables the reset.
        self.p_reset_threshold = p_reset_threshold
        # oru_threshold: observation-centric re-estimation update — on re-detection
        # after >= this many missed frames, override the Kalman velocity with a
        # virtual trajectory computed from (current - last_observed) / gap.
        # Technique from OC-SORT. 0 disables.
        self.oru_threshold = oru_threshold

        # Store last observed bbox for observation-centric velocity re-estimation
        self._last_observed_bbox: NDArray[np.float32] | None = None

        # For simplicity, we keep a small state vector:
        # (x, y, x2, y2, vx, vy, vx2, vy2).
        # We'll store the bounding box in "self.state"
        self.state = np.zeros((8, 1), dtype=np.float32)

        # Initialize state directly from the first detection
        bbox_float: NDArray[np.float32] = bbox.astype(np.float32)
        self.state[0, 0] = bbox_float[0]
        self.state[1, 0] = bbox_float[1]
        self.state[2, 0] = bbox_float[2]
        self.state[3, 0] = bbox_float[3]
        self._last_observed_bbox = bbox_float[:4].copy()

        # Basic constant velocity model
        self._initialize_kalman_filter()

    def _initialize_kalman_filter(self) -> None:
        """
        Sets up the matrices for the Kalman filter.
        """
        # State transition matrix (F): 8x8
        # We assume a constant velocity model. Positions are incremented by
        # velocity each step.
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix (H): we directly measure x1, y1, x2, y2
        self.H = np.eye(4, 8, dtype=np.float32)  # 4x8

        # Process covariance matrix (Q)
        self.Q = np.eye(8, dtype=np.float32) * 0.01

        # Measurement covariance (R): noise in detection
        self.R = np.eye(4, dtype=np.float32) * 0.1

        # Error covariance matrix (P)
        self.P = np.eye(8, dtype=np.float32)

    def predict(self) -> None:
        """
        Predict the next state of the bounding box (applies the state transition).
        """
        # Velocity decay: shrink velocity components when the track is lost to
        # prevent runaway linear extrapolation during occlusions.
        if self.time_since_update > 0 and self.velocity_decay < 1.0:
            self.state[4:8] = (self.state[4:8] * self.velocity_decay).astype(np.float32)

        # Predict state
        self.state = (self.F @ self.state).astype(np.float32)

        # Q inflation for missed frames: widen uncertainty so the filter gives
        # higher weight to fresh measurements on re-detection.
        if self.time_since_update > 0 and self.q_miss_alpha > 0.0:
            q_scale = 1.0 + self.q_miss_alpha * self.time_since_update
            q_eff = (self.Q * q_scale).astype(np.float32)
        else:
            q_eff = self.Q

        # Predict error covariance
        self.P = (self.F @ self.P @ self.F.T + q_eff).astype(np.float32)

        # Increase time since update
        self.time_since_update += 1

    def update(self, bbox: NDArray[np.float64]) -> None:
        """
        Updates the state with a new detected bounding box.

        Args:
            bbox: Detected bounding box in the form [x1, y1, x2, y2].
        """
        was_lost_for = self.time_since_update
        self.time_since_update = 0
        self.number_of_successful_updates += 1

        # Observation-centric velocity re-estimation (OC-SORT technique):
        # on re-detection after a gap, compute a "virtual trajectory" velocity
        # from (current_bbox - last_observed_bbox) / gap and override the
        # (decayed) Kalman velocity estimate.
        if (
            self.oru_threshold > 0
            and was_lost_for >= self.oru_threshold
            and self._last_observed_bbox is not None
        ):
            bbox_f = bbox.astype(np.float32)
            virtual_vel = (
                (bbox_f[:4] - self._last_observed_bbox) / was_lost_for
            )
            self.state[4, 0] = virtual_vel[0]
            self.state[5, 0] = virtual_vel[1]
            self.state[6, 0] = virtual_vel[2]
            self.state[7, 0] = virtual_vel[3]

        # Kalman Gain
        S: NDArray[np.float32] = (self.H @ self.P @ self.H.T + self.R).astype(
            np.float32
        )
        K: NDArray[np.float32] = (self.P @ self.H.T @ np.linalg.inv(S)).astype(
            np.float32
        )

        # Residual
        measurement: NDArray[np.float32] = bbox.reshape((4, 1)).astype(np.float32)
        y: NDArray[np.float32] = (
            measurement - self.H @ self.state
        )  # y should be float32 (4,1)

        # Update state
        self.state = (self.state + K @ y).astype(np.float32)

        # Update covariance
        identity_matrix: NDArray[np.float32] = np.eye(8, dtype=np.float32)
        self.P = ((identity_matrix - K @ self.H) @ self.P).astype(np.float32)

        # Store this observation for future ORU velocity re-estimation
        self._last_observed_bbox = bbox.astype(np.float32)[:4].copy()

        # P reset: after a long gap, discard stale accumulated uncertainty so
        # velocity estimation starts fresh from the re-detection.
        if self.p_reset_threshold > 0 and was_lost_for >= self.p_reset_threshold:
            self.P = np.eye(8, dtype=np.float32)

    def get_state_bbox(self) -> NDArray[np.float32]:
        """
        Returns the current bounding box estimate from the state vector.

        Returns:
            The bounding box [x1, y1, x2, y2].
        """
        return self.state[:4, 0].flatten().astype(np.float32)
