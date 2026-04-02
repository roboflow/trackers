# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
from numpy.typing import NDArray


class ByteTrackKalmanBoxTracker:
    """
    The `ByteTrackKalmanBoxTracker` class represents the internals of a single
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

    count_id = 0
    # Velocity decay factor applied per frame when the track is unmatched.
    # Attenuates velocity to prevent unbounded linear drift during occlusion,
    # keeping the predicted box near the last observed position.  Standard
    # technique from OC-SORT / BoT-SORT that directly improves association
    # accuracy (AssA) for re-identification after occlusion gaps.
    velocity_decay: float = 0.95
    # Process noise inflation rate for lost tracks.  During occlusion the true
    # object motion is unknown, so uncertainty should grow faster than in the
    # steady-state.  Each missed frame multiplies the process noise covariance
    # by (1 + q_miss_alpha * time_since_update), widening the predicted
    # covariance and increasing Kalman gain on re-appearance — the filter
    # trusts the fresh measurement more.  Orthogonal to velocity_decay which
    # acts on the state mean, not the covariance.
    q_miss_alpha: float = 0.1
    # Minimum number of missed frames before resetting P on re-detection.
    # When a lost track is re-matched after a long gap, the accumulated
    # error covariance P carries stale position-velocity cross-correlations
    # from the Q-inflated lost period.  Resetting P to a fresh initial state
    # after the Kalman update gives the filter a clean slate for velocity
    # estimation while keeping the measurement-corrected position from the
    # update step.  Set to 0 to disable the reset.
    p_reset_threshold: int = 5
    # Minimum gap (in missed frames) before applying Observation-Centric
    # Re-Update (ORU) on re-detection.  When a lost track is re-matched
    # after >= oru_threshold frames, the filter state is rolled back to the
    # last observed state and virtual observations are replayed along the
    # linearly interpolated trajectory from last_observation to the new
    # detection.  This re-estimates velocity correctly instead of relying on
    # stale/decayed velocity from the lost period.  Set to 0 to disable ORU.
    # Technique from OC-SORT (Cao et al., 2023).
    oru_threshold: int = 2
    state: NDArray[np.float32]
    F: NDArray[np.float32]
    H: NDArray[np.float32]
    Q: NDArray[np.float32]
    R: NDArray[np.float32]
    P: NDArray[np.float32]

    @classmethod
    def get_next_tracker_id(cls) -> int:
        """
        Class method that returns the next available tracker ID.

        Returns:
            The next available tracker ID.
        """
        next_id = cls.count_id
        cls.count_id += 1
        return next_id

    def __init__(self, bbox: np.ndarray):
        # Initialize with a temporary ID of -1
        # Will be assigned a real ID when the track is considered mature
        self.tracker_id = -1

        # Number of hits indicates how many times the object has been
        # updated successfully
        self.number_of_successful_updates = 1
        # Number of frames since the last update
        self.time_since_update = 0

        # For simplicity, we keep a small state vector:
        # (x, y, x2, y2, vx, vy, vx2, vy2).
        # We'll store the bounding box in "self.state"
        self.state = np.zeros((8, 1), dtype=np.float32)

        # Initialize state directly from the first detection
        self.state[0] = bbox[0]
        self.state[1] = bbox[1]
        self.state[2] = bbox[2]
        self.state[3] = bbox[3]

        # ORU: last observed bbox and frozen Kalman state for re-update
        self.last_observation: np.ndarray = bbox.copy()
        self._frozen_state: dict[str, NDArray[np.float32]] | None = None
        self._was_observed: bool = True

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

    def _freeze(self) -> None:
        """Save Kalman filter state at the moment of last observation.

        Called when a track transitions from observed to lost. The frozen
        state is used by ORU to roll back and replay virtual observations
        on re-detection.
        """
        self._frozen_state = {
            "state": self.state.copy(),
            "P": self.P.copy(),
        }

    def _apply_oru(self, new_bbox: np.ndarray) -> None:
        """Observation-Centric Re-Update: restore frozen state and replay.

        Rolls back to the frozen Kalman state, then replays
        ``time_since_update`` predict+update cycles with linearly
        interpolated virtual observations from ``last_observation`` to
        ``new_bbox``.  The final real update is done by the caller.

        Args:
            new_bbox: The new detection in ``[x1, y1, x2, y2]`` format.
        """
        if self._frozen_state is None:
            return

        # Restore frozen state
        self.state = self._frozen_state["state"]
        self.P = self._frozen_state["P"]
        self._frozen_state = None

        time_gap = self.time_since_update
        last = self.last_observation
        delta = (new_bbox - last) / time_gap

        # Replay virtual predict+update cycles (all but the last step;
        # the final real update is performed by the normal update() call).
        for i in range(1, time_gap):
            # Predict one step
            self.state = (self.F @ self.state).astype(np.float32)
            self.P = (self.F @ self.P @ self.F.T + self.Q).astype(np.float32)

            # Virtual observation at interpolated position
            virtual_obs = (last + i * delta).reshape((4, 1)).astype(np.float32)
            s_mat = self.H @ self.P @ self.H.T + self.R
            k_mat = (self.P @ self.H.T @ np.linalg.inv(s_mat)).astype(np.float32)
            y_res = virtual_obs - self.H @ self.state
            self.state = (self.state + k_mat @ y_res).astype(np.float32)
            ident = np.eye(8, dtype=np.float32)
            self.P = ((ident - k_mat @ self.H) @ self.P).astype(np.float32)

        # After ORU, do one final predict so the caller's update() call
        # sees a properly predicted state at the current time step.
        self.state = (self.F @ self.state).astype(np.float32)
        self.P = (self.F @ self.P @ self.F.T + self.Q).astype(np.float32)

    def predict(self) -> None:
        """
        Predict the next state of the bounding box (applies the state transition).
        """
        # Predict state
        self.state = (self.F @ self.state).astype(np.float32)

        # When the track is lost, inflate process noise to reflect growing
        # uncertainty about the object's true position.  The base Q is NOT
        # mutated — a scaled copy is used for this prediction step only.
        if self.time_since_update > 0:
            q_scale = 1.0 + self.q_miss_alpha * self.time_since_update
            q_eff = self.Q * q_scale
        else:
            q_eff = self.Q

        # Predict error covariance
        self.P = (self.F @ self.P @ self.F.T + q_eff).astype(np.float32)

        # Attenuate velocity components when track is lost (unmatched).
        # Indices 4-7 are (vx1, vy1, vx2, vy2).  Applied *after* the
        # transition so the current frame's prediction still uses full
        # velocity; only subsequent predictions on still-lost tracks see
        # progressively slower motion.
        if self.time_since_update > 0:
            self.state[4:8] *= self.velocity_decay

        # Freeze state on transition from observed → lost (first miss)
        if self.time_since_update == 0 and self._was_observed:
            self._freeze()
            self._was_observed = False

        # Increase time since update
        self.time_since_update += 1

    def update(self, bbox: np.ndarray) -> None:
        """
        Updates the state with a new detected bounding box.

        Args:
            bbox: Detected bounding box in the form [x1, y1, x2, y2].
        """
        was_lost_for = self.time_since_update

        # ORU: if track was lost long enough, roll back and replay virtual
        # observations so the Kalman filter re-learns velocity from the
        # interpolated trajectory rather than using the stale/decayed one.
        if (
            self.oru_threshold > 0
            and was_lost_for >= self.oru_threshold
            and self._frozen_state is not None
        ):
            self._apply_oru(bbox)

        self.time_since_update = 0
        self.number_of_successful_updates += 1
        self._was_observed = True

        # Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = (self.P @ self.H.T @ np.linalg.inv(S)).astype(np.float32)

        # Residual
        measurement = bbox.reshape((4, 1)).astype(np.float32)
        y = measurement - self.H @ self.state

        # Update state
        self.state = (self.state + K @ y).astype(np.float32)

        # Update covariance
        identity_matrix = np.eye(8, dtype=np.float32)
        self.P = ((identity_matrix - K @ self.H) @ self.P).astype(np.float32)

        # Reset P after long occlusion to clear stale cross-covariances.
        # The Kalman update above already used the inflated P (high gain,
        # trusting the fresh measurement).  Resetting afterwards gives the
        # filter a clean velocity-estimation slate for subsequent frames.
        if self.p_reset_threshold > 0 and was_lost_for >= self.p_reset_threshold:
            self.P = np.eye(8, dtype=np.float32)

        # Update last observation for future ORU interpolation
        self.last_observation = bbox.copy()

    def get_state_bbox(self) -> np.ndarray:
        """
        Returns the current bounding box estimate from the state vector.

        Returns:
            The bounding box [x1, y1, x2, y2].
        """
        return np.array(
            [
                self.state[0],  # x1
                self.state[1],  # y1
                self.state[2],  # x2
                self.state[3],  # y2
            ],
            dtype=float,
        ).reshape(-1)
