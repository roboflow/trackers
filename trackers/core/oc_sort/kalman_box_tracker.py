from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from trackers.core.sort.kalman_box_tracker import SORTKalmanBoxTracker


class OCSORTKalmanBoxTracker(SORTKalmanBoxTracker):
    """
    The `OCSORTKalmanBoxTracker` class extends `SORTKalmanBoxTracker` with
    Observation-Centric Re-Update (ORU) capabilities.
    """

    def __init__(
        self, bbox: NDArray[np.float64], ocm_observation_history_size: int
    ) -> None:
        super().__init__(bbox)
        self.last_observation: NDArray[np.float32] = (
            self.state[:4, 0].flatten().astype(np.float32)
        )
        self.history_observations: list[NDArray[np.float32]] = [self.last_observation]
        self.kf_state_at_last_observation: NDArray[np.float32] = self.state.copy()
        self.P_at_last_observation: NDArray[np.float32] = self.P.copy()
        self.ocm_observation_history_size = ocm_observation_history_size

    def update(self, bbox: NDArray[np.float64]) -> None:
        """
        Updates the state with a new detected bounding box and
        saves the observation and KF state.

        Args:
            bbox (np.ndarray): Detected bounding box in [x1, y1, x2, y2].
        """
        self.kf_state_at_last_observation = self.state.copy()
        self.P_at_last_observation = self.P.copy()
        super().update(bbox)
        self.last_observation = bbox.copy().astype(np.float32)

        # Keep history of observations to a certain length for OCM
        if len(self.history_observations) >= self.ocm_observation_history_size:
            self.history_observations.pop(0)
        self.history_observations.append(self.last_observation)

    def _observation_centric_re_update(self, bbox: NDArray[np.float32]) -> None:
        """
        Perform observation-centric re-update.
        """
        # Kalman Gain
        S: NDArray[np.float32] = self.H @ self.P @ self.H.T + self.R
        K: NDArray[np.float32] = (self.P @ self.H.T @ np.linalg.inv(S)).astype(
            np.float32
        )

        # Residual
        measurement: NDArray[np.float32] = bbox.reshape((4, 1)).astype(np.float32)
        y: NDArray[np.float32] = measurement - self.H @ self.state

        # Update state
        self.state = (self.state + K @ y).astype(np.float32)

        # Update covariance
        identity_matrix: NDArray[np.float32] = np.eye(8, dtype=np.float32)
        self.P = ((identity_matrix - K @ self.H) @ self.P).astype(np.float32)

    def apply_oru(self, observation: NDArray[np.float32]) -> None:
        """
        Applies Observation-centric Re-Update (ORU).

        Args:
            observation (NDArray[np.float32]): The new observation that re-activated
                the track.
        """
        if self.time_since_update <= 1:
            # No need for ORU if the track was not untracked for more than 1 frame
            return

        # Restore the KF state to when the track was last seen
        self.state = self.kf_state_at_last_observation.copy()
        self.P = self.P_at_last_observation.copy()

        untracked_period = self.time_since_update

        # Generate virtual trajectory and replay the predict-update cycle
        for t in range(1, untracked_period + 1):
            # Predict step
            self.state = (self.F @ self.state).astype(np.float32)
            self.P = (self.F @ self.P @ self.F.T + self.Q).astype(np.float32)

            # Generate virtual observation using linear interpolation
            # as per the adopted version of eqn 6 from section 5.1 in the paper
            interpolation_factor = t / untracked_period
            virtual_observation = self.last_observation + interpolation_factor * (
                observation - self.last_observation
            )

            # Re-update step with virtual observation
            self._observation_centric_re_update(virtual_observation)
