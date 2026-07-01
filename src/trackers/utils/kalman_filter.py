# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class KalmanFilter:
    """Generic Kalman filter for state estimation.
    A standard linear Kalman filter for state estimation. This is a clean,
    general-purpose implementation that can be used by any tracker.
    Performs predict/update algebra only. Process models (transition_mtx,
    process_noise) are supplied externally — see ``trackers.utils.motion_models``
    and ``BaseStateEstimator`` in ``state_representations.py``.

    Attributes:
        dim_x: Dimension of state vector.
        dim_z: Dimension of measurement vector.
        state: State vector (dim_x, 1).
        state_covariance: State covariance matrix (dim_x, dim_x).
        transition_mtx: State transition matrix (dim_x, dim_x).
        observation_mtx: Measurement function matrix (dim_z, dim_x).
        process_noise: Process noise covariance (dim_x, dim_x).
        measurement_noise: Measurement noise covariance (dim_z, dim_z).
        state_prior: Prior state estimate (after predict, before update).
        state_covariance_prior: Prior covariance (after predict, before update).
        state_post: Posterior state estimate (after update).
        state_covariance_post: Posterior covariance (after update).
        kalman_gain: Kalman gain matrix (dim_x, dim_z).
        innovation: Measurement residual (dim_z, 1).
        innovation_cov: Innovation covariance (dim_z, dim_z).
    """

    def __init__(self, dim_x: int, dim_z: int) -> None:
        """Initialize Kalman filter.

        Args:
            dim_x: Dimension of state vector.
            dim_z: Dimension of measurement vector.
        """
        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")

        self.dim_x = dim_x
        self.dim_z = dim_z

        # State and covariance
        self.state: NDArray[np.float64] = np.zeros((dim_x, 1), dtype=np.float64)
        self.state_covariance: NDArray[np.float64] = np.eye(dim_x, dtype=np.float64)

        # Process model
        self.transition_mtx: NDArray[np.float64] = np.eye(dim_x, dtype=np.float64)
        self.process_noise: NDArray[np.float64] = np.eye(dim_x, dtype=np.float64)

        # Measurement model
        self.observation_mtx: NDArray[np.float64] = np.zeros((dim_z, dim_x), dtype=np.float64)
        self.measurement_noise: NDArray[np.float64] = np.eye(dim_z, dtype=np.float64)

        # Prior and posterior (for inspection/debugging)
        self.state_prior: NDArray[np.float64] = self.state.copy()
        self.state_covariance_prior: NDArray[np.float64] = self.state_covariance.copy()
        self.state_post: NDArray[np.float64] = self.state.copy()
        self.state_covariance_post: NDArray[np.float64] = self.state_covariance.copy()

        # Kalman gain, innovation, innovation covariance (computed during update)
        self.kalman_gain: NDArray[np.float64] = np.zeros((dim_x, dim_z), dtype=np.float64)
        self.innovation: NDArray[np.float64] = np.zeros((dim_z, 1), dtype=np.float64)
        self.innovation_cov: NDArray[np.float64] = np.zeros((dim_z, dim_z), dtype=np.float64)

        self._identity: NDArray[np.float64] = np.eye(dim_x, dtype=np.float64)

    def predict(self) -> None:
        """Predict next state (prior) using ``transition_mtx`` and ``process_noise``.

        Computes:
            state = transition_mtx @ state
            state_covariance = transition_mtx @ state_covariance @ transition_mtx.T + process_noise
        """
        self.state = self.transition_mtx @ self.state
        self.state_covariance = self.transition_mtx @ self.state_covariance @ self.transition_mtx.T + self.process_noise

        # Save prior
        self.state_prior = self.state.copy()
        self.state_covariance_prior = self.state_covariance.copy()

    def update(self, z: NDArray[np.float64] | None) -> None:
        """Update state estimate with measurement.

        If z is None, the state is not updated (prediction only).

        Args:
            z: Measurement vector (dim_z, 1) or None for no observation.
        """
        if z is None:
            # No observation - posterior equals prior
            self.state_post = self.state.copy()
            self.state_covariance_post = self.state_covariance.copy()
            self.innovation = np.zeros((self.dim_z, 1), dtype=np.float64)
            return

        # Ensure z is column vector
        z = np.asarray(z, dtype=np.float64).reshape((self.dim_z, 1))

        # Residual: innovation = z - observation_mtx @ state
        self.innovation = z - self.observation_mtx @ self.state

        # System uncertainty: innovation_cov = H @ P @ H.T + measurement_noise
        PHT = self.state_covariance @ self.observation_mtx.T
        self.innovation_cov = self.observation_mtx @ PHT + self.measurement_noise

        # Kalman gain: kalman_gain = state_covariance @ observation_mtx.T @ innovation_cov^-1
        self.kalman_gain = PHT @ np.linalg.inv(self.innovation_cov)

        # State update: state = state + kalman_gain @ innovation
        self.state = self.state + self.kalman_gain @ self.innovation

        # Covariance update (Joseph form for numerical stability):
        # P = (I - K @ H) @ P @ (I - K @ H).T + K @ measurement_noise @ K.T
        I_KH = self._identity - self.kalman_gain @ self.observation_mtx
        self.state_covariance = (
            I_KH @ self.state_covariance @ I_KH.T + self.kalman_gain @ self.measurement_noise @ self.kalman_gain.T
        )

        # Save posterior
        self.state_post = self.state.copy()
        self.state_covariance_post = self.state_covariance.copy()

    def get_state(self) -> dict:
        """Get current filter state for saving.

        Returns:
            Dictionary with state vector and all matrices.
        """
        return {
            "state": self.state.copy(),
            "state_covariance": self.state_covariance.copy(),
            "transition_mtx": self.transition_mtx.copy(),
            "observation_mtx": self.observation_mtx.copy(),
            "process_noise": self.process_noise.copy(),
            "measurement_noise": self.measurement_noise.copy(),
        }

    def set_state(self, state: dict) -> None:
        """Restore filter state from saved dictionary.

        Args:
            state: Dictionary from get_state().
        """
        self.state = state["state"].copy()
        self.state_covariance = state["state_covariance"].copy()
        self.transition_mtx = state["transition_mtx"].copy()
        self.observation_mtx = state["observation_mtx"].copy()
        self.process_noise = state["process_noise"].copy()
        self.measurement_noise = state["measurement_noise"].copy()
