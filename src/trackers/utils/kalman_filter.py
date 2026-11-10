# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class KalmanFilter:
    """Generic Kalman filter for state estimation.

    A standard linear Kalman filter for state estimation.
    This is a clean, general-purpose implementation that can be used by any
    tracker. Performs predict/update algebra only. Process models
    (transition_mtx, process_noise) are supplied externally — see
    ``trackers.utils.motion_models`` and ``BaseStateEstimator`` in
    ``state_representations.py``.

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

        # Reference the prior (debug/inspection only). predict/update always
        # reassign self.state[_covariance] to freshly allocated arrays and never
        # mutate them in place on the hot path, so a plain reference is a faithful
        # snapshot without paying a per-step copy.
        self.state_prior = self.state
        self.state_covariance_prior = self.state_covariance

    def update(self, z: NDArray[np.float64] | None) -> None:
        """Update state estimate with measurement.

        If z is None, the state is not updated (prediction only).

        Args:
            z: Measurement vector (dim_z, 1) or None for no observation.
        """
        if z is None:
            # No observation - posterior equals prior (reference; see predict()).
            self.state_post = self.state
            self.state_covariance_post = self.state_covariance
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

        # Reference the posterior (debug/inspection only; see predict()).
        self.state_post = self.state
        self.state_covariance_post = self.state_covariance

    @staticmethod
    def update_batch(
        filters: Sequence[KalmanFilter],
        measurements: Sequence[NDArray[np.float64]],
        *,
        check_models: bool = True,
    ) -> None:
        """Update several filters with one observation each, using batched linear algebra.

        The filters must describe the same model (equal ``dim_x``, ``dim_z``, ``observation_mtx``
        and ``measurement_noise``) — as filters owned by tracklets of one tracker do. Only the
        state and covariance differ, so the algebra is stacked into `(N, dim_x, 1)` /
        `(N, dim_x, dim_x)` arrays and evaluated once. Each filter ends up in the same state as
        calling `update` on it individually.

        Args:
            filters: Filters to update; at least one.
            measurements: One measurement per filter, each shaped `(dim_z, 1)`.
            check_models: Verify that every filter shares the reference model. Callers that
                already know it — the tracklets of one tracker are constructed from one estimator
                class — can turn this off to skip the per-filter matrix comparison.

        Raises:
            ValueError: If the two sequences differ in length, or the filters do not share one model.
        """
        if len(filters) != len(measurements):
            raise ValueError("filters and measurements must have the same length")
        if not filters:
            return

        first = filters[0]
        dim_x, dim_z = first.dim_x, first.dim_z
        for kf in filters:
            if kf.dim_x != dim_x or kf.dim_z != dim_z:
                raise ValueError("update_batch requires filters with equal dimensions")
            if check_models and (
                not np.array_equal(kf.observation_mtx, first.observation_mtx)
                or not np.array_equal(kf.measurement_noise, first.measurement_noise)
            ):
                raise ValueError("update_batch requires filters with the same model")

        states = np.stack([kf.state for kf in filters])
        covariances = np.stack([kf.state_covariance for kf in filters])
        observations = np.stack([np.asarray(z, dtype=np.float64).reshape((dim_z, 1)) for z in measurements])

        observation_mtx = first.observation_mtx
        measurement_noise = first.measurement_noise

        innovations = observations - observation_mtx @ states
        cov_times_obs_t = covariances @ observation_mtx.T
        innovations_cov = observation_mtx @ cov_times_obs_t + measurement_noise
        kalman_gains = cov_times_obs_t @ np.linalg.inv(innovations_cov)

        states_post = states + kalman_gains @ innovations
        i_kh = first._identity - kalman_gains @ observation_mtx
        i_kh_t = i_kh.transpose(0, 2, 1)
        gain_t = kalman_gains.transpose(0, 2, 1)
        covariances_post = i_kh @ covariances @ i_kh_t + kalman_gains @ measurement_noise @ gain_t

        for i, kf in enumerate(filters):
            kf.state = states_post[i]
            kf.state_covariance = covariances_post[i]
            kf.kalman_gain = kalman_gains[i]
            kf.innovation = innovations[i]
            kf.innovation_cov = innovations_cov[i]
            kf.state_post = kf.state
            kf.state_covariance_post = kf.state_covariance

    def get_state(self) -> dict:
        """Get current filter state for saving.

        ``process_noise`` reflects the last predict step's Q, which may be DWNA gap-scaled rather than one-frame noise.

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
