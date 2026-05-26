# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FBuilder = Callable[[float], NDArray[np.float64]]
QBuilder = Callable[[float], NDArray[np.float64]]
MotionModelSync = Callable[[float], None]


class KalmanFilter:
    """Generic Kalman filter implementation.

    A standard linear Kalman filter for state estimation. This is a clean,
    general-purpose implementation that can be used by any tracker.

    Variable time-step support (opt-in):
        Call `set_motion_model_builders` to install dt-aware F/Q rebuilding.
        Until then, `predict(dt)` uses the stored `F`/`Q` matrices regardless
        of `dt` — the backward-compatible path for callers that never register
        builders.

    Attributes:
        dim_x: Dimension of state vector.
        dim_z: Dimension of measurement vector.
        x: State vector (dim_x, 1).
        P: State covariance matrix (dim_x, dim_x).
        F: State transition matrix (dim_x, dim_x).
        H: Measurement function matrix (dim_z, dim_x).
        Q: Process noise covariance (dim_x, dim_x).
        R: Measurement noise covariance (dim_z, dim_z).
        x_prior: Prior state estimate (after predict, before update).
        P_prior: Prior covariance (after predict, before update).
        x_post: Posterior state estimate (after update).
        P_post: Posterior covariance (after update).
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
        self.x: NDArray[np.float64] = np.zeros((dim_x, 1), dtype=np.float64)
        self.P: NDArray[np.float64] = np.eye(dim_x, dtype=np.float64)

        # Process model
        self.F: NDArray[np.float64] = np.eye(dim_x, dtype=np.float64)
        self.Q: NDArray[np.float64] = np.eye(dim_x, dtype=np.float64)

        # Measurement model
        self.H: NDArray[np.float64] = np.zeros((dim_z, dim_x), dtype=np.float64)
        self.R: NDArray[np.float64] = np.eye(dim_z, dtype=np.float64)

        # Prior and posterior (for inspection/debugging)
        self.x_prior: NDArray[np.float64] = self.x.copy()
        self.P_prior: NDArray[np.float64] = self.P.copy()
        self.x_post: NDArray[np.float64] = self.x.copy()
        self.P_post: NDArray[np.float64] = self.P.copy()

        # Kalman gain, residual, system uncertainty (computed during update)
        self.K: NDArray[np.float64] = np.zeros((dim_x, dim_z), dtype=np.float64)
        self.y: NDArray[np.float64] = np.zeros((dim_z, 1), dtype=np.float64)
        self.S: NDArray[np.float64] = np.zeros((dim_z, dim_z), dtype=np.float64)

        self._I: NDArray[np.float64] = np.eye(dim_x, dtype=np.float64)

        # No-op until set_motion_model_builders installs dt-aware syncing.
        self._sync_motion_model: MotionModelSync = lambda _dt: None

    def set_motion_model_builders(
        self,
        F_builder: FBuilder,
        Q_builder: QBuilder,
    ) -> None:
        """Install dt-aware F/Q rebuilding for subsequent `predict(dt)` calls.

        The first `predict(1.0)` preserves caller-supplied reference `F`/`Q`.
        Any other `dt`, or a later change in `dt`, rebuilds from the builders.

        Args:
            F_builder: Callable mapping `dt -> F(dt)` (dim_x, dim_x).
            Q_builder: Callable mapping `dt -> Q(dt)` (dim_x, dim_x).
        """
        cached_dt: float | None = None

        def sync(dt: float) -> None:
            nonlocal cached_dt
            if cached_dt is None:
                if dt != 1.0:
                    self.F = F_builder(dt)
                    self.Q = Q_builder(dt)
                cached_dt = dt
            elif dt != cached_dt:
                self.F = F_builder(dt)
                self.Q = Q_builder(dt)
                cached_dt = dt

        self._sync_motion_model = sync

    def predict(self, dt: float = 1.0) -> None:
        """Predict next state (prior) using the state transition model.

        Computes:
            x = F @ x
            P = F @ P @ F.T + Q

        Args:
            dt: Time elapsed since the last predict, in seconds. Default
                `1.0` corresponds to the implicit "one frame per call"
                semantics used everywhere before this change.
        """
        self._sync_motion_model(dt)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z: NDArray[np.float64] | None) -> None:
        """Update state estimate with measurement.

        If z is None, the state is not updated (prediction only).

        Args:
            z: Measurement vector (dim_z, 1) or None for no observation.
        """
        if z is None:
            # No observation - posterior equals prior
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1), dtype=np.float64)
            return

        # Ensure z is column vector
        z = np.asarray(z, dtype=np.float64).reshape((self.dim_z, 1))

        # Residual: y = z - H @ x
        self.y = z - self.H @ self.x

        # System uncertainty: S = H @ P @ H.T + R
        PHT = self.P @ self.H.T
        self.S = self.H @ PHT + self.R

        # Kalman gain: K = P @ H.T @ S^-1
        self.K = PHT @ np.linalg.inv(self.S)

        # State update: x = x + K @ y
        self.x = self.x + self.K @ self.y

        # Covariance update (Joseph form for numerical stability):
        # P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
        I_KH = self._I - self.K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T

        # Save posterior
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def get_state(self) -> dict:
        """Get current filter state for saving.

        Returns:
            Dictionary with x, P, and other matrices.
        """
        return {
            "x": self.x.copy(),
            "P": self.P.copy(),
            "F": self.F.copy(),
            "H": self.H.copy(),
            "Q": self.Q.copy(),
            "R": self.R.copy(),
        }

    def set_state(self, state: dict) -> None:
        """Restore filter state from saved dictionary.

        Args:
            state: Dictionary from get_state().
        """
        self.x = state["x"].copy()
        self.P = state["P"].copy()
        self.F = state["F"].copy()
        self.H = state["H"].copy()
        self.Q = state["Q"].copy()
        self.R = state["R"].copy()
