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


class KalmanFilter:
    """Generic Kalman filter implementation.

    A standard linear Kalman filter for state estimation. This is a clean,
    general-purpose implementation that can be used by any tracker.

    Variable time-step support (opt-in):
        Call `set_motion_model_builders` to register `F_builder` and
        `Q_builder` callables. That flips on time-aware prediction: `predict(dt)`
        rebuilds `self.F` and `self.Q` whenever `dt` differs from the last
        value used. When builders are not registered, `predict(dt)` ignores
        `dt` and uses the stored `F`/`Q` as-is — the backward-compatible path.

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

        # Time-parameterized motion model (opt-in via set_motion_model_builders).
        self._motion_model_builders_enabled: bool = False
        self._F_builder: FBuilder | None = None
        self._Q_builder: QBuilder | None = None
        # `_cached_dt is None` means no time-aware predict has run yet, so
        # the stored F/Q are still the caller-supplied "reference" matrices.
        self._cached_dt: float | None = None

    def set_motion_model_builders(
        self,
        F_builder: FBuilder,
        Q_builder: QBuilder,
    ) -> None:
        """Register time-parameterized F and Q builders for variable-dt predict.

        After registration, `predict(dt)` will rebuild `self.F` and `self.Q`
        from the builders whenever `dt` changes. Until the first call with a
        non-default `dt`, the stored F/Q are preserved unchanged — meaning
        callers that never opt into variable-dt predict get byte-for-byte
        backward compatible behaviour.

        Args:
            F_builder: Callable mapping `dt -> F(dt)` (dim_x, dim_x).
            Q_builder: Callable mapping `dt -> Q(dt)` (dim_x, dim_x).
        """
        self._F_builder = F_builder
        self._Q_builder = Q_builder
        self._motion_model_builders_enabled = True

    def predict(self, dt: float = 1.0) -> None:
        """Predict next state (prior) using the state transition model.

        Computes:
            x = F @ x
            P = F @ P @ F.T + Q

        If time-parameterized builders are registered (see
        `set_motion_model_builders`), `F` and `Q` are rebuilt from the
        builders when `dt` differs from the last `dt` used. The very first
        call is treated specially: if `dt == 1.0` and no prior time-aware
        call has happened, the stored `F`/`Q` are kept (preserving the
        caller's reference calibration); any other `dt` triggers a rebuild
        and from that point on every `dt` change rebuilds F and Q.

        Args:
            dt: Time elapsed since the last predict, in seconds. Default
                `1.0` corresponds to the implicit "one frame per call"
                semantics used everywhere before this change.
        """
        if self._motion_model_builders_enabled:
            assert self._F_builder is not None and self._Q_builder is not None
            if self._cached_dt is None:
                # First time-aware predict. Preserve stored F/Q only if dt is
                # the default; otherwise honour the builders from step one.
                if dt != 1.0:
                    self.F = self._F_builder(dt)
                    self.Q = self._Q_builder(dt)
                self._cached_dt = dt
            elif dt != self._cached_dt:
                self.F = self._F_builder(dt)
                self.Q = self._Q_builder(dt)
                self._cached_dt = dt

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
