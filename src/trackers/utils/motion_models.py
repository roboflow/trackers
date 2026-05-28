# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Process models used to populate Kalman filter ``F`` and ``Q`` matrices.

These are motion-model builders and sync helpers, not part of the Kalman filter
algebra itself. ``BaseStateEstimator`` wires a model to a ``KalmanFilter`` for
each bounding-box state layout. See ``docs/learn/track.md`` (Variable frame rate).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from trackers.utils.kalman_filter import KalmanFilter


def build_constant_velocity_F(
    dim_x: int,
    pos_idx: NDArray[np.int64],
    vel_idx: NDArray[np.int64],
    dt: float,
) -> NDArray[np.float64]:
    """Build a constant-velocity state-transition matrix ``F(dt)``.

    For each kinematic pair ``(pos_idx[k], vel_idx[k])`` sets
    ``F[pos, vel] = dt``; velocity rows stay identity.

    Args:
        dim_x: Full state dimension.
        pos_idx: Indices of position coordinates.
        vel_idx: Indices of matching velocity coordinates.
        dt: Step size (frame units or seconds, depending on caller).
    """
    F = np.eye(dim_x, dtype=np.float64)
    for p, v in zip(pos_idx, vel_idx, strict=True):
        F[int(p), int(v)] = dt
    return F


def build_dwna_Q(
    dim_x: int,
    pos_idx: NDArray[np.int64],
    vel_idx: NDArray[np.int64],
    sigma_a2: NDArray[np.float64],
    extra_q_diagonal: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """Build a Discrete White Noise Acceleration process-noise matrix ``Q(dt)``.

    Each kinematic pair receives the standard 2×2 DWNA block:

        ⎡  σ_a² · dt⁴/4    σ_a² · dt³/2 ⎤
        ⎣  σ_a² · dt³/2    σ_a² · dt²   ⎦

    Diagonal entries for state indices outside those pairs are copied from
    ``extra_q_diagonal`` (e.g. the aspect-ratio random walk in XCYCSR).

    Args:
        dim_x: Full state dimension.
        pos_idx: Indices of position coordinates.
        vel_idx: Indices of matching velocity coordinates.
        sigma_a2: Per-coordinate acceleration variance (back-calibrated from Q).
        extra_q_diagonal: Non-kinematic diagonal entries of the reference Q.
        dt: Step size (frame units or seconds, depending on caller).
    """
    Q = np.zeros((dim_x, dim_x), dtype=np.float64)
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    touched = set(int(i) for i in pos_idx) | set(int(i) for i in vel_idx)
    for k, (p, v) in enumerate(zip(pos_idx, vel_idx, strict=True)):
        p_i = int(p)
        v_i = int(v)
        sa2 = float(sigma_a2[k])
        Q[p_i, p_i] = sa2 * dt4 / 4.0
        Q[p_i, v_i] = sa2 * dt3 / 2.0
        Q[v_i, p_i] = sa2 * dt3 / 2.0
        Q[v_i, v_i] = sa2 * dt2
    for i in range(dim_x):
        if i not in touched:
            Q[i, i] = float(extra_q_diagonal[i])
    return Q


@dataclass
class ConstantVelocityDWNA:
    """Constant-velocity motion model with DWNA process noise.

    Holds per-coordinate calibration derived from a reference ``Q`` matrix and
    writes ``F(dt)`` / ``Q(dt)`` onto a ``KalmanFilter`` when ``dt`` changes.

    ``calibration_dt`` is the frame step at which the caller's tuned ``Q`` was
    defined (``1.0`` frame unit for SORT / ByteTrack). ``sync`` re-calibrates
    when the active nominal step changes so ``build_Q(nominal_frame_step)``
    reproduces that reference matrix.
    """

    dim_x: int
    pos_idx: NDArray[np.int64]
    vel_idx: NDArray[np.int64]
    sigma_a2: NDArray[np.float64]
    extra_q_diagonal: NDArray[np.float64]
    calibration_dt: float = 1.0
    reference_Q: NDArray[np.float64] | None = None
    cached_dt: float | None = field(default=None, init=False)

    @classmethod
    def from_filter(
        cls,
        kf: KalmanFilter,
        pos_idx: NDArray[np.int64],
        vel_idx: NDArray[np.int64],
    ) -> ConstantVelocityDWNA:
        """Create a model sized for *kf* with default σ_a² until ``calibrate_from_Q``."""
        return cls(
            dim_x=kf.dim_x,
            pos_idx=pos_idx,
            vel_idx=vel_idx,
            sigma_a2=np.ones(len(pos_idx), dtype=np.float64),
            extra_q_diagonal=np.diag(kf.Q).astype(np.float64).copy(),
            reference_Q=kf.Q.copy(),
        )

    def calibrate_from_Q(self, Q: np.ndarray, calibration_dt: float = 1.0) -> None:
        """Back-calibrate σ_a² from velocity diagonals of a reference ``Q`` matrix.

        Args:
            Q: Reference process-noise matrix valid at ``calibration_dt``.
            calibration_dt: Step size (frame units or seconds) at which ``Q``
                was tuned. Kinematic blocks scale as ``(dt / calibration_dt)²``.
        """
        if calibration_dt <= 0:
            raise ValueError("calibration_dt must be positive.")
        dt2 = calibration_dt * calibration_dt
        self.calibration_dt = calibration_dt
        self.sigma_a2 = np.asarray([float(Q[v, v]) / dt2 for v in self.vel_idx], dtype=np.float64)
        self.extra_q_diagonal = np.diag(Q).astype(np.float64).copy()
        self.reference_Q = np.asarray(Q, dtype=np.float64).copy()
        self.cached_dt = None

    def build_F(self, dt: float) -> NDArray[np.float64]:
        return build_constant_velocity_F(self.dim_x, self.pos_idx, self.vel_idx, dt)

    def build_Q(self, dt: float) -> NDArray[np.float64]:
        return build_dwna_Q(
            self.dim_x,
            self.pos_idx,
            self.vel_idx,
            self.sigma_a2,
            self.extra_q_diagonal,
            dt,
        )

    def apply(self, kf: KalmanFilter, dt: float) -> None:
        """Write ``F(dt)`` and ``Q(dt)`` onto *kf*."""
        kf.F = self.build_F(dt)
        kf.Q = self.build_Q(dt)

    def sync(
        self,
        kf: KalmanFilter,
        frame_step: float,
        nominal_frame_step: float = 1.0,
    ) -> None:
        """Update *kf* motion matrices for a predict step in frame units.

        ``nominal_frame_step`` is the reference step where tuned ``Q`` was
        calibrated (``1.0`` for SORT / ByteTrack). When it changes, σ_a² is
        re-derived from the current ``Q`` so gap scaling uses the tuned noise
        level.

        At the nominal step the caller's ``Q`` is kept unchanged (SORT /
        ByteTrack tuning is diagonal and must not be replaced by a DWNA block
        rebuild). Longer or shorter steps rebuild ``Q(frame_step)`` from the
        calibrated model.
        """
        if not np.isclose(self.calibration_dt, nominal_frame_step):
            self.calibrate_from_Q(kf.Q, calibration_dt=nominal_frame_step)

        kf.F = self.build_F(frame_step)

        if self.cached_dt is not None and np.isclose(frame_step, self.cached_dt):
            return

        if not np.isclose(frame_step, nominal_frame_step):
            kf.Q = self.build_Q(frame_step)
        elif self.reference_Q is not None:
            kf.Q = self.reference_Q.copy()
        self.cached_dt = frame_step

    def reset_cached_dt(self) -> None:
        """Forget the last synced ``dt`` (e.g. after restoring filter state)."""
        self.cached_dt = None


def init_constant_velocity_filter(
    dim_x: int,
    dim_z: int,
    pos_idx: NDArray[np.int64],
    vel_idx: NDArray[np.int64],
    measurement: np.ndarray,
) -> KalmanFilter:
    """Create a Kalman filter with reference ``F(1)``, identity ``H``, and initial state."""
    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf.F = build_constant_velocity_F(dim_x, pos_idx, vel_idx, 1.0)
    kf.H = np.eye(dim_z, dim_x, dtype=np.float64)
    kf.x[:dim_z] = np.asarray(measurement, dtype=np.float64).reshape((dim_z, 1))
    return kf
