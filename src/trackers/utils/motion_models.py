# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Build Kalman predict matrices (``F`` and ``Q``) for bbox trackers.

``KalmanFilter`` only runs the predict math (``x = F @ x``, etc.). This module
sets ``F`` and ``Q`` on the filter before each step.

``Q`` is *process noise*: extra uncertainty added on each predict — how much
the box is allowed to drift when there is no new detection. Each tracklet sets
``Q`` in ``_configure_noise()``.
Those values assume **one frame** between updates. ``frame_step=1.0`` is
that case; other values rescale ``F`` and ``Q`` for shorter or longer gaps.

``Q`` uses the standard constant-velocity + white-noise-acceleration (DWNA)
layout for gaps (``frame_step > 1``) and shorter-than-nominal steps
(``frame_step < 1``): velocity variance scales as ``Δt²``, position as ``Δt⁴``.
At ``frame_step == 1.0`` the original configured Q (set by ``_configure_noise``
via ``set_kf_covariances``) is returned unchanged — this preserves the
hand-tuned per-tracker noise and ensures backward compatibility when
``timestamp`` is omitted. Same block structure as
``filterpy.common.discrete_white_noise`` — see the filterpy docs or source if
you want the formulas side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from trackers.utils.kalman_filter import KalmanFilter


def constant_velocity_F(
    dim_x: int,
    pos_idx: NDArray[np.int64],
    vel_idx: NDArray[np.int64],
    frame_step: float,
) -> NDArray[np.float64]:
    """Constant velocity: each position row picks up ``velocity * frame_step``."""
    F = np.eye(dim_x, dtype=np.float64)
    for p, v in zip(pos_idx, vel_idx, strict=True):
        F[int(p), int(v)] = frame_step
    return F


@dataclass
class ScalableProcessNoise:
    """Store the tracker's one-frame ``Q`` and scale it for any gap.

    On tracklet creation, ``_configure_noise()`` sets ``Q`` with values that
    work when exactly one frame passes between updates (see ``SORTTracklet``,
    ``ByteTrackTracklet``, etc.). ``calibrate`` extracts the per-axis
    acceleration variance ``sigma_a2`` from that reference ``Q`` and stores the
    DWNA-at-1 layout as ``baseline_Q``.

    At ``frame_step == 1.0``, ``build_Q`` returns the original configured Q
    stored in ``baseline_Q`` — preserving hand-tuned per-tracker noise and
    backward compatibility. For other frame steps, DWNA scaling is applied:
    smaller steps (``frame_step < 1``) shrink uncertainty; larger steps
    (``frame_step > 1``) grow it. Frozen entries (e.g. aspect ratio in XCYCSR)
    stay at the values from ``_configure_noise()``.
    """

    dim_x: int
    pos_idx: NDArray[np.int64]
    vel_idx: NDArray[np.int64]
    baseline_Q: NDArray[np.float64]
    sigma_a2: NDArray[np.float64]
    extra_q_diagonal: NDArray[np.float64]

    def calibrate(self, Q: np.ndarray) -> None:
        """Calibrate ``sigma_a2`` from ``Q`` and store the configured-Q reference.

        Called from ``set_kf_covariances``. The velocity diagonal of ``Q``
        defines the per-axis acceleration variance used when ``frame_step != 1``.
        ``baseline_Q`` stores ``Q`` itself; ``build_Q(1.0)`` returns it directly
        so the hand-tuned one-frame noise is preserved exactly.
        """
        self.sigma_a2 = np.asarray([float(Q[v, v]) for v in self.vel_idx], dtype=np.float64)
        self.extra_q_diagonal = np.diag(Q).astype(np.float64).copy()
        self.baseline_Q = Q.copy()

    def build_Q(self, frame_step: float) -> NDArray[np.float64]:
        """Return process noise ``Q`` for *frame_step*.

        At ``frame_step == 1.0`` the original configured Q (``baseline_Q``) is
        returned to preserve hand-tuned noise and backward compatibility.
        For any other step the DWNA formula is applied.
        """
        if frame_step == 1.0:
            return self.baseline_Q.copy()
        return self._dwna(frame_step)

    def _dwna(self, frame_step: float) -> NDArray[np.float64]:
        """Build gap-scaled ``Q`` (white-noise acceleration blocks per axis)."""
        Q = np.zeros((self.dim_x, self.dim_x), dtype=np.float64)
        dt2 = frame_step * frame_step
        dt3 = dt2 * frame_step
        dt4 = dt2 * dt2
        kinematic = set(int(i) for i in self.pos_idx) | set(int(i) for i in self.vel_idx)
        for k, (p, v) in enumerate(zip(self.pos_idx, self.vel_idx, strict=True)):
            p_i = int(p)
            v_i = int(v)
            sa2 = float(self.sigma_a2[k])
            Q[p_i, p_i] = sa2 * dt4 / 4.0
            Q[p_i, v_i] = sa2 * dt3 / 2.0
            Q[v_i, p_i] = sa2 * dt3 / 2.0
            Q[v_i, v_i] = sa2 * dt2
        for i in range(self.dim_x):
            if i not in kinematic:
                Q[i, i] = float(self.extra_q_diagonal[i])
        return Q


@dataclass
class KalmanMotionModel:
    """Write ``F`` and ``Q`` onto a ``KalmanFilter`` for one predict step."""

    dim_x: int
    pos_idx: NDArray[np.int64]
    vel_idx: NDArray[np.int64]
    process_noise: ScalableProcessNoise
    cached_step: float | None = field(default=None, init=False)

    @classmethod
    def from_filter(
        cls,
        kf: KalmanFilter,
        pos_idx: NDArray[np.int64],
        vel_idx: NDArray[np.int64],
    ) -> KalmanMotionModel:
        """Create a motion model; uses the reference ``Q`` from the filter."""
        dim_x = kf.dim_x
        return cls(
            dim_x=dim_x,
            pos_idx=pos_idx,
            vel_idx=vel_idx,
            process_noise=ScalableProcessNoise(
                dim_x=dim_x,
                pos_idx=pos_idx,
                vel_idx=vel_idx,
                baseline_Q=kf.Q.copy(),
                sigma_a2=np.ones(len(pos_idx), dtype=np.float64),
                extra_q_diagonal=np.diag(kf.Q).astype(np.float64).copy(),
            ),
        )

    def calibrate_from_Q(self, Q: np.ndarray) -> None:
        """Update the one-frame reference after ``Q`` changes in ``set_kf_covariances``."""
        self.process_noise.calibrate(Q)
        self.cached_step = None

    def apply(self, kf: KalmanFilter, frame_step: float) -> None:
        """Set ``kf.F`` and ``kf.Q`` for *frame_step* (``Q`` is cached per step)."""
        kf.F = constant_velocity_F(self.dim_x, self.pos_idx, self.vel_idx, frame_step)
        if self.cached_step is not None and np.isclose(frame_step, self.cached_step):
            return
        kf.Q = self.process_noise.build_Q(frame_step)
        self.cached_step = frame_step

    def reset_cache(self) -> None:
        """Clear cached step (e.g. after restoring filter state)."""
        self.cached_step = None


def init_constant_velocity_filter(
    dim_x: int,
    dim_z: int,
    pos_idx: NDArray[np.int64],
    vel_idx: NDArray[np.int64],
    measurement: np.ndarray,
) -> KalmanFilter:
    """New constant-velocity filter with ``F(1)``, identity ``H``, and initial state."""
    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf.F = constant_velocity_F(dim_x, pos_idx, vel_idx, 1.0)
    kf.H = np.eye(dim_z, dim_x, dtype=np.float64)
    kf.x[:dim_z] = np.asarray(measurement, dtype=np.float64).reshape((dim_z, 1))
    return kf
