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


def constant_velocity_transition_matrix(
    dim_x: int,
    pos_idx: NDArray[np.int64],
    vel_idx: NDArray[np.int64],
    frame_step: float,
) -> NDArray[np.float64]:
    """Build the constant-velocity state-transition matrix.

    Each position state is coupled to its paired velocity state by
    ``frame_step``; all other entries remain as in the identity matrix.

    Args:
        dim_x: State vector dimension.
        pos_idx: Indices of position states in the state vector.
        vel_idx: Indices of velocity states paired with ``pos_idx``
            (must have the same length as ``pos_idx``).
        frame_step: Elapsed time in frame units; the off-diagonal coupling
            ``transition_matrix[pos, vel]`` is set to this value.

    Returns:
        Transition matrix of shape ``(dim_x, dim_x)``.
    """
    mtx = np.eye(dim_x, dtype=np.float64)
    for p, v in zip(pos_idx, vel_idx, strict=True):
        mtx[int(p), int(v)] = frame_step
    return mtx


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
    _nonkinematic_idx: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        kinematic = {int(i) for i in self.pos_idx} | {int(i) for i in self.vel_idx}
        self._nonkinematic_idx = [i for i in range(self.dim_x) if i not in kinematic]

    def calibrate(self, Q: np.ndarray) -> None:
        """Extract per-axis acceleration variance from Q and store the reference.

        Called from ``set_kf_covariances``. The velocity-diagonal entries of
        ``Q`` define ``sigma_a2``, which is used to scale noise for
        ``frame_step != 1.0`` via DWNA. ``baseline_Q`` stores ``Q`` itself so
        that ``build_Q(1.0)`` returns it directly, preserving hand-tuned
        one-frame noise exactly.

        Args:
            Q: Configured one-frame process noise matrix of shape
                ``(dim_x, dim_x)``, as produced by ``_configure_noise`` via
                ``set_kf_covariances``.
        """
        self.sigma_a2 = np.asarray([float(Q[v, v]) for v in self.vel_idx], dtype=np.float64)
        self.extra_q_diagonal = np.diag(Q).astype(np.float64).copy()
        self.baseline_Q = Q.copy()
        kinematic = {int(i) for i in self.pos_idx} | {int(i) for i in self.vel_idx}
        self._nonkinematic_idx = [i for i in range(self.dim_x) if i not in kinematic]

    def build_Q(self, frame_step: float) -> NDArray[np.float64]:
        """Return process noise Q scaled for the given frame step.

        At ``frame_step == 1.0`` the original configured Q (``baseline_Q``) is
        returned unchanged to preserve hand-tuned noise and backward
        compatibility. For any other step the DWNA formula is applied via
        ``_dwna``.

        Args:
            frame_step: Elapsed time in frame units; ``1.0`` returns
                ``baseline_Q`` exactly.

        Returns:
            Process noise matrix of shape ``(dim_x, dim_x)``.
        """
        if frame_step == 1.0:
            return self.baseline_Q.copy()
        return self._dwna(frame_step)

    def _dwna(self, frame_step: float) -> NDArray[np.float64]:
        """Build gap-scaled Q using the DWNA (discrete white noise acceleration) model.

        Args:
            frame_step: Elapsed time in frame units.

        Returns:
            Process noise matrix of shape ``(dim_x, dim_x)`` with kinematic
            blocks scaled by ``frame_step`` and non-kinematic diagonal entries
            taken from the one-frame reference.
        """
        Q = np.zeros((self.dim_x, self.dim_x), dtype=np.float64)
        dt2 = frame_step * frame_step
        dt3 = dt2 * frame_step
        dt4 = dt2 * dt2
        for k, (p, v) in enumerate(zip(self.pos_idx, self.vel_idx, strict=True)):
            p_i = int(p)
            v_i = int(v)
            sa2 = float(self.sigma_a2[k])
            Q[p_i, p_i] = sa2 * dt4 / 4.0
            Q[p_i, v_i] = sa2 * dt3 / 2.0
            Q[v_i, p_i] = sa2 * dt3 / 2.0
            Q[v_i, v_i] = sa2 * dt2
        for i in self._nonkinematic_idx:
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
    _cached_transition_mtx: NDArray[np.float64] | None = field(default=None, init=False)
    _cached_process_noise: NDArray[np.float64] | None = field(default=None, init=False)

    @classmethod
    def from_filter(
        cls,
        kf: KalmanFilter,
        pos_idx: NDArray[np.int64],
        vel_idx: NDArray[np.int64],
    ) -> KalmanMotionModel:
        """Create a motion model seeded from an already-configured Kalman filter.

        Uses the filter's current ``Q`` as the one-frame reference noise.
        Call ``calibrate_from_Q`` later if ``Q`` is updated via ``set_kf_covariances``.

        Args:
            kf: ``KalmanFilter`` with ``Q`` already set to the one-frame reference.
            pos_idx: Indices of position states in the state vector.
            vel_idx: Indices of velocity states paired with ``pos_idx``.

        Returns:
            ``KalmanMotionModel`` ready to apply ``transition_mtx`` and ``Q`` per predict step.
        """
        dim_x = kf.dim_x
        return cls(
            dim_x=dim_x,
            pos_idx=pos_idx,
            vel_idx=vel_idx,
            process_noise=ScalableProcessNoise(
                dim_x=dim_x,
                pos_idx=pos_idx,
                vel_idx=vel_idx,
                baseline_Q=kf.process_noise.copy(),
                sigma_a2=np.ones(len(pos_idx), dtype=np.float64),
                extra_q_diagonal=np.diag(kf.process_noise).astype(np.float64).copy(),
            ),
        )

    def calibrate_from_process_noise(self, process_noise: np.ndarray) -> None:
        """Update the one-frame noise reference when process noise changes.

        Call this after ``set_kf_covariances`` updates ``process_noise`` on the
        filter so that ``build_Q`` and future cached steps use the new reference.

        Args:
            process_noise: Reference one-frame process noise matrix, as produced
                by ``set_kf_covariances``. Shape must be ``(dim_x, dim_x)``.
        """
        self.process_noise.calibrate(process_noise)
        self.cached_step = None

    def apply(self, kf: KalmanFilter, frame_step: float) -> None:
        """Set transition_matrix and Q on a Kalman filter for the given frame step.

        Both matrices are cached per unique step value to avoid redundant
        computation when the same step is repeated across consecutive frames.

        Args:
            kf: ``KalmanFilter`` to update in-place.
            frame_step: Elapsed time in frame units for this predict step.
                Use ``1.0`` for a single nominal frame; larger values for gaps.
        """
        if (
            self.cached_step is not None
            and frame_step == self.cached_step
            and self._cached_transition_mtx is not None
            and self._cached_process_noise is not None
        ):
            kf.transition_mtx = self._cached_transition_mtx
            kf.process_noise = self._cached_process_noise
            return
        kf.transition_mtx = constant_velocity_transition_matrix(self.dim_x, self.pos_idx, self.vel_idx, frame_step)
        kf.process_noise = self.process_noise.build_Q(frame_step)
        self._cached_transition_mtx = kf.transition_mtx
        self._cached_process_noise = kf.process_noise
        self.cached_step = frame_step

    def reset_cache(self) -> None:
        """Clear cached step and matrices.

        Call after restoring a filter state (e.g. via ``set_state``) to ensure
        the next ``apply`` recomputes transition_matrix and Q rather than reusing stale values.
        """
        self.cached_step = None
        self._cached_transition_mtx = None
        self._cached_process_noise = None


def init_constant_velocity_filter(
    dim_x: int,
    dim_z: int,
    pos_idx: NDArray[np.int64],
    vel_idx: NDArray[np.int64],
    measurement: np.ndarray,
) -> KalmanFilter:
    """Create a constant-velocity Kalman filter with F(1), identity H, and initial state.

    Initialises ``F`` for one nominal frame step (``frame_step = 1.0``),
    sets ``H = I[:dim_z, :dim_x]``, and seeds the state vector with the
    first measurement.

    Args:
        dim_x: Full state vector dimension (positions + velocities).
        dim_z: Measurement dimension (number of observed coordinates).
        pos_idx: Indices of position states in the state vector.
        vel_idx: Indices of velocity states paired with ``pos_idx``.
        measurement: Initial measurement of shape ``(dim_z,)`` used to
            seed the filter state.

    Returns:
        Configured ``KalmanFilter`` with ``F``, ``H``, and initial state set.
        Noise matrices (``Q``, ``R``, ``P``) are left at their filter defaults
        and must be set by the caller via ``set_kf_covariances``.
    """
    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf.transition_mtx = constant_velocity_transition_matrix(dim_x, pos_idx, vel_idx, 1.0)
    kf.observation_mtx = np.eye(dim_z, dim_x, dtype=np.float64)
    kf.state[:dim_z] = np.asarray(measurement, dtype=np.float64).reshape((dim_z, 1))
    return kf
