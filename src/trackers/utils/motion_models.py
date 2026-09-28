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
At a *near-nominal* ``frame_step`` — within a tolerance band of ``1.0`` — the
original configured Q (set by ``_configure_noise`` via ``set_kf_covariances``)
is returned unchanged. This preserves the hand-tuned per-tracker noise, keeps
backward compatibility when ``timestamp`` is omitted, and makes a timestamped
stream at its nominal FPS behave like the fixed-rate one: ``frame_step`` is a
product of a subtraction, so it lands near ``1.0`` without ever hitting it. The
band's half-width is time-based (``_NOMINAL_FRAME_STEP_TOLERANCE_SECONDS *
frame_rate``) so it stays fps-invariant; callers that do not supply a frame
rate fall back to the fixed frame-unit band ``_NOMINAL_FRAME_STEP_TOLERANCE``.
Same block structure as ``filterpy.common.discrete_white_noise`` — see the
filterpy docs or source if you want the formulas side by side.
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


# Half-width, in wall-clock seconds, of the near-nominal frame_step band.
# build_Q multiplies this by frame_rate to get the frame-unit half-width, so
# the band absorbs a fixed physical-clock-error budget at any FPS. 4ms covers
# the two error sources: ms-clock rounding (bounded ~1ms) + capture jitter
# (~2ms), with ~33% headroom on the ~3ms worst-case sum.
_NOMINAL_FRAME_STEP_TOLERANCE_SECONDS = 0.004

# Fallback half-width, in frame units, used when no frame_rate is given (e.g.
# OC-SORT's ORU sub-step replay, which is frame-unit not wall-clock). Fixed in
# frame units while the physical error it absorbs grows with FPS, so this path
# has a practical ~50fps ceiling; the time-based tolerance above has none.
_NOMINAL_FRAME_STEP_TOLERANCE = 0.1


@dataclass
class ScalableProcessNoise:
    """Store the tracker's one-frame ``Q`` and scale it for any gap.

    On tracklet creation, ``_configure_noise()`` sets ``Q`` with values that
    work when exactly one frame passes between updates (see ``SORTTracklet``,
    ``ByteTrackTracklet``, etc.). ``calibrate`` extracts the per-axis
    acceleration variance ``sigma_a2`` from that reference ``Q`` and stores the
    DWNA-at-1 layout as ``baseline_Q``.

    Within a near-nominal band of ``frame_step = 1.0``, ``build_Q`` returns the
    original configured Q stored in ``baseline_Q`` — preserving hand-tuned
    per-tracker noise and backward compatibility. The band half-width is
    time-based when a ``frame_rate`` is supplied (fps-invariant) and falls back
    to a fixed frame-unit width otherwise; see ``build_Q``. Outside that band
    the step is a real gap and DWNA scaling is applied: smaller steps shrink
    uncertainty; larger steps grow it. Frozen entries (e.g. aspect ratio in
    XCYCSR) stay at the values from ``_configure_noise()``.
    """

    dim_x: int
    pos_idx: NDArray[np.int64]
    vel_idx: NDArray[np.int64]
    baseline_Q: NDArray[np.float64]
    sigma_a2: NDArray[np.float64]
    extra_q_diagonal: NDArray[np.float64]
    _nonkinematic_idx: list[int] = field(default_factory=list, init=False)
    _needs_calibration: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        kinematic = {int(i) for i in self.pos_idx} | {int(i) for i in self.vel_idx}
        self._nonkinematic_idx = [i for i in range(self.dim_x) if i not in kinematic]

    def calibrate(self, Q: np.ndarray) -> None:
        """Store the new one-frame reference ``Q`` and defer the extraction.

        Called from ``set_kf_covariances``. ``baseline_Q`` stores ``Q`` itself
        so that ``build_Q(1.0)`` returns it directly, preserving hand-tuned
        one-frame noise exactly. The DWNA calibration (``sigma_a2``,
        ``extra_q_diagonal``) is deferred until the first ``build_Q`` with a
        non-unit frame step, keeping per-frame noise refreshes (e.g. BoT-SORT
        scale-aware noise) cheap on the fixed-rate path.

        Args:
            Q: Configured one-frame process noise matrix of shape
                ``(dim_x, dim_x)``, as produced by ``_configure_noise`` via
                ``set_kf_covariances``.
        """
        self.baseline_Q = Q.copy()
        self._needs_calibration = True

    def _ensure_calibrated(self) -> None:
        """Extract per-axis acceleration variance from the stored reference Q.

        The velocity-diagonal entries of ``baseline_Q`` define ``sigma_a2``, which scales noise for ``frame_step !=
        1.0`` via DWNA. Runs only when a ``calibrate`` call marked the extraction as pending.
        """
        if not self._needs_calibration:
            return
        Q = self.baseline_Q
        self.sigma_a2 = Q[self.vel_idx, self.vel_idx].astype(np.float64)
        self.extra_q_diagonal = np.diag(Q).astype(np.float64).copy()
        self._needs_calibration = False

    def build_Q(self, frame_step: float, frame_rate: float | None = None) -> NDArray[np.float64]:
        """Return process noise Q scaled for the given frame step.

        Within a near-nominal band of ``1.0`` the step counts as one nominal
        frame and the configured Q (``baseline_Q``) is returned unchanged,
        preserving hand-tuned noise. Outside that band the step is a real gap
        and the DWNA formula is applied via ``_dwna``.

        The band is what makes dynamic-rate mode line up with fixed-rate mode.
        ``frame_step`` comes from ``(t - t_prev) * frame_rate``, so a stream at
        its nominal FPS lands *near* ``1.0``, never on it; an exact comparison
        would send every such step down the gap path, where ``_dwna`` rebuilds Q
        from the velocity diagonal alone and drops the configured position noise.

        The band half-width adapts to ``frame_rate`` so it stays fps-invariant.
        The physical error a nominal stream carries (ms-clock rounding, capture
        jitter) is bounded in *wall-clock seconds*, but expressed in *frame
        units* it grows with FPS, so a fixed frame-unit band would break at high
        FPS. Multiplying a seconds budget by ``frame_rate`` makes the error and
        the band scale together:

        - ``frame_rate`` given (the normal tracker path):
          ``tolerance = _NOMINAL_FRAME_STEP_TOLERANCE_SECONDS * frame_rate`` —
          4 ms of wall-clock slack at any FPS.
        - ``frame_rate`` absent (e.g. OC-SORT ORU sub-steps, which are frame-unit
          replay steps): the fixed fallback ``_NOMINAL_FRAME_STEP_TOLERANCE``
          (0.1 frame units), whose own ~50 FPS ceiling is documented on that
          constant.

        The band is symmetric in ``frame_step``, but the DWNA growth it forgoes
        is not: treating ``1.0 + tol`` as nominal skips ``(1 + tol)**4``
        position-variance growth while ``1.0 - tol`` skips ``(1 - tol)**4``
        shrink, so the upper edge is marginally worse-behaved than the lower —
        immaterial this close to ``1.0``.

        What this does **not** cover: capture jitter wider than the tolerance
        still takes the gap path, and lost-track pruning remains asymmetric
        between the two modes (dynamic-rate mode drops time-expired tracks
        before association, while fixed-rate mode enforces its frame budget
        after association), so the two modes can still differ by one
        ``lost_track_buffer`` frame.

        Args:
            frame_step: Elapsed time in frame units; a value within the
                near-nominal band of ``1.0`` returns ``baseline_Q`` unchanged.
            frame_rate: Tracker reference FPS. When given (and positive) the band
                half-width is ``_NOMINAL_FRAME_STEP_TOLERANCE_SECONDS *
                frame_rate`` frame units, keeping it fps-invariant. When ``None``
                the fixed fallback ``_NOMINAL_FRAME_STEP_TOLERANCE`` is used.

        Returns:
            Process noise matrix of shape ``(dim_x, dim_x)``.
        """
        if frame_rate is not None and frame_rate > 0:
            tolerance = _NOMINAL_FRAME_STEP_TOLERANCE_SECONDS * frame_rate
        else:
            tolerance = _NOMINAL_FRAME_STEP_TOLERANCE
        if abs(frame_step - 1.0) <= tolerance:
            return self.baseline_Q.copy()
        self._ensure_calibrated()
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
        Only the cached ``Q`` is invalidated: the cached transition matrix
        depends solely on the frame step, so it survives noise-only changes
        (e.g. BoT-SORT's per-frame scale-aware refresh).

        Args:
            process_noise: Reference one-frame process noise matrix, as produced
                by ``set_kf_covariances``. Shape must be ``(dim_x, dim_x)``.
        """
        self.process_noise.calibrate(process_noise)
        self._cached_process_noise = None

    def apply(self, kf: KalmanFilter, frame_step: float, frame_rate: float | None = None) -> None:
        """Set transition_matrix and Q on a Kalman filter for the given frame step.

        Both matrices are cached to avoid redundant computation. The transition
        matrix is cached per unique step value; ``Q`` is additionally
        invalidated by ``calibrate_from_process_noise`` and rebuilt from the
        current reference on the next call.

        The transition-matrix cache key stays keyed on ``frame_step`` alone and
        is compared exactly (``frame_step != self.cached_step``). Unlike Q's
        near-nominal band, this exactness is intentional: the cache key is a
        memoization key, and ``F`` genuinely varies with the precise
        ``frame_step`` (the coupling term is set to it directly), so an exact
        key is correct — a near-nominal ``frame_step`` only costs a cache miss,
        never a wrong matrix. ``frame_rate`` is constant per tracker instance,
        not a per-call variable, so it does not enter the cache key.

        Args:
            kf: ``KalmanFilter`` to update in-place.
            frame_step: Elapsed time in frame units for this predict step.
                Use ``1.0`` for a single nominal frame; larger values for gaps.
            frame_rate: Tracker reference FPS, forwarded to
                ``ScalableProcessNoise.build_Q`` to keep the near-nominal Q band
                fps-invariant. ``None`` (fixed-rate mode or unavailable) selects
                the fixed frame-unit fallback band.
        """
        if frame_step != self.cached_step or self._cached_transition_mtx is None:
            self._cached_transition_mtx = constant_velocity_transition_matrix(
                self.dim_x, self.pos_idx, self.vel_idx, frame_step
            )
            self.cached_step = frame_step
            self._cached_process_noise = None
        if self._cached_process_noise is None:
            self._cached_process_noise = self.process_noise.build_Q(frame_step, frame_rate)
        kf.transition_mtx = self._cached_transition_mtx
        kf.process_noise = self._cached_process_noise

    def reset_cache(self) -> None:
        """Clear cached step and matrices.

        Call after restoring a filter state (e.g. via ``set_state``) to ensure the next ``apply`` recomputes
        transition_matrix and Q rather than reusing stale values.
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
