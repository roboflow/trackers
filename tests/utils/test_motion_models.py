# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for transition matrix / ``Q`` builders in ``motion_models.py``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trackers.utils.kalman_filter import KalmanFilter
from trackers.utils.motion_models import (
    _NOMINAL_FRAME_STEP_TOLERANCE,
    _NOMINAL_FRAME_STEP_TOLERANCE_SECONDS,
    KalmanMotionModel,
    ScalableProcessNoise,
    constant_velocity_transition_matrix,
)

POS_1D = np.array([0], dtype=np.int64)
VEL_1D = np.array([1], dtype=np.int64)
POS_4D = np.array([0, 1, 2, 3], dtype=np.int64)
VEL_4D = np.array([4, 5, 6, 7], dtype=np.int64)
POS_XCYCSR = np.array([0, 1, 2], dtype=np.int64)
VEL_XCYCSR = np.array([4, 5, 6], dtype=np.int64)


def test_transition_mtx_scales_velocity_coupling_with_frame_step() -> None:
    """Velocity coupling in the transition matrix scales linearly with frame_step."""
    mtx1 = constant_velocity_transition_matrix(8, POS_4D, VEL_4D, 1.0)
    mtx2 = constant_velocity_transition_matrix(8, POS_4D, VEL_4D, 2.0)
    mtx_half = constant_velocity_transition_matrix(8, POS_4D, VEL_4D, 0.5)

    for v in VEL_4D:
        for j in range(8):
            assert mtx1[v, j] == mtx2[v, j] == mtx_half[v, j]

    for p, v in zip(POS_4D, VEL_4D):
        assert mtx1[p, v] == pytest.approx(1.0)
        assert mtx2[p, v] == pytest.approx(2.0)
        assert mtx_half[p, v] == pytest.approx(0.5)


def test_dwna_gap_noise_scales_with_frame_step_polynomial() -> None:
    gap = ScalableProcessNoise(
        dim_x=8,
        pos_idx=POS_4D,
        vel_idx=VEL_4D,
        baseline_Q=np.eye(8, dtype=np.float64) * 0.01,
        sigma_a2=np.ones(4, dtype=np.float64) * 0.01,
        extra_q_diagonal=np.ones(8, dtype=np.float64) * 0.01,
    )
    # Compare two gap steps (doubling) so both go through DWNA, not the baseline.
    Q1 = gap.build_Q(2.0)
    Q2 = gap.build_Q(4.0)

    for p, v in zip(POS_4D, VEL_4D):
        assert Q2[v, v] == pytest.approx(Q1[v, v] * 4.0)
        assert Q2[p, p] == pytest.approx(Q1[p, p] * 16.0)
        assert Q2[p, v] == pytest.approx(Q1[p, v] * 8.0)
        assert Q2[v, p] == pytest.approx(Q1[v, p] * 8.0)


def test_dwna_gap_noise_preserves_non_kinematic_diagonal() -> None:
    extra = np.ones(7, dtype=np.float64) * 0.01
    extra[3] = 7.5
    gap = ScalableProcessNoise(
        dim_x=7,
        pos_idx=POS_XCYCSR,
        vel_idx=VEL_XCYCSR,
        baseline_Q=np.diag(extra),
        sigma_a2=np.ones(3, dtype=np.float64) * 0.01,
        extra_q_diagonal=extra,
    )
    # Non-unit steps exercise the DWNA path; the non-kinematic diagonal is fixed.
    for frame_step in (0.5, 2.0):
        Q_built = gap.build_Q(frame_step)
        assert Q_built[3, 3] == pytest.approx(7.5), f"Q[3,3] not preserved at frame_step={frame_step}"


def test_build_Q_at_frame_step_1_returns_baseline() -> None:
    """``build_Q(1.0)`` returns the configured baseline Q unchanged (backward-compat gate)."""
    baseline = np.eye(8, dtype=np.float64) * 0.01
    gap = ScalableProcessNoise(
        dim_x=8,
        pos_idx=POS_4D,
        vel_idx=VEL_4D,
        baseline_Q=baseline.copy(),
        sigma_a2=np.ones(4, dtype=np.float64) * 0.01,
        extra_q_diagonal=np.ones(8, dtype=np.float64) * 0.01,
    )
    q1 = gap.build_Q(1.0)
    np.testing.assert_array_equal(q1, baseline)


def _noise_8d(baseline: np.ndarray) -> ScalableProcessNoise:
    return ScalableProcessNoise(
        dim_x=8,
        pos_idx=POS_4D,
        vel_idx=VEL_4D,
        baseline_Q=baseline.copy(),
        sigma_a2=np.ones(4, dtype=np.float64) * 0.01,
        extra_q_diagonal=np.diag(baseline).astype(np.float64).copy(),
    )


# Deviations a stream running at its nominal FPS actually produces. `frame_step`
# is `(t - t_prev) * frame_rate`, so it lands near 1.0 and never on it.
NEAR_NOMINAL_FRAME_STEPS = [
    pytest.param(1.0 + 5e-13, id="float-clock"),
    pytest.param(1.0 - 5e-13, id="float-clock-below"),
    pytest.param(1.0 + 2.0e-2, id="millisecond-clock-30fps"),
    pytest.param(1.0 + 4.1e-2, id="millisecond-clock-60fps"),
    pytest.param(1.0 - 4.1e-2, id="millisecond-clock-60fps-below"),
    pytest.param(1.0 + 1.0e-3, id="ntsc-29.97-declared-30"),
]


@pytest.mark.parametrize("frame_step", NEAR_NOMINAL_FRAME_STEPS)
def test_build_Q_returns_baseline_for_near_nominal_frame_step(frame_step: float) -> None:
    """A step off 1.0 by real clock error still counts as one nominal frame.

    Compared bit for bit rather than with ``approx``: an approximate check would
    pass on the rebuilt DWNA matrix this guards against, which differs from the
    configured Q by two orders of magnitude on the position block.
    """
    baseline = np.diag([16.0, 81.0, 16.0, 81.0, 0.25, 1.2656, 0.25, 1.2656])
    noise = _noise_8d(baseline)

    built = noise.build_Q(frame_step)

    assert built.tobytes() == baseline.tobytes()


FRAME_RATE_FOR_BOUNDARY_TEST = 60.0


def _edge_step(tolerance: float, sign: float, *, included: bool) -> float:
    """Return the float ``frame_step`` nearest ``1.0 + sign * tolerance``.

    ``1.0 + tolerance`` is not itself guaranteed to satisfy
    ``abs(frame_step - 1.0) <= tolerance`` in float64 (the same rounding trap
    that makes the literal ``1.1`` land outside a ``0.1`` band): adding and then
    subtracting ``1.0`` can round away from ``tolerance``. This walks the float
    representable just past the raw sum back toward ``1.0`` until the actual
    ``build_Q`` comparison would include it, so the probe matches the real
    boundary of the implementation, not an algebraic approximation of it.

    Args:
        tolerance: Band half-width in frame units.
        sign: ``1.0`` for the high edge, ``-1.0`` for the low edge.
        included: If ``True``, return the last included step; if ``False``,
            return the first excluded step just beyond it.

    Returns:
        The boundary ``frame_step`` value for the requested edge.
    """
    candidate = 1.0 + sign * tolerance
    while abs(candidate - 1.0) > tolerance:
        candidate = math.nextafter(candidate, 1.0)
    return candidate if included else math.nextafter(candidate, sign * math.inf)


@pytest.mark.parametrize(
    "frame_step,expect_baseline",
    [
        pytest.param(
            _edge_step(_NOMINAL_FRAME_STEP_TOLERANCE, 1.0, included=True),
            True,
            id="fallback-high-edge-included",
        ),
        pytest.param(
            _edge_step(_NOMINAL_FRAME_STEP_TOLERANCE, 1.0, included=False),
            False,
            id="fallback-high-edge-excluded",
        ),
        pytest.param(
            _edge_step(_NOMINAL_FRAME_STEP_TOLERANCE, -1.0, included=True),
            True,
            id="fallback-low-edge-included",
        ),
        pytest.param(
            _edge_step(_NOMINAL_FRAME_STEP_TOLERANCE, -1.0, included=False),
            False,
            id="fallback-low-edge-excluded",
        ),
    ],
)
def test_build_Q_fallback_tolerance_boundary_without_frame_rate(frame_step: float, expect_baseline: bool) -> None:
    """No ``frame_rate``: the fixed frame-unit band edge is inclusive at exactly the tolerance."""
    baseline = np.diag([16.0, 81.0, 16.0, 81.0, 0.25, 1.2656, 0.25, 1.2656])
    noise = _noise_8d(baseline)

    built = noise.build_Q(frame_step)

    is_baseline = built.tobytes() == baseline.tobytes()
    assert is_baseline is expect_baseline, (
        f"frame_step={frame_step!r} (no frame_rate) expected baseline={expect_baseline}"
    )


@pytest.mark.parametrize(
    "frame_step,expect_baseline",
    [
        pytest.param(
            _edge_step(_NOMINAL_FRAME_STEP_TOLERANCE_SECONDS * FRAME_RATE_FOR_BOUNDARY_TEST, 1.0, included=True),
            True,
            id="fps-invariant-high-edge-included",
        ),
        pytest.param(
            _edge_step(_NOMINAL_FRAME_STEP_TOLERANCE_SECONDS * FRAME_RATE_FOR_BOUNDARY_TEST, 1.0, included=False),
            False,
            id="fps-invariant-high-edge-excluded",
        ),
        pytest.param(
            _edge_step(_NOMINAL_FRAME_STEP_TOLERANCE_SECONDS * FRAME_RATE_FOR_BOUNDARY_TEST, -1.0, included=True),
            True,
            id="fps-invariant-low-edge-included",
        ),
        pytest.param(
            _edge_step(_NOMINAL_FRAME_STEP_TOLERANCE_SECONDS * FRAME_RATE_FOR_BOUNDARY_TEST, -1.0, included=False),
            False,
            id="fps-invariant-low-edge-excluded",
        ),
    ],
)
def test_build_Q_fps_invariant_tolerance_boundary_with_frame_rate(frame_step: float, expect_baseline: bool) -> None:
    """``frame_rate`` given: the fps-scaled band edge is inclusive at exactly ``tolerance * frame_rate``."""
    baseline = np.diag([16.0, 81.0, 16.0, 81.0, 0.25, 1.2656, 0.25, 1.2656])
    noise = _noise_8d(baseline)

    built = noise.build_Q(frame_step, frame_rate=FRAME_RATE_FOR_BOUNDARY_TEST)

    is_baseline = built.tobytes() == baseline.tobytes()
    assert is_baseline is expect_baseline, (
        f"frame_step={frame_step!r} frame_rate={FRAME_RATE_FOR_BOUNDARY_TEST} expected baseline={expect_baseline}"
    )


def test_build_Q_still_rescales_a_real_gap() -> None:
    """Steps outside the band keep the DWNA gap scaling untouched."""
    baseline = np.eye(8, dtype=np.float64) * 0.01
    noise = _noise_8d(baseline)

    for frame_step in (1.5, 2.0, 5.0, 15.0):
        built = noise.build_Q(frame_step)
        assert built.tobytes() != baseline.tobytes(), f"frame_step={frame_step} must take the gap path"

    for p, v in zip(POS_4D, VEL_4D):
        assert noise.build_Q(4.0)[v, v] == pytest.approx(noise.build_Q(2.0)[v, v] * 4.0)


def test_build_Q_alternating_nominal_and_gap_steps_leaves_no_stale_state() -> None:
    """Second occurrence and the return to the original state, not just the first transition."""
    baseline = np.diag([16.0, 81.0, 16.0, 81.0, 0.25, 1.2656, 0.25, 1.2656])
    noise = _noise_8d(baseline)

    gap_first = noise.build_Q(3.0)
    for _ in range(2):
        assert noise.build_Q(1.0 + 2.0e-2).tobytes() == baseline.tobytes()
        np.testing.assert_array_equal(noise.build_Q(3.0), gap_first)
        assert noise.build_Q(1.0).tobytes() == baseline.tobytes()


def test_build_Q_recalibrates_within_the_nominal_band() -> None:
    """A near-nominal step returns the *current* reference, not a cached one.

    BoT-SORT refreshes its scale-aware Q on every frame, so the band must track
    ``calibrate`` rather than pin the matrix seen on the first call.
    """
    first = np.eye(8, dtype=np.float64) * 0.01
    noise = _noise_8d(first)
    assert noise.build_Q(1.0 + 2.0e-2).tobytes() == first.tobytes()

    second = np.eye(8, dtype=np.float64) * 0.5
    noise.calibrate(second)

    assert noise.build_Q(1.0 + 2.0e-2).tobytes() == second.tobytes()


def test_build_Q_at_zero_frame_step_zeroes_kinematic_block_only() -> None:
    """frame_step=0.0 sits outside both tolerance bands: the DWNA kinematic block collapses
    to all-zero (dt2=dt3=dt4=0), but non-kinematic diagonal entries still come from the
    one-frame reference Q — this is not a blanket all-zero matrix.
    """
    extra = np.ones(7, dtype=np.float64) * 0.01
    extra[3] = 7.5
    noise = ScalableProcessNoise(
        dim_x=7,
        pos_idx=POS_XCYCSR,
        vel_idx=VEL_XCYCSR,
        baseline_Q=np.diag(extra),
        sigma_a2=np.ones(3, dtype=np.float64) * 0.01,
        extra_q_diagonal=extra,
    )

    built = noise.build_Q(0.0)

    kinematic_idx = np.array([*POS_XCYCSR, *VEL_XCYCSR])
    kinematic_block = built[np.ix_(kinematic_idx, kinematic_idx)]
    np.testing.assert_array_equal(kinematic_block, np.zeros((6, 6)))
    assert built[3, 3] == pytest.approx(7.5), "non-kinematic diagonal must survive frame_step=0.0"


def test_build_Q_at_negative_frame_step_documents_sign_inverted_dwna() -> None:
    """frame_step=-1.0 is not degenerate like 0.0: dt2=1, dt3=-1, dt4=1 build a real,
    non-zero, sign-inverted DWNA matrix. This pins current documented behavior rather than
    guarding a bug — ``build_Q`` does not validate ``frame_step`` and no shipped tracker
    calls it with a negative step.
    """
    extra = np.ones(7, dtype=np.float64) * 0.01
    extra[3] = 7.5
    sigma_a2 = np.ones(3, dtype=np.float64) * 0.01
    noise = ScalableProcessNoise(
        dim_x=7,
        pos_idx=POS_XCYCSR,
        vel_idx=VEL_XCYCSR,
        baseline_Q=np.diag(extra),
        sigma_a2=sigma_a2,
        extra_q_diagonal=extra,
    )
    zero_case = noise.build_Q(0.0)

    built = noise.build_Q(-1.0)

    assert not np.array_equal(built, zero_case), "frame_step=-1.0 must not collapse like frame_step=0.0"
    expected = np.zeros((7, 7), dtype=np.float64)
    expected[POS_XCYCSR, POS_XCYCSR] = sigma_a2 * 1.0 / 4.0  # dt4 = (-1)**4 = 1
    expected[POS_XCYCSR, VEL_XCYCSR] = sigma_a2 * -1.0 / 2.0  # dt3 = (-1)**3 = -1
    expected[VEL_XCYCSR, POS_XCYCSR] = sigma_a2 * -1.0 / 2.0
    expected[VEL_XCYCSR, VEL_XCYCSR] = sigma_a2 * 1.0  # dt2 = (-1)**2 = 1
    expected[3, 3] = extra[3]
    np.testing.assert_allclose(built, expected)


def test_sync_preserves_configured_q_at_unit_frame_step() -> None:
    """At frame_step=1.0, kf.process_noise must equal the configured Q exactly (backward-compat gate)."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    custom_Q = np.diag([0.01, 0.02])
    kf.process_noise = custom_Q.copy()
    model = KalmanMotionModel.from_filter(kf, POS_1D, VEL_1D)
    model.calibrate_from_process_noise(custom_Q)

    model.apply(kf, 1.0)

    # frame_step=1.0 returns the hand-tuned Q unchanged — no DWNA rescaling.
    np.testing.assert_array_equal(kf.process_noise, custom_Q)


def test_sync_gap_scales_q_relative_to_unit_frame_step() -> None:
    """Gap step must scale velocity variance as ``dt^2`` relative to ``frame_step = 1``."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    custom_Q = np.diag([0.01, 0.02])
    kf.process_noise = custom_Q.copy()
    model = KalmanMotionModel.from_filter(kf, POS_1D, VEL_1D)
    model.calibrate_from_process_noise(custom_Q)

    model.apply(kf, 1.0)
    q_unit = kf.process_noise.copy()
    model.apply(kf, 3.0)

    assert kf.process_noise[1, 1] == pytest.approx(custom_Q[1, 1] * 9.0)
    model.apply(kf, 1.0)
    # Returning to frame_step=1 restores the configured baseline Q exactly.
    np.testing.assert_allclose(kf.process_noise, q_unit, rtol=1e-12)
