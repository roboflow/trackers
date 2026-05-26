# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the variable-dt Kalman filter and DWNA state-estimator builders.

PR 1 of the dynamic-frame-rate refactor; see
`docs/design/dynamic-frame-rate.md` for the full specification.
"""

from __future__ import annotations

import numpy as np
import pytest

from trackers.utils.kalman_filter import KalmanFilter
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCSRStateEstimator,
    XCYCWHStateEstimator,
    XYXYStateEstimator,
)

ALL_ESTIMATORS: list[type[BaseStateEstimator]] = [
    XYXYStateEstimator,
    XCYCSRStateEstimator,
    XCYCWHStateEstimator,
]


# ---------------------------------------------------------------------------
# Backward-compatibility tests
# ---------------------------------------------------------------------------


def test_predict_without_builders_ignores_dt() -> None:
    """A bare KalmanFilter (no builders registered) must ignore `dt` entirely.

    Existing tracker code never registers builders directly, so its behaviour
    is byte-for-byte preserved regardless of any `dt` value.
    """
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.0], [1.0]])  # position 0, velocity 1
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # CV with dt = 1
    kf.Q = np.array([[0.1, 0.0], [0.0, 0.1]])
    kf.predict(dt=999.0)  # absurd dt; with no builders, must use stored F, Q

    np.testing.assert_allclose(kf.x, np.array([[1.0], [1.0]]))


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_predict_default_dt_matches_legacy(estimator_cls: type[BaseStateEstimator]) -> None:
    """`predict()` (no args) and `predict(1.0)` must produce identical state.

    This is the byte-for-byte backward-compat guarantee — today's call sites
    that invoke `predict()` with no args must not change behaviour after the
    builder registration.
    """
    bbox = np.array([10.0, 20.0, 30.0, 40.0])

    est_a = estimator_cls(bbox.copy())
    est_b = estimator_cls(bbox.copy())

    for _ in range(5):
        est_a.predict()
        est_b.predict(dt=1.0)

    np.testing.assert_allclose(est_a.kf.x, est_b.kf.x, atol=1e-12)
    np.testing.assert_allclose(est_a.kf.P, est_b.kf.P, atol=1e-12)


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_first_predict_with_dt_one_preserves_reference_Q(
    estimator_cls: type[BaseStateEstimator],
) -> None:
    """The very first `predict(1.0)` must not rebuild the caller-supplied Q.

    This is the "preserve calibration" rule from
    `docs/design/dynamic-frame-rate.md` §4.3 and the
    `_cached_dt is None` branch in `KalmanFilter.predict`.
    """
    bbox = np.array([10.0, 20.0, 30.0, 40.0])
    est = estimator_cls(bbox)

    # Install a custom Q with a non-DWNA pattern (off-diagonals zero, all
    # diagonal entries unequal). If the builder were called on the first
    # predict, this exact matrix would not survive.
    custom_Q = np.diag(np.arange(1, est.kf.dim_x + 1, dtype=np.float64) * 0.01)
    est.set_kf_covariances(Q=custom_Q)
    Q_before = est.kf.Q.copy()
    est.predict(dt=1.0)

    np.testing.assert_allclose(est.kf.Q, Q_before, atol=1e-12)


# ---------------------------------------------------------------------------
# DWNA structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_build_F_is_constant_velocity(estimator_cls: type[BaseStateEstimator]) -> None:
    """Velocity columns of F scale linearly with dt; identity blocks unchanged."""
    bbox = np.array([10.0, 20.0, 30.0, 40.0])
    est = estimator_cls(bbox)
    F1 = est.build_F(1.0)
    F2 = est.build_F(2.0)
    F_half = est.build_F(0.5)

    # The identity block on the velocity sub-diagonal is preserved
    pos_idx, vel_idx = est._kinematic_indices()
    n = est.kf.dim_x
    for v in vel_idx:
        for j in range(n):
            assert F1[v, j] == F2[v, j] == F_half[v, j], f"velocity row {v} col {j} changed across dt"

    # The kinematic coupling F[p, v] scales with dt
    for p, v in zip(pos_idx, vel_idx):
        assert F1[p, v] == pytest.approx(1.0)
        assert F2[p, v] == pytest.approx(2.0)
        assert F_half[p, v] == pytest.approx(0.5)


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_build_Q_velocity_diagonal_back_calibration(
    estimator_cls: type[BaseStateEstimator],
) -> None:
    """`build_Q(1.0)[v, v]` equals the velocity diagonal of the caller-supplied Q.

    This is the central back-calibration guarantee (§4.3): at the reference
    dt = 1, the velocity diagonals of `build_Q(dt)` exactly reproduce the
    caller's tuning. (Position diagonals differ because today's code never
    set them per a DWNA model; that small discrepancy is documented.)
    """
    bbox = np.array([10.0, 20.0, 30.0, 40.0])
    est = estimator_cls(bbox)

    # Set a Q with distinguishable velocity diagonals so we can detect any
    # mis-assignment of σ_a² to coordinates.
    base = np.eye(est.kf.dim_x, dtype=np.float64) * 0.01
    pos_idx, vel_idx = est._kinematic_indices()
    distinct = np.array([1.0, 2.0, 3.0, 4.0])[: len(vel_idx)]
    for k, v in enumerate(vel_idx):
        base[v, v] = float(distinct[k]) * 0.01
    est.set_kf_covariances(Q=base)

    Q_built = est.build_Q(1.0)
    for k, v in enumerate(vel_idx):
        assert Q_built[v, v] == pytest.approx(distinct[k] * 0.01)


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_build_Q_scales_with_dt_polynomial(
    estimator_cls: type[BaseStateEstimator],
) -> None:
    """`Q(dt)` blocks follow the DWNA polynomial: dt⁴/4, dt³/2, dt²."""
    bbox = np.array([10.0, 20.0, 30.0, 40.0])
    est = estimator_cls(bbox)
    est.set_kf_covariances(Q=np.eye(est.kf.dim_x, dtype=np.float64) * 0.01)

    Q1 = est.build_Q(1.0)
    Q2 = est.build_Q(2.0)

    pos_idx, vel_idx = est._kinematic_indices()
    for p, v in zip(pos_idx, vel_idx):
        # Velocity diagonal scales as dt²; ratio at dt=2 vs dt=1 is 4.
        assert Q2[v, v] == pytest.approx(Q1[v, v] * 4.0)
        # Position diagonal scales as dt⁴; ratio is 16.
        assert Q2[p, p] == pytest.approx(Q1[p, p] * 16.0)
        # Off-diagonals scale as dt³; ratio is 8.
        assert Q2[p, v] == pytest.approx(Q1[p, v] * 8.0)
        assert Q2[v, p] == pytest.approx(Q1[v, p] * 8.0)


def test_build_Q_xcycsr_preserves_aspect_ratio_diagonal() -> None:
    """The aspect-ratio random-walk variance (Q[3,3]) survives rebuild.

    XCYCSR has a non-kinematic dimension (aspect ratio) whose Q diagonal is
    set independently from the constant-velocity DWNA model. `build_Q(dt)`
    must restore it from `_extra_q_diagonal` rather than zero it out.
    """
    bbox = np.array([10.0, 20.0, 30.0, 40.0])
    est = XCYCSRStateEstimator(bbox)
    Q_custom = np.eye(7, dtype=np.float64) * 0.01
    Q_custom[3, 3] = 7.5  # arbitrary, distinctive aspect-ratio random-walk variance
    est.set_kf_covariances(Q=Q_custom)

    for dt in (0.5, 1.0, 2.0):
        Q_built = est.build_Q(dt)
        assert Q_built[3, 3] == pytest.approx(7.5), f"Q[3,3] not preserved at dt={dt}"


# ---------------------------------------------------------------------------
# End-to-end variable-dt sanity test
# ---------------------------------------------------------------------------


def test_synthetic_cv_trajectory_recovers_velocity_under_variable_dt() -> None:
    """A constant-velocity ground truth fed at non-uniform dts should be tracked.

    Generates a synthetic 1D constant-velocity trajectory with irregular
    sampling intervals, runs a manually-configured Kalman filter with the
    DWNA F(dt)/Q(dt) builders, and checks that the estimated velocity and
    position converge to the true values. This is the "is the variable-dt
    machinery actually wired correctly" sanity check.
    """
    rng = np.random.default_rng(seed=42)

    true_v = 3.0  # units per second
    true_p0 = 5.0

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[true_p0 + rng.normal(0.0, 0.5)], [true_v + rng.normal(0.0, 1.0)]])
    kf.P = np.eye(2) * 10.0
    kf.H = np.array([[1.0, 0.0]])
    kf.R = np.array([[0.05]])

    sigma_a2 = 0.5

    def F_builder(dt: float) -> np.ndarray:
        return np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)

    def Q_builder(dt: float) -> np.ndarray:
        return sigma_a2 * np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=np.float64)

    kf.set_motion_model_builders(F_builder, Q_builder)

    # Non-uniform dts in [0.02, 0.5] seconds.
    dts = rng.uniform(0.02, 0.5, size=200)
    t = 0.0
    for dt in dts:
        t += dt
        # Noisy measurement of true position at time t.
        z = np.array([[true_p0 + true_v * t + rng.normal(0.0, np.sqrt(0.05))]])
        kf.predict(dt=float(dt))
        kf.update(z)

    # After 200 noisy observations spanning ~50 s, the filter should be very
    # close to the truth.
    assert abs(float(kf.x[1, 0]) - true_v) < 0.1, f"velocity estimate off: {kf.x[1, 0]} vs {true_v}"
    expected_p = true_p0 + true_v * t
    assert abs(float(kf.x[0, 0]) - expected_p) < 1.0, f"position estimate off: {kf.x[0, 0]} vs {expected_p}"


def test_frame_skip_equivalence_under_dt_one() -> None:
    """Calling `predict(1.0)` N times must equal calling `predict()` N times.

    Stronger than test_predict_default_dt_matches_legacy: also exercises the
    cached_dt path so that subsequent `predict(1.0)` calls (post-bootstrap)
    still preserve the reference Q.
    """
    bbox = np.array([10.0, 20.0, 30.0, 40.0])
    a = XYXYStateEstimator(bbox.copy())
    b = XYXYStateEstimator(bbox.copy())

    a.set_kf_covariances(Q=np.eye(8) * 0.01)
    b.set_kf_covariances(Q=np.eye(8) * 0.01)

    for _ in range(10):
        a.predict()
        b.predict(dt=1.0)

    np.testing.assert_allclose(a.kf.x, b.kf.x, atol=1e-12)
    np.testing.assert_allclose(a.kf.P, b.kf.P, atol=1e-12)
