# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Smoke tests that each bbox estimator wires motion sync correctly.

Timestamp integration is covered in ``tests/core/test_timestamp_plumbing.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

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

BBOX = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_predict_default_matches_unit_frame_step(
    estimator_cls: type[BaseStateEstimator],
) -> None:
    default = estimator_cls(BBOX.copy())
    explicit = estimator_cls(BBOX.copy())

    for _ in range(5):
        default.predict()
        explicit.predict(1.0)

    np.testing.assert_allclose(default.kf.state, explicit.kf.state, atol=1e-12)
    np.testing.assert_allclose(default.kf.state_covariance, explicit.kf.state_covariance, atol=1e-12)


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_set_state_resets_motion_cache(
    estimator_cls: type[BaseStateEstimator],
) -> None:
    est = estimator_cls(BBOX.copy())
    est.predict(0.5)
    assert est.motion.cached_step == pytest.approx(0.5)

    state = est.get_state()
    est.predict(2.0)
    est.set_state(state)

    assert est.motion.cached_step is None


@pytest.mark.parametrize(
    ("scale_velocity", "expected_scale_velocity"),
    [
        pytest.param(-100.0, 0.0, id="projection-non-positive-frozen"),
        pytest.param(-10.0, -10.0, id="projection-positive-preserved"),
    ],
)
def test_xcycsr_clamp_velocity_guards_projected_scale_over_frame_step(
    scale_velocity: float, expected_scale_velocity: float
) -> None:
    """``clamp_velocity`` guards the *projected* scale ``s + frame_step * vs``.

    A negative scale velocity that passes the one-frame check (``s + vs > 0``)
    used to extrapolate scale below zero over a gap, and ``xcycsr_to_xyxy``
    then took ``sqrt`` of a negative scale -> NaN boxes. The guard must fire
    on a non-positive projection and stay out of the way otherwise.
    """
    est = XCYCSRStateEstimator(BBOX.copy())
    est.kf.state[2] = 1000.0  # scale (area); s + vs > 0, so the one-frame check passes
    est.kf.state[6] = scale_velocity  # s + 20 * vs: -1000 (fires) vs 800 (preserved)

    est.clamp_velocity(20.0)

    assert est.kf.state[6] == pytest.approx(expected_scale_velocity)


def test_xcycsr_clamp_velocity_exact_zero_projection_bound() -> None:
    est = XCYCSRStateEstimator(BBOX.copy())
    est.kf.state[2] = 1000.0
    est.kf.state[6] = -100.0  # 1000.0 + 10.0 * -100.0 == 0.0

    est.clamp_velocity(10.0)

    assert est.kf.state[6] == pytest.approx(0.0)
    assert est.kf.state[2] + 10.0 * est.kf.state[6] > 0.0


def test_xcycsr_clamp_velocity_large_gap_growing_box_keeps_state_unchanged() -> None:
    est = XCYCSRStateEstimator(BBOX.copy())
    before = est.kf.state.copy()

    est.kf.state[0] = 12.0
    est.kf.state[1] = 14.0
    est.kf.state[2] = 1000.0
    est.kf.state[3] = 0.75
    est.kf.state[4] = 50.0
    est.kf.state[5] = -30.0
    est.kf.state[6] = 200.0
    before = est.kf.state.copy()

    est.clamp_velocity(10.0)

    np.testing.assert_array_equal(est.kf.state, before)


def test_xcycsr_clamp_velocity_large_gap_shrink_mutates_only_scale_velocity() -> None:
    est = XCYCSRStateEstimator(BBOX.copy())
    est.kf.state[0] = 24.0
    est.kf.state[1] = 27.0
    est.kf.state[2] = 1000.0
    est.kf.state[3] = 1.25
    est.kf.state[4] = 10.0
    est.kf.state[5] = 20.0
    est.kf.state[6] = -100.0
    before = est.kf.state.copy()

    est.clamp_velocity(20.0)

    assert est.kf.state[6] == pytest.approx(0.0)
    np.testing.assert_array_equal(est.kf.state[:6], before[:6])


def test_xcycsr_clamp_velocity_fractional_frame_step() -> None:
    est = XCYCSRStateEstimator(BBOX.copy())
    est.kf.state[2] = 1000.0
    est.kf.state[6] = -1500.0  # 1000.0 + 0.5 * -1500.0 == 250.0 (still positive)

    est.clamp_velocity(0.5)

    assert est.kf.state[6] == pytest.approx(-1500.0)
    assert est.kf.state[2] + 0.5 * est.kf.state[6] == pytest.approx(250.0)


def test_xcycsr_clamp_velocity_single_frame_step_regression() -> None:
    legacy_like = XCYCSRStateEstimator(BBOX.copy())
    explicit_step = XCYCSRStateEstimator(BBOX.copy())

    for est in (legacy_like, explicit_step):
        est.kf.state[2] = 1000.0
        est.kf.state[6] = -1200.0

    legacy_like.clamp_velocity()
    explicit_step.clamp_velocity(1.0)

    assert legacy_like.kf.state[6] == pytest.approx(0.0)
    assert legacy_like.kf.state[6] == explicit_step.kf.state[6]


def test_xcycsr_clamp_velocity_unfixed_skip_for_zero_negative_nan_inf_steps() -> None:
    # This intentionally documents current estimator behavior when callers pass
    # edge-case frame steps directly to clamp_velocity: NaN comparisons are
    # always False in Python, so the clamp guard is skipped and velocity is not
    # reset. This remains safe for tracker runtime because base.py filters these
    # values before invoking predict()/clamp_velocity.
    est = XCYCSRStateEstimator(BBOX.copy())
    est.kf.state[2] = 1000.0
    est.kf.state[6] = -1200.0
    est.clamp_velocity(1.0)
    assert est.kf.state[6] == pytest.approx(0.0)

    for frame_step, scale_velocity in [
        (0.0, 1200.0),
        (-2.0, -1200.0),
        (float("nan"), 1200.0),
        (float("inf"), 1200.0),
    ]:
        est = XCYCSRStateEstimator(BBOX.copy())
        est.kf.state[2] = 1000.0
        est.kf.state[6] = scale_velocity
        est.clamp_velocity(frame_step)

        assert est.kf.state[6] == pytest.approx(scale_velocity)
