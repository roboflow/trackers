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
