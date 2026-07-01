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
