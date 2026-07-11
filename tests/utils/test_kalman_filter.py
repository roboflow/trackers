# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for ``KalmanFilter`` predict/update algebra."""

from __future__ import annotations

import numpy as np

from trackers.utils.kalman_filter import KalmanFilter


def test_predict_uses_stored_motion_matrices() -> None:
    """A bare KalmanFilter uses transition_mtx / process_noise stored on the instance."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.state = np.array([[0.0], [1.0]])  # position 0, velocity 1
    kf.transition_mtx = np.array([[1.0, 1.0], [0.0, 1.0]])  # CV with dt = 1
    kf.process_noise = np.array([[0.1, 0.0], [0.0, 0.1]])
    kf.predict()

    np.testing.assert_allclose(kf.state, np.array([[1.0], [1.0]]))


def test_update_with_none_preserves_posterior_as_prior() -> None:
    """Update with None leaves state and covariance equal to the prior."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.state = np.array([[1.0], [0.5]])
    kf.state_covariance = np.eye(2) * 0.5
    kf.predict()
    state_prior = kf.state.copy()
    cov_prior = kf.state_covariance.copy()

    kf.update(None)

    np.testing.assert_allclose(kf.state, state_prior)
    np.testing.assert_allclose(kf.state_covariance, cov_prior)
