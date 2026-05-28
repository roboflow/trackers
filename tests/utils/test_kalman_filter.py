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
    """A bare KalmanFilter uses the ``F`` / ``Q`` matrices stored on the instance."""
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[0.0], [1.0]])  # position 0, velocity 1
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # CV with dt = 1
    kf.Q = np.array([[0.1, 0.0], [0.0, 0.1]])
    kf.predict()

    np.testing.assert_allclose(kf.x, np.array([[1.0], [1.0]]))


def test_update_with_none_preserves_posterior_as_prior() -> None:
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[1.0], [0.5]])
    kf.P = np.eye(2) * 0.5
    kf.predict()
    x_prior = kf.x.copy()
    P_prior = kf.P.copy()

    kf.update(None)

    np.testing.assert_allclose(kf.x, x_prior)
    np.testing.assert_allclose(kf.P, P_prior)
