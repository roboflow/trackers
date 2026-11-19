# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for the batched Kalman filter / state-estimator updates."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from trackers.utils.kalman_filter import KalmanFilter
from trackers.utils.state_representations import XYXYStateEstimator


def _make_filters(count: int, rng: np.random.Generator) -> list[KalmanFilter]:
    """Return `count` independent 8-state / 4-measurement filters."""
    filters = []
    for _ in range(count):
        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.state = rng.normal(size=(8, 1))
        kf.state_covariance = np.eye(8) + rng.normal(scale=0.01, size=(8, 8))
        # A nonzero observation matrix gives a nonzero Kalman gain, so the batched correction and
        # the Joseph covariance update are exercised instead of leaving the state untouched.
        kf.observation_mtx = np.hstack([np.eye(4), np.zeros((4, 4))])
        filters.append(kf)
    return filters


def test_update_batch_matches_single_updates() -> None:
    """Batched and individual updates end in bit-identical states and covariances."""
    rng = np.random.default_rng(0)
    filters = _make_filters(5, rng)
    measurements = [rng.normal(size=(4, 1)) for _ in filters]
    expected = copy.deepcopy(filters)

    for kf, z in zip(expected, measurements):
        kf.update(z)
    KalmanFilter.update_batch(filters, measurements)

    for batched, single in zip(filters, expected):
        np.testing.assert_array_equal(batched.state, single.state)
        np.testing.assert_array_equal(batched.state_covariance, single.state_covariance)


def test_update_batch_length_mismatch_raises() -> None:
    """A measurement count that does not match the filter count is rejected."""
    rng = np.random.default_rng(1)
    filters = _make_filters(2, rng)

    with pytest.raises(ValueError, match="same length"):
        KalmanFilter.update_batch(filters, [rng.normal(size=(4, 1))])


def test_update_batch_rejects_different_models() -> None:
    """Filters whose measurement noise differs are not silently batched together."""
    rng = np.random.default_rng(2)
    filters = _make_filters(2, rng)
    filters[1].measurement_noise = filters[1].measurement_noise * 2.0
    measurements = [rng.normal(size=(4, 1)) for _ in filters]

    with pytest.raises(ValueError, match="same model"):
        KalmanFilter.update_batch(filters, measurements)


def test_update_batch_can_skip_model_check() -> None:
    """The fast path skips the model comparison but keeps the algebra unchanged."""
    rng = np.random.default_rng(3)
    filters = _make_filters(3, rng)
    measurements = [rng.normal(size=(4, 1)) for _ in filters]
    expected = copy.deepcopy(filters)

    for kf, z in zip(expected, measurements):
        kf.update(z)
    KalmanFilter.update_batch(filters, measurements, check_models=False)

    for batched, single in zip(filters, expected):
        np.testing.assert_array_equal(batched.state, single.state)
        np.testing.assert_array_equal(batched.state_covariance, single.state_covariance)


def test_estimator_update_batch_matches_individual_updates() -> None:
    """Estimator-level batching matches one `update` call per estimator."""
    boxes = [np.array([10.0, 20.0, 40.0, 80.0]) + i for i in range(3)]
    estimators = [XYXYStateEstimator(box) for box in boxes]
    expected = copy.deepcopy(estimators)

    for estimator, box in zip(expected, boxes):
        estimator.update(box + 1.0)
    XYXYStateEstimator.update_batch(estimators, [box + 1.0 for box in boxes])

    for batched, single in zip(estimators, expected):
        np.testing.assert_array_equal(batched.kf.state, single.kf.state)
        np.testing.assert_array_equal(batched.kf.state_covariance, single.kf.state_covariance)


def test_estimator_update_batch_mixed_types_fall_back() -> None:
    """A mixed estimator list is still updated, one estimator at a time."""
    boxes = [np.array([10.0, 20.0, 40.0, 80.0]), np.array([15.0, 25.0, 45.0, 85.0])]

    class OtherEstimator(XYXYStateEstimator):
        pass

    estimators = [XYXYStateEstimator(boxes[0]), OtherEstimator(boxes[1])]
    expected = copy.deepcopy(estimators)

    for estimator, box in zip(expected, boxes):
        estimator.update(box + 2.0)
    XYXYStateEstimator.update_batch(estimators, [box + 2.0 for box in boxes])

    for updated, single in zip(estimators, expected):
        np.testing.assert_array_equal(updated.kf.state, single.kf.state)


def test_estimator_update_batch_falls_back_for_different_noise() -> None:
    """Estimators of one class with different measurement noise are updated individually."""
    boxes = [np.array([10.0, 20.0, 40.0, 80.0]), np.array([12.0, 22.0, 42.0, 82.0])]
    estimators = [XYXYStateEstimator(box) for box in boxes]
    estimators[1].set_kf_covariances(measurement_noise=estimators[1].kf.measurement_noise * 2.0)
    expected = copy.deepcopy(estimators)

    for estimator, box in zip(expected, boxes):
        estimator.update(box + 3.0)
    XYXYStateEstimator.update_batch(estimators, [box + 3.0 for box in boxes])

    for batched, single in zip(estimators, expected):
        np.testing.assert_array_equal(batched.kf.state, single.kf.state)
        np.testing.assert_array_equal(batched.kf.state_covariance, single.kf.state_covariance)


def test_estimator_update_batch_falls_back_when_update_is_overridden() -> None:
    """An estimator subclass that overrides `update` keeps its override."""
    boxes = [np.array([10.0, 20.0, 40.0, 80.0]), np.array([12.0, 22.0, 42.0, 82.0])]
    calls: list[int] = []

    class CountingEstimator(XYXYStateEstimator):
        def update(self, bbox: np.ndarray | None) -> None:
            calls.append(1)
            super().update(bbox)

    estimators = [CountingEstimator(box) for box in boxes]
    XYXYStateEstimator.update_batch(estimators, [box + 1.0 for box in boxes])

    assert len(calls) == len(estimators)
