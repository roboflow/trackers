# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""BoT-SORT-specific tracklet tests.

Generic predict/update contracts (time_since_update, age) are covered for all
tracklet classes in test_tracklets.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from trackers.core.botsort.tracklet import BoTSORTTracklet
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCSRStateEstimator,
    XCYCWHStateEstimator,
    XYXYStateEstimator,
)


@pytest.fixture
def bbox() -> np.ndarray:
    """A 40x60 bounding box in xyxy format."""
    return np.array([10.0, 20.0, 50.0, 80.0])


@pytest.fixture(params=[XCYCWHStateEstimator, XYXYStateEstimator, XCYCSRStateEstimator])
def tracklet(bbox: np.ndarray, request: pytest.FixtureRequest) -> BoTSORTTracklet:
    estimator_class = request.param
    return BoTSORTTracklet(bbox, state_estimator_class=estimator_class)


# -------------------------------------------------------------------
# predict()
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator_class",
    [XCYCWHStateEstimator, XYXYStateEstimator, XCYCSRStateEstimator],
)
def test_botsort_tracklet_predict_keeps_valid_bbox(
    bbox: np.ndarray,
    estimator_class: type[BaseStateEstimator],
) -> None:
    """BBox width/height stay positive even after many predictions."""
    tracklet = BoTSORTTracklet(bbox, state_estimator_class=estimator_class)
    for _ in range(50):
        tracklet.predict()
    state_bbox = tracklet.get_state_bbox()
    assert state_bbox[2] > state_bbox[0], "width must stay positive after predicts"
    assert state_bbox[3] > state_bbox[1], "height must stay positive after predicts"


# -------------------------------------------------------------------
# update()
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Scale-aware noise
# -------------------------------------------------------------------


def test_botsort_tracklet_larger_box_has_larger_process_noise() -> None:
    """A bigger bounding box must produce strictly larger Q diagonal values."""
    small = BoTSORTTracklet(np.array([0.0, 0.0, 20.0, 20.0]))
    large = BoTSORTTracklet(np.array([0.0, 0.0, 200.0, 200.0]))

    small_Q = np.diag(small.state_estimator.kf.process_noise)
    large_Q = np.diag(large.state_estimator.kf.process_noise)

    assert np.all(large_Q > small_Q), "larger box must produce larger process noise diagonal"


def test_botsort_tracklet_larger_box_has_larger_measurement_noise() -> None:
    """A bigger bounding box must produce strictly larger R diagonal values."""
    small = BoTSORTTracklet(np.array([0.0, 0.0, 20.0, 20.0]))
    large = BoTSORTTracklet(np.array([0.0, 0.0, 200.0, 200.0]))

    small_R = np.diag(small.state_estimator.kf.measurement_noise)
    large_R = np.diag(large.state_estimator.kf.measurement_noise)

    assert np.all(large_R > small_R), "larger box must produce larger measurement noise diagonal"


# -------------------------------------------------------------------
# apply_cmc()
# -------------------------------------------------------------------


def test_botsort_tracklet_apply_cmc_none_is_noop(
    tracklet: BoTSORTTracklet,
) -> None:
    """apply_cmc(None) must leave state and covariance unchanged."""
    state_before = tracklet.state_estimator.kf.state.copy()
    P_before = tracklet.state_estimator.kf.state_covariance.copy()

    tracklet.apply_cmc(None)

    np.testing.assert_array_equal(tracklet.state_estimator.kf.state, state_before)
    np.testing.assert_array_equal(tracklet.state_estimator.kf.state_covariance, P_before)


def test_botsort_tracklet_apply_cmc_identity_is_noop(
    tracklet: BoTSORTTracklet,
) -> None:
    """apply_cmc with an identity transform must leave state unchanged."""
    state_before = tracklet.state_estimator.kf.state.copy()
    tracklet.apply_cmc(np.eye(2, 3, dtype=np.float32))
    np.testing.assert_allclose(tracklet.state_estimator.kf.state, state_before, atol=1e-6)


def test_botsort_tracklet_apply_cmc_translates_center(
    tracklet: BoTSORTTracklet,
) -> None:
    """A pure translation H must shift center by (tx, ty)."""
    x_before = tracklet.state_estimator.kf.state.reshape(-1).copy()
    cx, cy = x_before[0], x_before[1]

    tx, ty = 10.0, -5.0
    H = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    tracklet.apply_cmc(H)

    x_after = tracklet.state_estimator.kf.state.reshape(-1)
    np.testing.assert_allclose(x_after[0], cx + tx, atol=1e-6)
    np.testing.assert_allclose(x_after[1], cy + ty, atol=1e-6)


def test_botsort_tracklet_apply_cmc_does_not_affect_wh(
    tracklet: BoTSORTTracklet,
) -> None:
    """CMC must preserve bbox width and height in xyxy space."""
    bbox_before = tracklet.get_state_bbox().copy()
    w_before = bbox_before[2] - bbox_before[0]
    h_before = bbox_before[3] - bbox_before[1]

    H = np.array([[1.0, 0.0, 15.0], [0.0, 1.0, 7.0]], dtype=np.float32)
    tracklet.apply_cmc(H)

    bbox_after = tracklet.get_state_bbox()
    w_after = bbox_after[2] - bbox_after[0]
    h_after = bbox_after[3] - bbox_after[1]
    np.testing.assert_allclose(w_after, w_before, atol=1e-6)
    np.testing.assert_allclose(h_after, h_before, atol=1e-6)
