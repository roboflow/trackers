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


@pytest.fixture
def bbox() -> np.ndarray:
    """A 40x60 bounding box in xyxy format."""
    return np.array([10.0, 20.0, 50.0, 80.0])


@pytest.fixture
def tracklet(bbox: np.ndarray) -> BoTSORTTracklet:
    return BoTSORTTracklet(bbox)


# -------------------------------------------------------------------
# predict()
# -------------------------------------------------------------------


def test_botsort_tracklet_predict_clamps_wh_positive(
    bbox: np.ndarray,
) -> None:
    """Width and height stay positive even after many predictions with no update."""
    tracklet = BoTSORTTracklet(bbox)
    for _ in range(50):
        tracklet.predict()
    state = tracklet.state_estimator.kf.x.reshape(-1)
    assert state[2] > 0, "width must stay positive after many predicts"
    assert state[3] > 0, "height must stay positive after many predicts"


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

    small_Q = np.diag(small.state_estimator.kf.Q)
    large_Q = np.diag(large.state_estimator.kf.Q)

    assert np.all(large_Q > small_Q), (
        "larger box must produce larger process noise diagonal"
    )


def test_botsort_tracklet_larger_box_has_larger_measurement_noise() -> None:
    """A bigger bounding box must produce strictly larger R diagonal values."""
    small = BoTSORTTracklet(np.array([0.0, 0.0, 20.0, 20.0]))
    large = BoTSORTTracklet(np.array([0.0, 0.0, 200.0, 200.0]))

    small_R = np.diag(small.state_estimator.kf.R)
    large_R = np.diag(large.state_estimator.kf.R)

    assert np.all(large_R > small_R), (
        "larger box must produce larger measurement noise diagonal"
    )


# -------------------------------------------------------------------
# apply_cmc()
# -------------------------------------------------------------------


def test_botsort_tracklet_apply_cmc_none_is_noop(
    tracklet: BoTSORTTracklet,
) -> None:
    """apply_cmc(None) must leave state and covariance unchanged."""
    state_before = tracklet.state_estimator.kf.x.copy()
    P_before = tracklet.state_estimator.kf.P.copy()

    tracklet.apply_cmc(None)

    np.testing.assert_array_equal(tracklet.state_estimator.kf.x, state_before)
    np.testing.assert_array_equal(tracklet.state_estimator.kf.P, P_before)


def test_botsort_tracklet_apply_cmc_identity_is_noop(
    tracklet: BoTSORTTracklet,
) -> None:
    """apply_cmc with an identity transform must leave state unchanged."""
    state_before = tracklet.state_estimator.kf.x.copy()
    tracklet.apply_cmc(np.eye(2, 3, dtype=np.float32))
    np.testing.assert_allclose(tracklet.state_estimator.kf.x, state_before, atol=1e-6)


def test_botsort_tracklet_apply_cmc_translates_center(
    tracklet: BoTSORTTracklet,
) -> None:
    """A pure translation H must shift center by (tx, ty)."""
    x_before = tracklet.state_estimator.kf.x.reshape(-1).copy()
    cx, cy = x_before[0], x_before[1]

    tx, ty = 10.0, -5.0
    H = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    tracklet.apply_cmc(H)

    x_after = tracklet.state_estimator.kf.x.reshape(-1)
    np.testing.assert_allclose(x_after[0], cx + tx, atol=1e-6)
    np.testing.assert_allclose(x_after[1], cy + ty, atol=1e-6)


def test_botsort_tracklet_apply_cmc_does_not_affect_wh(
    tracklet: BoTSORTTracklet,
) -> None:
    """CMC must not change the width and height components of the state."""
    x_before = tracklet.state_estimator.kf.x.reshape(-1).copy()
    w_before, h_before = x_before[2], x_before[3]

    H = np.array([[1.0, 0.0, 15.0], [0.0, 1.0, 7.0]], dtype=np.float32)
    tracklet.apply_cmc(H)

    x_after = tracklet.state_estimator.kf.x.reshape(-1)
    np.testing.assert_allclose(x_after[2], w_before, atol=1e-6)
    np.testing.assert_allclose(x_after[3], h_before, atol=1e-6)


# -------------------------------------------------------------------
# apply_cmc_batch()
# -------------------------------------------------------------------


def test_botsort_tracklet_apply_cmc_batch_matches_single(
    bbox: np.ndarray,
) -> None:
    """Batch CMC must produce the same result as applying apply_cmc individually."""
    H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float32)

    single = BoTSORTTracklet(bbox)
    batched = BoTSORTTracklet(bbox)

    single.apply_cmc(H)
    BoTSORTTracklet.apply_cmc_batch([batched], H)

    np.testing.assert_allclose(
        single.state_estimator.kf.x,
        batched.state_estimator.kf.x,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        single.state_estimator.kf.P,
        batched.state_estimator.kf.P,
        atol=1e-6,
    )


def test_botsort_tracklet_apply_cmc_batch_multiple_tracklets(
    bbox: np.ndarray,
) -> None:
    """Batch CMC applies the same transform to every tracklet in the list."""
    H = np.array([[1.0, 0.0, 8.0], [0.0, 1.0, -2.0]], dtype=np.float32)

    singles = [BoTSORTTracklet(bbox) for _ in range(3)]
    batched = [BoTSORTTracklet(bbox) for _ in range(3)]

    for t in singles:
        t.apply_cmc(H)
    BoTSORTTracklet.apply_cmc_batch(batched, H)

    for s, b in zip(singles, batched):
        np.testing.assert_allclose(
            s.state_estimator.kf.x, b.state_estimator.kf.x, atol=1e-6
        )


def test_botsort_tracklet_apply_cmc_batch_none_is_noop(
    bbox: np.ndarray,
) -> None:
    """apply_cmc_batch with H=None must not change any tracklet state."""
    tracklet = BoTSORTTracklet(bbox)
    state_before = tracklet.state_estimator.kf.x.copy()

    BoTSORTTracklet.apply_cmc_batch([tracklet], None)

    np.testing.assert_array_equal(tracklet.state_estimator.kf.x, state_before)


def test_botsort_tracklet_apply_cmc_batch_empty_list_is_noop() -> None:
    """apply_cmc_batch with an empty list must not raise."""
    H = np.eye(2, 3, dtype=np.float32)
    BoTSORTTracklet.apply_cmc_batch([], H)  # must not raise
