# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""McByte-specific tracklet tests.

McByteTracklet duplicates BoTSORTTracklet's scale-aware noise machinery; see test_botsort_tracklet.py for the equivalent
coverage on that class.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from trackers.core.mcbyte.tracklet import McByteTracklet
from trackers.utils.state_representations import (
    XCYCSRStateEstimator,
    XCYCWHStateEstimator,
    XYXYStateEstimator,
)


@pytest.fixture
def bbox() -> np.ndarray:
    """A 40x60 bounding box in xyxy format."""
    return np.array([10.0, 20.0, 50.0, 80.0])


@pytest.fixture(params=[XCYCWHStateEstimator, XYXYStateEstimator, XCYCSRStateEstimator])
def tracklet(bbox: np.ndarray, request: pytest.FixtureRequest) -> McByteTracklet:
    estimator_class = request.param
    return McByteTracklet(bbox, state_estimator_class=estimator_class)


# -------------------------------------------------------------------
# Scale-aware noise
# -------------------------------------------------------------------


def test_mcbyte_tracklet_predict_only_refreshes_process_noise() -> None:
    """Predict() must rebuild Q but leave R alone: KalmanFilter.predict never reads measurement_noise, so rebuilding it
    there is wasted work that update() will overwrite anyway."""
    tracklet = McByteTracklet(np.array([0.0, 0.0, 20.0, 20.0]))

    with patch.object(
        tracklet.state_estimator, "set_kf_covariances", wraps=tracklet.state_estimator.set_kf_covariances
    ) as mock_set:
        tracklet.predict()

    mock_set.assert_called_once()
    kwargs = mock_set.call_args.kwargs
    assert kwargs.get("process_noise") is not None, "predict() must refresh Q"
    assert kwargs.get("measurement_noise") is None, "predict() must not rebuild R"


def test_mcbyte_tracklet_update_only_refreshes_measurement_noise() -> None:
    """Update() must rebuild R but leave Q alone: KalmanFilter.update never reads process_noise, so rebuilding it there
    is wasted work that predict() will overwrite anyway."""
    tracklet = McByteTracklet(np.array([0.0, 0.0, 20.0, 20.0]))

    with patch.object(
        tracklet.state_estimator, "set_kf_covariances", wraps=tracklet.state_estimator.set_kf_covariances
    ) as mock_set:
        tracklet.update(np.array([5.0, 5.0, 205.0, 205.0]))

    mock_set.assert_called_once()
    kwargs = mock_set.call_args.kwargs
    assert kwargs.get("measurement_noise") is not None, "update() must refresh R"
    assert kwargs.get("process_noise") is None, "update() must not rebuild Q"


def test_mcbyte_tracklet_predict_process_noise_matches_pre_predict_box(
    tracklet: McByteTracklet,
) -> None:
    """Q after predict() must equal _build_process_noise() for the box size the tracklet had *before* that predict()
    call — the same size _refresh_process_noise_from_state() reads internally — across all three state representations,
    not just the kwarg it was called with."""
    bbox = tracklet.get_state_bbox()
    w = max(float(bbox[2] - bbox[0]), 1e-3)
    h = max(float(bbox[3] - bbox[1]), 1e-3)
    expected_Q = tracklet._build_process_noise(w, h)

    tracklet.predict()

    np.testing.assert_array_equal(tracklet.state_estimator.kf.process_noise, expected_Q)


def test_mcbyte_tracklet_update_measurement_noise_matches_pre_update_box(
    tracklet: McByteTracklet,
) -> None:
    """R after update() must equal _build_measurement_noise() for the box size the tracklet had *before* that update()
    call (the predicted box, not the new observation) — across all three state representations, not just the kwarg it
    was called with."""
    bbox = tracklet.get_state_bbox()
    w = max(float(bbox[2] - bbox[0]), 1e-3)
    h = max(float(bbox[3] - bbox[1]), 1e-3)
    expected_R = tracklet._build_measurement_noise(w, h)

    tracklet.update(np.array([5.0, 5.0, 205.0, 205.0]))

    np.testing.assert_array_equal(tracklet.state_estimator.kf.measurement_noise, expected_R)
