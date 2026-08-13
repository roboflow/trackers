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
from trackers.utils.predict_timing import PredictTiming
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


def test_mcbyte_tracklet_larger_box_has_larger_process_noise() -> None:
    """A bigger bounding box must produce strictly larger Q diagonal values."""
    small = McByteTracklet(np.array([0.0, 0.0, 20.0, 20.0]))
    large = McByteTracklet(np.array([0.0, 0.0, 200.0, 200.0]))

    small_Q = np.diag(small.state_estimator.kf.process_noise)
    large_Q = np.diag(large.state_estimator.kf.process_noise)

    assert np.all(large_Q > small_Q), "larger box must produce larger process noise diagonal"


def test_mcbyte_tracklet_larger_box_has_larger_measurement_noise() -> None:
    """A bigger bounding box must produce strictly larger R diagonal values."""
    small = McByteTracklet(np.array([0.0, 0.0, 20.0, 20.0]))
    large = McByteTracklet(np.array([0.0, 0.0, 200.0, 200.0]))

    small_R = np.diag(small.state_estimator.kf.measurement_noise)
    large_R = np.diag(large.state_estimator.kf.measurement_noise)

    assert np.all(large_R > small_R), "larger box must produce larger measurement noise diagonal"


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

    # Direct Q equality depends on the default frame_step=1.0 staying in the near-nominal band.
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


def test_mcbyte_tracklet_gap_predict_uses_refreshed_dwna_noise(
    tracklet: McByteTracklet,
) -> None:
    """A second gap predict must DWNA-scale Q from the post-update box size."""
    gap_timing = PredictTiming(frame_step=2.0, elapsed_seconds=None)
    tracklet.predict(gap_timing)
    tracklet.update(np.array([5.0, 5.0, 205.0, 205.0]))

    bbox = tracklet.get_state_bbox()
    w = max(float(bbox[2] - bbox[0]), 1e-3)
    h = max(float(bbox[3] - bbox[1]), 1e-3)
    refreshed_baseline_Q = tracklet._build_process_noise(w, h)

    tracklet.predict(gap_timing)

    scalable_noise = tracklet.state_estimator.motion.process_noise
    np.testing.assert_array_equal(scalable_noise.baseline_Q, refreshed_baseline_Q)
    expected_Q = scalable_noise.build_Q(gap_timing.frame_step, gap_timing.frame_rate)
    np.testing.assert_allclose(tracklet.state_estimator.kf.process_noise, expected_Q)
    assert not np.array_equal(tracklet.state_estimator.kf.process_noise, refreshed_baseline_Q)


def test_mcbyte_tracklet_interleaved_predict_update_predict_refreshes_split_noise(
    tracklet: McByteTracklet,
) -> None:
    """Interleaved calls must refresh only the noise matrix consumed by each operation."""
    bbox = tracklet.get_state_bbox()
    w = max(float(bbox[2] - bbox[0]), 1e-3)
    h = max(float(bbox[3] - bbox[1]), 1e-3)
    expected_first_Q = tracklet._build_process_noise(w, h)
    initial_R = tracklet.state_estimator.kf.measurement_noise.copy()

    tracklet.predict()

    np.testing.assert_array_equal(tracklet.state_estimator.kf.process_noise, expected_first_Q)
    np.testing.assert_array_equal(tracklet.state_estimator.kf.measurement_noise, initial_R)

    bbox = tracklet.get_state_bbox()
    w = max(float(bbox[2] - bbox[0]), 1e-3)
    h = max(float(bbox[3] - bbox[1]), 1e-3)
    expected_R = tracklet._build_measurement_noise(w, h)
    process_noise_after_first_predict = tracklet.state_estimator.kf.process_noise.copy()

    tracklet.update(np.array([5.0, 5.0, 205.0, 205.0]))

    np.testing.assert_array_equal(tracklet.state_estimator.kf.measurement_noise, expected_R)
    np.testing.assert_array_equal(tracklet.state_estimator.kf.process_noise, process_noise_after_first_predict)

    bbox = tracklet.get_state_bbox()
    w = max(float(bbox[2] - bbox[0]), 1e-3)
    h = max(float(bbox[3] - bbox[1]), 1e-3)
    expected_second_Q = tracklet._build_process_noise(w, h)
    measurement_noise_after_update = tracklet.state_estimator.kf.measurement_noise.copy()

    tracklet.predict()

    np.testing.assert_array_equal(tracklet.state_estimator.kf.process_noise, expected_second_Q)
    np.testing.assert_array_equal(tracklet.state_estimator.kf.measurement_noise, measurement_noise_after_update)
    assert not np.array_equal(expected_first_Q, expected_second_Q)
