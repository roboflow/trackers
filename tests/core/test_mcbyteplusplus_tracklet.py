# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Independent McByte++ motion, noise, timing and camera-compensation contracts."""

import numpy as np
import pytest

from trackers.core.mcbyteplusplus.tracklet import McBytePlusPlusTracklet
from trackers.utils.predict_timing import PredictTiming
from trackers.utils.state_representations import XCYCSRStateEstimator, XCYCWHStateEstimator, XYXYStateEstimator


def test_tracklet_predict_and_update_clocks() -> None:
    tracklet = McBytePlusPlusTracklet(np.array([1.0, 1.0, 5.0, 9.0]))
    tracklet.predict(PredictTiming(frame_step=2.0, elapsed_seconds=0.08, frame_rate=25.0))
    assert tracklet.time_since_update == 1 and tracklet.time_since_update_seconds == 0.08
    tracklet.update(np.array([2.0, 1.0, 6.0, 9.0]))
    assert tracklet.time_since_update == 0 and tracklet.time_since_update_seconds == 0
    assert tracklet.number_of_successful_updates == 2


@pytest.fixture
def bbox() -> np.ndarray:
    """A 40x60 bounding box in xyxy format."""
    return np.array([10.0, 20.0, 50.0, 80.0])


@pytest.fixture(params=[XCYCWHStateEstimator, XYXYStateEstimator, XCYCSRStateEstimator])
def tracklet(bbox: np.ndarray, request: pytest.FixtureRequest) -> McBytePlusPlusTracklet:
    estimator_class = request.param
    return McBytePlusPlusTracklet(bbox, state_estimator_class=estimator_class)


@pytest.mark.parametrize("frame_step", [0.5, 1.0, 2.0])
def test_stationary_prediction_and_camera_translation_preserve_identity(tracklet, bbox, frame_step) -> None:
    """Zero object velocity keeps the box fixed; camera translation shifts both corners equally."""
    assert (tracklet.tracker_id, tracklet.public_id) == (-1, -1)
    tracklet.tracker_id, tracklet.public_id = 7, 900
    prediction = tracklet.predict(
        PredictTiming(frame_step=frame_step, elapsed_seconds=frame_step / 25, frame_rate=25.0)
    )
    np.testing.assert_allclose(prediction, bbox)
    covariance = tracklet.state_estimator.kf.state_covariance.copy()
    tracklet.apply_cmc(np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]]))
    translated = bbox + np.array([5.0, -3.0, 5.0, -3.0])
    np.testing.assert_allclose(tracklet.get_state_bbox(), translated)
    np.testing.assert_allclose(tracklet.state_estimator.kf.state_covariance, covariance)
    tracklet.apply_cmc(None)
    np.testing.assert_allclose(tracklet.get_state_bbox(), translated)
    np.testing.assert_allclose(tracklet.state_estimator.kf.state_covariance, covariance)
    assert tracklet.time_since_update == 1
    tracklet.update(translated)
    np.testing.assert_allclose(tracklet.get_state_bbox(), translated)
    assert np.isfinite(tracklet.state_estimator.kf.state_covariance).all()
    assert (tracklet.tracker_id, tracklet.public_id) == (7, 900)


def test_predict_forwards_frame_rate_for_mid_band_gap() -> None:
    """At 60 FPS a 1.15-frame step remains within the nominal-noise tolerance."""
    box = np.array([100.0, 100.0, 150.0, 200.0])
    nominal = McBytePlusPlusTracklet(box)
    nominal.predict(PredictTiming(frame_step=1.0, elapsed_seconds=1.0 / 60.0, frame_rate=60.0))
    gapped = McBytePlusPlusTracklet(box)
    gapped.predict(PredictTiming(frame_step=1.15, elapsed_seconds=1.15 / 60.0, frame_rate=60.0))
    np.testing.assert_array_equal(gapped.state_estimator.kf.process_noise, nominal.state_estimator.kf.process_noise)


class TestMcBytePlusPlusNoise:
    """Scale-aware noise refresh behavior for McBytePlusPlusTracklet."""

    def test_larger_box_has_larger_process_noise(self) -> None:
        """A bigger bounding box must produce strictly larger Q diagonal values."""
        small = McBytePlusPlusTracklet(np.array([0.0, 0.0, 20.0, 20.0]))
        large = McBytePlusPlusTracklet(np.array([0.0, 0.0, 200.0, 200.0]))

        small_Q = np.diag(small.state_estimator.kf.process_noise)
        large_Q = np.diag(large.state_estimator.kf.process_noise)

        assert np.all(large_Q > small_Q), "larger box must produce larger process noise diagonal"

    def test_larger_box_has_larger_measurement_noise(self) -> None:
        """A bigger bounding box must produce strictly larger R diagonal values."""
        small = McBytePlusPlusTracklet(np.array([0.0, 0.0, 20.0, 20.0]))
        large = McBytePlusPlusTracklet(np.array([0.0, 0.0, 200.0, 200.0]))

        small_R = np.diag(small.state_estimator.kf.measurement_noise)
        large_R = np.diag(large.state_estimator.kf.measurement_noise)

        assert np.all(large_R > small_R), "larger box must produce larger measurement noise diagonal"

    def test_gap_predict_uses_refreshed_dwna_noise(
        self,
        tracklet: McBytePlusPlusTracklet,
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

    def test_interleaved_predict_update_predict_refreshes_split_noise(
        self,
        tracklet: McBytePlusPlusTracklet,
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
