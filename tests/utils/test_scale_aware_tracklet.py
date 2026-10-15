# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Direct tests for the ``ScaleAwareNoiseTracklet`` mixin.

``BoTSORTTracklet`` and ``McByteTracklet`` already exercise this mixin through their own ``predict``/``update`` wiring
in ``tests/core/test_botsort_tracklet.py``. These tests instead target the mixin's own methods — the noise formulas and
the clamp dispatch — through a minimal concrete subclass, independent of any tracker-specific behavior (CMC, successful-
update counters, etc.).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from trackers.utils.predict_timing import FIXED_RATE_TIMING, PredictTiming
from trackers.utils.scale_aware_tracklet import ScaleAwareNoiseTracklet
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCSRStateEstimator,
    XCYCWHStateEstimator,
    XYXYStateEstimator,
)


class _MinimalScaleAwareTracklet(ScaleAwareNoiseTracklet):
    """Smallest concrete subclass that wires the mixin's noise machinery to `BaseTracklet`.

    Carries no tracker-specific behavior (CMC, update counters) so tests target only what `ScaleAwareNoiseTracklet`
    itself contributes.
    """

    def __init__(
        self,
        bbox: np.ndarray,
        state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator,
    ) -> None:
        super().__init__(bbox, state_estimator_class)
        self._configure_initial_noise(bbox)

    def update(self, bbox: np.ndarray) -> None:
        self._refresh_measurement_noise_from_state()
        self.state_estimator.update(bbox)
        self._clamp_state_bbox()

    def predict(self, timing: PredictTiming = FIXED_RATE_TIMING) -> np.ndarray:
        self._refresh_process_noise_from_state()
        self.state_estimator.predict(timing.frame_step, timing.frame_rate)
        self._clamp_state_bbox()
        self._advance_miss_clocks(timing)
        return self.state_estimator.state_to_bbox()

    def get_state_bbox(self) -> np.ndarray:
        return self.state_estimator.state_to_bbox()


@pytest.fixture
def bbox() -> np.ndarray:
    """A 40x60 bounding box in xyxy format."""
    return np.array([10.0, 20.0, 50.0, 80.0])


class TestScaleAwareNoiseTracklet:
    """Noise-formula and clamp-dispatch contract owned by `ScaleAwareNoiseTracklet` itself."""

    # -------------------------------------------------------------------
    # _build_process_noise() / _build_measurement_noise()
    # -------------------------------------------------------------------

    @pytest.mark.parametrize(
        "estimator_class",
        [XCYCWHStateEstimator, XYXYStateEstimator],
    )
    def test_build_process_noise_non_xcycsr_duplicates_wh_terms(
        self,
        bbox: np.ndarray,
        estimator_class: type[BaseStateEstimator],
    ) -> None:
        """For 8-dim state representations, Q's position/velocity blocks each repeat (w, h) twice."""
        tracklet = _MinimalScaleAwareTracklet(bbox, state_estimator_class=estimator_class)
        w, h = 40, 60

        Q = tracklet._build_process_noise(w, h)

        expected_diag = [
            (0.05 * w) ** 2,
            (0.05 * h) ** 2,
            (0.05 * w) ** 2,
            (0.05 * h) ** 2,
            (0.00625 * w) ** 2,
            (0.00625 * h) ** 2,
            (0.00625 * w) ** 2,
            (0.00625 * h) ** 2,
        ]
        assert Q.shape == (8, 8)
        np.testing.assert_allclose(np.diag(Q), expected_diag)
        np.testing.assert_array_equal(Q - np.diag(np.diag(Q)), np.zeros((8, 8)))

    def test_build_process_noise_xcycsr_uses_sqrt_area_scale(self, bbox: np.ndarray) -> None:
        """For XCYCSR, the scale term uses `sqrt(w*h)` and aspect ratio carries a fixed unit sigma."""
        tracklet = _MinimalScaleAwareTracklet(bbox, state_estimator_class=XCYCSRStateEstimator)
        w, h = 40, 60

        Q = tracklet._build_process_noise(w, h)

        expected_diag = [
            (0.05 * w) ** 2,
            (0.05 * h) ** 2,
            0.05**2 * w * h,  # (sigma_p * sqrt(w*h)) ** 2
            0.05**2,  # (sigma_p * 1.0) ** 2 — aspect ratio
            (0.00625 * w) ** 2,
            (0.00625 * h) ** 2,
            0.00625**2 * w * h,  # (sigma_v * sqrt(w*h)) ** 2
        ]
        assert Q.shape == (7, 7)
        np.testing.assert_allclose(np.diag(Q), expected_diag)

    @pytest.mark.parametrize(
        "estimator_class",
        [XCYCWHStateEstimator, XYXYStateEstimator],
    )
    def test_build_measurement_noise_non_xcycsr_duplicates_wh_terms(
        self,
        bbox: np.ndarray,
        estimator_class: type[BaseStateEstimator],
    ) -> None:
        """For 8-dim state representations, R repeats (w, h) twice across its 4 measurement dims."""
        tracklet = _MinimalScaleAwareTracklet(bbox, state_estimator_class=estimator_class)
        w, h = 40, 60

        R = tracklet._build_measurement_noise(w, h)

        expected_diag = [(0.05 * w) ** 2, (0.05 * h) ** 2, (0.05 * w) ** 2, (0.05 * h) ** 2]
        assert R.shape == (4, 4)
        np.testing.assert_allclose(np.diag(R), expected_diag)

    def test_build_measurement_noise_xcycsr_uses_sqrt_area_scale(self, bbox: np.ndarray) -> None:
        """For XCYCSR, R's scale term uses `sqrt(w*h)` and aspect ratio carries a fixed unit sigma."""
        tracklet = _MinimalScaleAwareTracklet(bbox, state_estimator_class=XCYCSRStateEstimator)
        w, h = 40, 60

        R = tracklet._build_measurement_noise(w, h)

        expected_diag = [
            (0.05 * w) ** 2,
            (0.05 * h) ** 2,
            0.05**2 * w * h,  # (sigma_m * sqrt(w*h)) ** 2
            0.05**2,  # (sigma_m * 1.0) ** 2 — aspect ratio
        ]
        assert R.shape == (4, 4)
        np.testing.assert_allclose(np.diag(R), expected_diag)

    # -------------------------------------------------------------------
    # _configure_initial_noise() / _set_scale_aware_noise()
    # -------------------------------------------------------------------

    @pytest.mark.parametrize(
        "estimator_class",
        [XCYCWHStateEstimator, XYXYStateEstimator, XCYCSRStateEstimator],
    )
    def test_configure_initial_noise_sets_Q_and_R_from_first_bbox(
        self,
        bbox: np.ndarray,
        estimator_class: type[BaseStateEstimator],
    ) -> None:
        """Construction must leave Q, R matching `_build_process_noise`/`_build_measurement_noise` for the first bbox.

        Bbox fixture is 40x60 in xyxy — its xywh width/height feed the noise formulas directly.
        """
        tracklet = _MinimalScaleAwareTracklet(bbox, state_estimator_class=estimator_class)
        expected_Q = tracklet._build_process_noise(40, 60)
        expected_R = tracklet._build_measurement_noise(40, 60)

        np.testing.assert_array_equal(tracklet.state_estimator.kf.process_noise, expected_Q)
        np.testing.assert_array_equal(tracklet.state_estimator.kf.measurement_noise, expected_R)

    def test_configure_initial_noise_state_covariance_scales_position_by_2x_velocity_by_10x(
        self,
        bbox: np.ndarray,
    ) -> None:
        """P's position block uses 2x sigma_p, its velocity block 10x sigma_v — for the non-XCYCSR (w, h) layout."""
        tracklet = _MinimalScaleAwareTracklet(bbox, state_estimator_class=XCYCWHStateEstimator)
        w, h = 40, 60

        expected_diag = [
            (2 * 0.05 * w) ** 2,
            (2 * 0.05 * h) ** 2,
            (2 * 0.05 * w) ** 2,
            (2 * 0.05 * h) ** 2,
            (10 * 0.00625 * w) ** 2,
            (10 * 0.00625 * h) ** 2,
            (10 * 0.00625 * w) ** 2,
            (10 * 0.00625 * h) ** 2,
        ]
        np.testing.assert_allclose(np.diag(tracklet.state_estimator.kf.state_covariance), expected_diag)

    def test_configure_initial_noise_state_covariance_scales_scale_term_for_xcycsr(
        self,
        bbox: np.ndarray,
    ) -> None:
        """P's scale/aspect-ratio terms follow the same 2x/10x scaling for the XCYCSR (sqrt-area) layout."""
        tracklet = _MinimalScaleAwareTracklet(bbox, state_estimator_class=XCYCSRStateEstimator)
        w, h = 40, 60

        expected_diag = [
            (2 * 0.05 * w) ** 2,
            (2 * 0.05 * h) ** 2,
            (2 * 0.05) ** 2 * w * h,  # (2 * sigma_p * sqrt(w*h)) ** 2
            (2 * 0.05) ** 2,  # (2 * sigma_p * 1.0) ** 2 — aspect ratio
            (10 * 0.00625 * w) ** 2,
            (10 * 0.00625 * h) ** 2,
            (10 * 0.00625) ** 2 * w * h,  # (10 * sigma_v * sqrt(w*h)) ** 2
        ]
        np.testing.assert_allclose(np.diag(tracklet.state_estimator.kf.state_covariance), expected_diag)

    # -------------------------------------------------------------------
    # _refresh_process_noise_from_state() / _refresh_measurement_noise_from_state()
    # -------------------------------------------------------------------

    def test_refresh_process_noise_from_state_sets_only_process_noise(self, bbox: np.ndarray) -> None:
        """Calling the process-noise refresh directly must pass only `process_noise` to `set_kf_covariances`."""
        tracklet = _MinimalScaleAwareTracklet(bbox)

        with patch.object(
            tracklet.state_estimator, "set_kf_covariances", wraps=tracklet.state_estimator.set_kf_covariances
        ) as mock_set:
            tracklet._refresh_process_noise_from_state()

        mock_set.assert_called_once()
        kwargs = mock_set.call_args.kwargs
        assert kwargs.get("process_noise") is not None
        assert kwargs.get("measurement_noise") is None
        assert kwargs.get("state_covariance") is None

    def test_refresh_measurement_noise_from_state_sets_only_measurement_noise(self, bbox: np.ndarray) -> None:
        """Calling the measurement-noise refresh directly must pass only `measurement_noise` to `set_kf_covariances`."""
        tracklet = _MinimalScaleAwareTracklet(bbox)

        with patch.object(
            tracklet.state_estimator, "set_kf_covariances", wraps=tracklet.state_estimator.set_kf_covariances
        ) as mock_set:
            tracklet._refresh_measurement_noise_from_state()

        mock_set.assert_called_once()
        kwargs = mock_set.call_args.kwargs
        assert kwargs.get("measurement_noise") is not None
        assert kwargs.get("process_noise") is None
        assert kwargs.get("state_covariance") is None

    # -------------------------------------------------------------------
    # predict() / update() flow
    # -------------------------------------------------------------------

    def test_predict_update_predict_refreshes_Q_and_R_from_box_size_at_each_step(self, bbox: np.ndarray) -> None:
        """A predict/update/predict sequence must rebuild Q and R from the box size current at each call.

        Each refresh reads the box size *before* the state-estimator step, so Q after the second predict must reflect
        the box the update() call left behind, not the one from before that update.
        """
        tracklet = _MinimalScaleAwareTracklet(bbox)

        pre_predict_bbox = tracklet.get_state_bbox()
        w = max(float(pre_predict_bbox[2] - pre_predict_bbox[0]), 1e-3)
        h = max(float(pre_predict_bbox[3] - pre_predict_bbox[1]), 1e-3)
        expected_first_Q = tracklet._build_process_noise(w, h)

        tracklet.predict()
        np.testing.assert_array_equal(tracklet.state_estimator.kf.process_noise, expected_first_Q)

        pre_update_bbox = tracklet.get_state_bbox()
        w = max(float(pre_update_bbox[2] - pre_update_bbox[0]), 1e-3)
        h = max(float(pre_update_bbox[3] - pre_update_bbox[1]), 1e-3)
        expected_R = tracklet._build_measurement_noise(w, h)

        tracklet.update(np.array([5.0, 5.0, 205.0, 205.0]))
        np.testing.assert_array_equal(tracklet.state_estimator.kf.measurement_noise, expected_R)

        pre_second_predict_bbox = tracklet.get_state_bbox()
        w = max(float(pre_second_predict_bbox[2] - pre_second_predict_bbox[0]), 1e-3)
        h = max(float(pre_second_predict_bbox[3] - pre_second_predict_bbox[1]), 1e-3)
        expected_second_Q = tracklet._build_process_noise(w, h)

        tracklet.predict()
        np.testing.assert_array_equal(tracklet.state_estimator.kf.process_noise, expected_second_Q)
        assert not np.array_equal(expected_first_Q, expected_second_Q)

    # -------------------------------------------------------------------
    # _clamp_state_bbox() dispatch
    # -------------------------------------------------------------------

    def test_clamp_state_bbox_fixes_inverted_xyxy_corners(self, bbox: np.ndarray) -> None:
        """For XYXY state, an inverted corner pair must be pulled back to `first_corner + 1e-3`."""
        tracklet = _MinimalScaleAwareTracklet(bbox, state_estimator_class=XYXYStateEstimator)
        kf_state = tracklet.state_estimator.kf.state
        kf_state[0, 0], kf_state[2, 0] = 10.0, 5.0  # x2 <= x1
        kf_state[1, 0], kf_state[3, 0] = 20.0, 3.0  # y2 <= y1

        tracklet._clamp_state_bbox()

        assert kf_state[2, 0] == pytest.approx(10.0 + 1e-3)
        assert kf_state[3, 0] == pytest.approx(20.0 + 1e-3)

    def test_clamp_state_bbox_leaves_valid_xyxy_corners_unchanged(self, bbox: np.ndarray) -> None:
        """A valid XYXY box (x2 > x1, y2 > y1) must pass through `_clamp_state_bbox` untouched."""
        tracklet = _MinimalScaleAwareTracklet(bbox, state_estimator_class=XYXYStateEstimator)
        before = tracklet.state_estimator.kf.state.copy()

        tracklet._clamp_state_bbox()

        np.testing.assert_array_equal(tracklet.state_estimator.kf.state, before)

    @pytest.mark.parametrize(
        ("width", "height"),
        [
            pytest.param(0.0, 0.0, id="zero"),
            pytest.param(-4.0, -6.0, id="negative"),
        ],
    )
    def test_clamp_state_bbox_clamps_nonpositive_xcycwh_dimensions(
        self,
        bbox: np.ndarray,
        width: float,
        height: float,
    ) -> None:
        """For XCYCWH state, non-positive width/height must clamp to `1e-3`."""
        tracklet = _MinimalScaleAwareTracklet(bbox, state_estimator_class=XCYCWHStateEstimator)
        kf_state = tracklet.state_estimator.kf.state
        kf_state[2, 0], kf_state[3, 0] = width, height

        tracklet._clamp_state_bbox()

        assert kf_state[2, 0] == pytest.approx(1e-3)
        assert kf_state[3, 0] == pytest.approx(1e-3)

    @pytest.mark.parametrize(
        ("scale", "aspect_ratio"),
        [
            pytest.param(0.0, 0.0, id="zero"),
            pytest.param(-100.0, -2.0, id="negative"),
        ],
    )
    def test_clamp_state_bbox_clamps_nonpositive_xcycsr_scale_and_ratio(
        self,
        bbox: np.ndarray,
        scale: float,
        aspect_ratio: float,
    ) -> None:
        """For XCYCSR state, non-positive scale/aspect-ratio must clamp to `1e-3`."""
        tracklet = _MinimalScaleAwareTracklet(bbox, state_estimator_class=XCYCSRStateEstimator)
        kf_state = tracklet.state_estimator.kf.state
        kf_state[2, 0], kf_state[3, 0] = scale, aspect_ratio

        tracklet._clamp_state_bbox()

        assert kf_state[2, 0] == pytest.approx(1e-3)
        assert kf_state[3, 0] == pytest.approx(1e-3)
