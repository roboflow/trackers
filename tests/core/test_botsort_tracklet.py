# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""BoT-SORT-specific tracklet tests.

Generic predict/update contracts (time_since_update, age) are covered for all tracklet classes in test_tracklets.py.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from trackers.core.botsort.tracklet import BoTSORTTracklet
from trackers.core.mcbyte.tracklet import McByteTracklet
from trackers.utils.predict_timing import PredictTiming
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


class TestBotsortTracklet:
    """BoT-SORT tracklet predict/update, scale-aware noise, and CMC behavior."""

    # -------------------------------------------------------------------
    # predict()
    # -------------------------------------------------------------------

    @pytest.mark.parametrize(
        "estimator_class",
        [XCYCWHStateEstimator, XYXYStateEstimator, XCYCSRStateEstimator],
    )
    def test_predict_keeps_valid_bbox(
        self,
        bbox: np.ndarray,
        estimator_class: type[BaseStateEstimator],
    ) -> None:
        """BBox geometry and scale-aware Q stay valid through repeated predictions."""
        tracklet = BoTSORTTracklet(bbox, state_estimator_class=estimator_class)
        tracklet.state_estimator.kf.state[6, 0] = 1.0
        if estimator_class is not XCYCSRStateEstimator:
            tracklet.state_estimator.kf.state[7, 0] = 1.0

        initial_process_noise = tracklet.state_estimator.kf.process_noise.copy()
        for _ in range(50):
            state_bbox = tracklet.get_state_bbox()
            w = max(float(state_bbox[2] - state_bbox[0]), 1e-3)
            h = max(float(state_bbox[3] - state_bbox[1]), 1e-3)
            expected_Q = tracklet._build_process_noise(w, h)
            tracklet.predict()
            np.testing.assert_array_equal(tracklet.state_estimator.kf.process_noise, expected_Q)

        state_bbox = tracklet.get_state_bbox()
        assert state_bbox[2] > state_bbox[0], "width must stay positive after predicts"
        assert state_bbox[3] > state_bbox[1], "height must stay positive after predicts"
        assert not np.array_equal(tracklet.state_estimator.kf.process_noise, initial_process_noise)

    # -------------------------------------------------------------------
    # Scale-aware noise
    # -------------------------------------------------------------------

    @pytest.mark.parametrize("tracklet_cls", [BoTSORTTracklet, McByteTracklet])
    def test_larger_box_has_larger_process_noise(
        self,
        tracklet_cls: type[BoTSORTTracklet] | type[McByteTracklet],
    ) -> None:
        """A bigger bounding box must produce strictly larger Q diagonal values."""
        small = tracklet_cls(np.array([0.0, 0.0, 20.0, 20.0]))
        large = tracklet_cls(np.array([0.0, 0.0, 200.0, 200.0]))

        small_Q = np.diag(small.state_estimator.kf.process_noise)
        large_Q = np.diag(large.state_estimator.kf.process_noise)

        assert np.all(large_Q > small_Q), "larger box must produce larger process noise diagonal"

    @pytest.mark.parametrize("tracklet_cls", [BoTSORTTracklet, McByteTracklet])
    def test_current_wh_reconstructs_corner_subtraction_without_bbox_decode(
        self,
        tracklet_cls: type[BoTSORTTracklet] | type[McByteTracklet],
    ) -> None:
        """The default estimator fast path must preserve center-dependent corner rounding without decoding an array."""
        tracklet = tracklet_cls(np.array([0.0, 0.0, 20.0, 20.0]))
        tracklet.state_estimator.kf.state[:4, 0] = np.array([1e12, -1e12, 0.1, 0.2])
        bbox = tracklet.state_estimator.state_to_bbox()
        expected = np.array(
            [
                max(float(bbox[2] - bbox[0]), 1e-3),
                max(float(bbox[3] - bbox[1]), 1e-3),
            ]
        )
        direct = tracklet.state_estimator.kf.state[2:4, 0]
        assert not np.array_equal(expected.view(np.uint64), direct.view(np.uint64))

        with patch.object(tracklet.state_estimator, "state_to_bbox", side_effect=AssertionError("decoded bbox")):
            actual = np.array(tracklet._current_wh())

        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))

    @pytest.mark.parametrize("tracklet_cls", [BoTSORTTracklet, McByteTracklet])
    def test_current_wh_matches_decoder_for_float32_restored_state(
        self,
        tracklet_cls: type[BoTSORTTracklet] | type[McByteTracklet],
    ) -> None:
        """Restored float32 state must retain the decoder's float64 corner arithmetic."""
        tracklet = tracklet_cls(np.array([0.0, 0.0, 20.0, 20.0]))
        restored_state = tracklet.state_estimator.get_state()
        restored_state["state"] = restored_state["state"].astype(np.float32)
        restored_state["state"][:4, 0] = np.array([1e12, -1e12, 0.1, 0.2], dtype=np.float32)
        tracklet.state_estimator.set_state(restored_state)
        assert tracklet.state_estimator.kf.state.dtype == np.float32

        decoded_bbox = tracklet.state_estimator.state_to_bbox()
        expected = (
            max(float(decoded_bbox[2] - decoded_bbox[0]), 1e-3),
            max(float(decoded_bbox[3] - decoded_bbox[1]), 1e-3),
        )

        assert tracklet._current_wh() == expected

    @pytest.mark.parametrize("tracklet_cls", [BoTSORTTracklet, McByteTracklet])
    @pytest.mark.parametrize(
        ("width", "height"),
        [
            pytest.param(0.0, 0.0, id="zero"),
            pytest.param(-4.0, -6.0, id="negative"),
        ],
    )
    def test_current_wh_clamps_nonpositive_dimensions_to_minimum(
        self,
        tracklet_cls: type[BoTSORTTracklet] | type[McByteTracklet],
        width: float,
        height: float,
    ) -> None:
        """Zero and negative default-estimator dimensions must match the decoder's 1e-3 clamp."""
        tracklet = tracklet_cls(np.array([0.0, 0.0, 20.0, 20.0]))
        tracklet.state_estimator.kf.state[:4, 0] = np.array([100.0, -100.0, width, height])

        decoded_bbox = tracklet.state_estimator.state_to_bbox()
        expected = (
            max(float(decoded_bbox[2] - decoded_bbox[0]), 1e-3),
            max(float(decoded_bbox[3] - decoded_bbox[1]), 1e-3),
        )

        assert expected == (1e-3, 1e-3)
        assert tracklet._current_wh() == expected

    @pytest.mark.parametrize("tracklet_cls", [BoTSORTTracklet, McByteTracklet])
    @pytest.mark.parametrize("estimator_class", [XYXYStateEstimator, XCYCSRStateEstimator])
    def test_current_wh_keeps_bbox_decode_for_other_state_representations(
        self,
        bbox: np.ndarray,
        tracklet_cls: type[BoTSORTTracklet] | type[McByteTracklet],
        estimator_class: type[BaseStateEstimator],
    ) -> None:
        """Non-XCYCWH representations retain the state estimator's bbox decoder."""
        tracklet = tracklet_cls(bbox, state_estimator_class=estimator_class)
        decoded_bbox = tracklet.state_estimator.state_to_bbox()
        expected = (
            max(float(decoded_bbox[2] - decoded_bbox[0]), 1e-3),
            max(float(decoded_bbox[3] - decoded_bbox[1]), 1e-3),
        )

        with patch.object(
            tracklet.state_estimator, "state_to_bbox", wraps=tracklet.state_estimator.state_to_bbox
        ) as mock_decode:
            actual = tracklet._current_wh()

        mock_decode.assert_called_once_with()
        assert actual == expected

    @pytest.mark.parametrize("tracklet_cls", [BoTSORTTracklet, McByteTracklet])
    def test_current_wh_keeps_bbox_decode_for_xcycwh_subclass(
        self,
        bbox: np.ndarray,
        tracklet_cls: type[BoTSORTTracklet] | type[McByteTracklet],
    ) -> None:
        """A subclass overriding ``state_to_bbox()`` must not be routed through the exact-class fast path."""

        class CustomXCYCWHStateEstimator(XCYCWHStateEstimator):
            def state_to_bbox(self) -> np.ndarray:
                return np.array([3.0, 5.0, 50.0, 76.0])

        tracklet = tracklet_cls(bbox, state_estimator_class=CustomXCYCWHStateEstimator)

        with patch.object(
            tracklet.state_estimator, "state_to_bbox", wraps=tracklet.state_estimator.state_to_bbox
        ) as mock_decode:
            actual = tracklet._current_wh()

        mock_decode.assert_called_once_with()
        assert actual == (47.0, 71.0)

    @pytest.mark.parametrize("tracklet_cls", [BoTSORTTracklet, McByteTracklet])
    def test_larger_box_has_larger_measurement_noise(
        self,
        tracklet_cls: type[BoTSORTTracklet] | type[McByteTracklet],
    ) -> None:
        """A bigger bounding box must produce strictly larger R diagonal values."""
        small = tracklet_cls(np.array([0.0, 0.0, 20.0, 20.0]))
        large = tracklet_cls(np.array([0.0, 0.0, 200.0, 200.0]))

        small_R = np.diag(small.state_estimator.kf.measurement_noise)
        large_R = np.diag(large.state_estimator.kf.measurement_noise)

        assert np.all(large_R > small_R), "larger box must produce larger measurement noise diagonal"

    @pytest.mark.parametrize("tracklet_cls", [BoTSORTTracklet, McByteTracklet])
    def test_predict_only_refreshes_process_noise(
        self,
        tracklet_cls: type[BoTSORTTracklet] | type[McByteTracklet],
    ) -> None:
        """Predict() must rebuild Q but leave R alone: KalmanFilter.predict never reads measurement_noise, so rebuilding
        it there is wasted work that update() will overwrite anyway."""
        tracklet = tracklet_cls(np.array([0.0, 0.0, 20.0, 20.0]))

        with patch.object(
            tracklet.state_estimator, "set_kf_covariances", wraps=tracklet.state_estimator.set_kf_covariances
        ) as mock_set:
            tracklet.predict()

        mock_set.assert_called_once()
        kwargs = mock_set.call_args.kwargs
        assert kwargs.get("process_noise") is not None, "predict() must refresh Q"
        assert kwargs.get("measurement_noise") is None, "predict() must not rebuild R"

    @pytest.mark.parametrize("tracklet_cls", [BoTSORTTracklet, McByteTracklet])
    def test_update_only_refreshes_measurement_noise(
        self,
        tracklet_cls: type[BoTSORTTracklet] | type[McByteTracklet],
    ) -> None:
        """Update() must rebuild R but leave Q alone: KalmanFilter.update never reads process_noise, so rebuilding it
        there is wasted work that predict() will overwrite anyway."""
        tracklet = tracklet_cls(np.array([0.0, 0.0, 20.0, 20.0]))

        with patch.object(
            tracklet.state_estimator, "set_kf_covariances", wraps=tracklet.state_estimator.set_kf_covariances
        ) as mock_set:
            tracklet.update(np.array([5.0, 5.0, 205.0, 205.0]))

        mock_set.assert_called_once()
        kwargs = mock_set.call_args.kwargs
        assert kwargs.get("measurement_noise") is not None, "update() must refresh R"
        assert kwargs.get("process_noise") is None, "update() must not rebuild Q"

    @pytest.mark.parametrize("tracklet_cls", [BoTSORTTracklet, McByteTracklet])
    @pytest.mark.parametrize(
        "estimator_class",
        [XCYCWHStateEstimator, XYXYStateEstimator, XCYCSRStateEstimator],
    )
    def test_predict_process_noise_matches_pre_predict_box(
        self,
        bbox: np.ndarray,
        tracklet_cls: type[BoTSORTTracklet] | type[McByteTracklet],
        estimator_class: type[BaseStateEstimator],
    ) -> None:
        """Q after predict() must equal _build_process_noise() for the box size the tracklet had *before* that predict()
        call — the same size _refresh_process_noise_from_state() reads internally — across all three state
        representations, not just the kwarg it was called with."""
        tracklet = tracklet_cls(bbox, state_estimator_class=estimator_class)
        state_bbox = tracklet.get_state_bbox()
        w = max(float(state_bbox[2] - state_bbox[0]), 1e-3)
        h = max(float(state_bbox[3] - state_bbox[1]), 1e-3)
        expected_Q = tracklet._build_process_noise(w, h)

        tracklet.predict()

        # Direct Q equality depends on the default frame_step=1.0 staying in the near-nominal band.
        np.testing.assert_array_equal(tracklet.state_estimator.kf.process_noise, expected_Q)

    @pytest.mark.parametrize("tracklet_cls", [BoTSORTTracklet, McByteTracklet])
    @pytest.mark.parametrize(
        "estimator_class",
        [XCYCWHStateEstimator, XYXYStateEstimator, XCYCSRStateEstimator],
    )
    def test_update_measurement_noise_matches_pre_update_box(
        self,
        bbox: np.ndarray,
        tracklet_cls: type[BoTSORTTracklet] | type[McByteTracklet],
        estimator_class: type[BaseStateEstimator],
    ) -> None:
        """R after update() must equal _build_measurement_noise() for the box size the tracklet had *before* that
        update() call (the predicted box, not the new observation) — across all three state representations, not just
        the kwarg it was called with."""
        tracklet = tracklet_cls(bbox, state_estimator_class=estimator_class)
        state_bbox = tracklet.get_state_bbox()
        w = max(float(state_bbox[2] - state_bbox[0]), 1e-3)
        h = max(float(state_bbox[3] - state_bbox[1]), 1e-3)
        expected_R = tracklet._build_measurement_noise(w, h)

        tracklet.update(np.array([5.0, 5.0, 205.0, 205.0]))

        np.testing.assert_array_equal(tracklet.state_estimator.kf.measurement_noise, expected_R)

    def test_gap_predict_uses_refreshed_dwna_noise(
        self,
        tracklet: BoTSORTTracklet,
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
        tracklet: BoTSORTTracklet,
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

    @pytest.mark.parametrize("tracklet_cls", [BoTSORTTracklet, McByteTracklet])
    @pytest.mark.parametrize(
        "estimator_class",
        [XCYCWHStateEstimator, XYXYStateEstimator, XCYCSRStateEstimator],
    )
    def test_noise_state_round_trip_restores_filter_matrices(
        self,
        bbox: np.ndarray,
        tracklet_cls: type[BoTSORTTracklet] | type[McByteTracklet],
        estimator_class: type[BaseStateEstimator],
    ) -> None:
        """Kalman state restoration must preserve split Q/R after predict and update."""
        source = tracklet_cls(bbox, state_estimator_class=estimator_class)
        source.predict()
        source.update(np.array([5.0, 5.0, 205.0, 205.0]))
        snapshot = source.state_estimator.kf.get_state()

        restored = tracklet_cls(np.array([0.0, 0.0, 10.0, 10.0]), state_estimator_class=estimator_class)
        restored.state_estimator.kf.set_state(snapshot)

        restored_kf = restored.state_estimator.kf
        np.testing.assert_array_equal(restored_kf.process_noise, snapshot["process_noise"])
        np.testing.assert_array_equal(restored_kf.measurement_noise, snapshot["measurement_noise"])
        np.testing.assert_array_equal(restored_kf.state, snapshot["state"])
        np.testing.assert_array_equal(restored_kf.state_covariance, snapshot["state_covariance"])

    def test_botsort_and_mcbyte_noise_constants_agree(self) -> None:
        """Sibling tracklets must retain identical scale-aware noise constants."""
        assert (BoTSORTTracklet._SIGMA_P, BoTSORTTracklet._SIGMA_V, BoTSORTTracklet._SIGMA_M) == (
            McByteTracklet._SIGMA_P,
            McByteTracklet._SIGMA_V,
            McByteTracklet._SIGMA_M,
        )

    # -------------------------------------------------------------------
    # apply_cmc()
    # -------------------------------------------------------------------

    def test_apply_cmc_none_is_noop(
        self,
        tracklet: BoTSORTTracklet,
    ) -> None:
        """apply_cmc(None) must leave state and covariance unchanged."""
        state_before = tracklet.state_estimator.kf.state.copy()
        P_before = tracklet.state_estimator.kf.state_covariance.copy()

        tracklet.apply_cmc(None)

        np.testing.assert_array_equal(tracklet.state_estimator.kf.state, state_before)
        np.testing.assert_array_equal(tracklet.state_estimator.kf.state_covariance, P_before)

    def test_apply_cmc_identity_is_noop(
        self,
        tracklet: BoTSORTTracklet,
    ) -> None:
        """apply_cmc with an identity transform must leave state unchanged."""
        state_before = tracklet.state_estimator.kf.state.copy()
        tracklet.apply_cmc(np.eye(2, 3, dtype=np.float32))
        np.testing.assert_allclose(tracklet.state_estimator.kf.state, state_before, atol=1e-6)

    def test_apply_cmc_translates_center(
        self,
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

    def test_apply_cmc_does_not_affect_wh(
        self,
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
