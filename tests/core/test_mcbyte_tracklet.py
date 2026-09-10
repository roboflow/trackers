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

from trackers.core.mcbyte.tracklet import McByteTracklet, _diagonal_matrix
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


class TestMcbyteTracklet:
    """Scale-aware noise refresh behavior for McByteTracklet."""

    def test_larger_box_has_larger_process_noise(self) -> None:
        """A bigger bounding box must produce strictly larger Q diagonal values."""
        small = McByteTracklet(np.array([0.0, 0.0, 20.0, 20.0]))
        large = McByteTracklet(np.array([0.0, 0.0, 200.0, 200.0]))

        small_Q = np.diag(small.state_estimator.kf.process_noise)
        large_Q = np.diag(large.state_estimator.kf.process_noise)

        assert np.all(large_Q > small_Q), "larger box must produce larger process noise diagonal"

    def test_larger_box_has_larger_measurement_noise(self) -> None:
        """A bigger bounding box must produce strictly larger R diagonal values."""
        small = McByteTracklet(np.array([0.0, 0.0, 20.0, 20.0]))
        large = McByteTracklet(np.array([0.0, 0.0, 200.0, 200.0]))

        small_R = np.diag(small.state_estimator.kf.measurement_noise)
        large_R = np.diag(large.state_estimator.kf.measurement_noise)

        assert np.all(large_R > small_R), "larger box must produce larger measurement noise diagonal"

    def test_predict_only_refreshes_process_noise(self) -> None:
        """Predict() must rebuild Q but leave R alone: KalmanFilter.predict never reads measurement_noise, so rebuilding
        it there is wasted work that update() will overwrite anyway."""
        tracklet = McByteTracklet(np.array([0.0, 0.0, 20.0, 20.0]))

        with patch.object(
            tracklet.state_estimator, "set_kf_covariances", wraps=tracklet.state_estimator.set_kf_covariances
        ) as mock_set:
            tracklet.predict()

        mock_set.assert_called_once()
        kwargs = mock_set.call_args.kwargs
        assert kwargs.get("process_noise") is not None, "predict() must refresh Q"
        assert kwargs.get("measurement_noise") is None, "predict() must not rebuild R"

    def test_update_only_refreshes_measurement_noise(self) -> None:
        """Update() must rebuild R but leave Q alone: KalmanFilter.update never reads process_noise, so rebuilding it
        there is wasted work that predict() will overwrite anyway."""
        tracklet = McByteTracklet(np.array([0.0, 0.0, 20.0, 20.0]))

        with patch.object(
            tracklet.state_estimator, "set_kf_covariances", wraps=tracklet.state_estimator.set_kf_covariances
        ) as mock_set:
            tracklet.update(np.array([5.0, 5.0, 205.0, 205.0]))

        mock_set.assert_called_once()
        kwargs = mock_set.call_args.kwargs
        assert kwargs.get("measurement_noise") is not None, "update() must refresh R"
        assert kwargs.get("process_noise") is None, "update() must not rebuild Q"

    def test_predict_process_noise_matches_pre_predict_box(
        self,
        tracklet: McByteTracklet,
    ) -> None:
        """Q after predict() must equal _build_process_noise() for the box size the tracklet had *before* that predict()
        call — the same size _refresh_process_noise_from_state() reads internally — across all three state
        representations, not just the kwarg it was called with."""
        bbox = tracklet.get_state_bbox()
        w = max(float(bbox[2] - bbox[0]), 1e-3)
        h = max(float(bbox[3] - bbox[1]), 1e-3)
        expected_Q = tracklet._build_process_noise(w, h)

        tracklet.predict()

        # Direct Q equality depends on the default frame_step=1.0 staying in the near-nominal band.
        np.testing.assert_array_equal(tracklet.state_estimator.kf.process_noise, expected_Q)

    def test_update_measurement_noise_matches_pre_update_box(
        self,
        tracklet: McByteTracklet,
    ) -> None:
        """R after update() must equal _build_measurement_noise() for the box size the tracklet had *before* that
        update() call (the predicted box, not the new observation) — across all three state representations, not just
        the kwarg it was called with."""
        bbox = tracklet.get_state_bbox()
        w = max(float(bbox[2] - bbox[0]), 1e-3)
        h = max(float(bbox[3] - bbox[1]), 1e-3)
        expected_R = tracklet._build_measurement_noise(w, h)

        tracklet.update(np.array([5.0, 5.0, 205.0, 205.0]))

        np.testing.assert_array_equal(tracklet.state_estimator.kf.measurement_noise, expected_R)

    def test_gap_predict_uses_refreshed_dwna_noise(
        self,
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

    def test_interleaved_predict_update_predict_refreshes_split_noise(
        self,
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


class TestDiagonalMatrix:
    """Regression coverage for ``_diagonal_matrix`` (PR #578 review, findings T1-T3).

    The existing Q/R/P assertions elsewhere in this file build their expected value by calling the production method
    under test, so a stride, offset, or ordering error in ``_diagonal_matrix`` would appear identically on both sides of
    the comparison and cancel. These tests build the expected matrix independently of the tracklet's noise-construction
    methods.
    """

    @pytest.mark.parametrize(
        "size",
        [
            pytest.param(1, id="stride-boundary"),
            pytest.param(4, id="measurement-noise-size"),
            pytest.param(7, id="xcycsr-process-noise-size"),
            pytest.param(8, id="default-process-noise-size"),
        ],
    )
    def test_matches_np_diag(self, size: int) -> None:
        """Output equals ``np.diag`` on the same 1-D input, at every runtime size.

        ``np.diag`` is NumPy's own trusted diagonal constructor, independent of ``_diagonal_matrix``'s hand-rolled
        stride write, so a divergence here is a real defect rather than an error shared with the assertion.
        """
        values = [float(i + 1) for i in range(size)]

        result = _diagonal_matrix(values)

        np.testing.assert_array_equal(result, np.diag(values))
        assert result.dtype == np.float64

    def test_off_diagonal_entries_are_zero(self) -> None:
        """Off-diagonal positions stay zero at the default 8x8 process-noise size.

        The stride trick ``matrix.flat[::size + 1]`` is correctness-critical at
        this width; a wrong stride would leave nonzero off-diagonal noise
        undetected by the monotonicity-only checks elsewhere in this file.
        """
        values = [float(i + 1) for i in range(8)]

        result = _diagonal_matrix(values)

        assert np.count_nonzero(result - np.diag(np.diag(result))) == 0

    def test_diagonal_values_preserve_input_order(self) -> None:
        """Diagonal entries land in the same order as the input list, unreversed.

        Expected values are written independently of both ``_diagonal_matrix`` and ``np.diag`` via direct fancy
        indexing, so a reversed or shuffled stride write would be caught rather than masked by a shared construction
        path.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
        expected = np.zeros((7, 7), dtype=np.float64)
        expected[np.arange(7), np.arange(7)] = values

        result = _diagonal_matrix(values)

        np.testing.assert_array_equal(result, expected)
