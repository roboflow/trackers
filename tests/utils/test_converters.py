# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from trackers.utils.converters import (
    xcycsr_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_xcycsr,
    xyxy_to_xywh,
)


class TestXYWHConversion:
    """xyxy ↔ xywh conversions and round-trip."""

    @pytest.mark.parametrize(
        ("xyxy", "expected"),
        [
            pytest.param(np.array([0.0, 0.0, 10.0, 20.0]), np.array([5.0, 10.0, 10.0, 20.0]), id="origin-rect"),
            pytest.param(np.array([-2.0, -4.0, 2.0, 4.0]), np.array([0.0, 0.0, 4.0, 8.0]), id="neg-coords"),
        ],
    )
    def test_xyxy_to_xywh(self, xyxy: np.ndarray, expected: np.ndarray) -> None:
        """xyxy_to_xywh converts a single 1-D xyxy box to (cx, cy, w, h)."""
        result = xyxy_to_xywh(xyxy)
        assert result.shape == (4,)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_xyxy_to_xywh_batch(self) -> None:
        """xyxy_to_xywh converts a 2-D batch of xyxy boxes element-wise."""
        xyxy = np.array([[0.0, 0.0, 2.0, 2.0], [10.0, 20.0, 30.0, 50.0]])
        expected = np.array([[1.0, 1.0, 2.0, 2.0], [20.0, 35.0, 20.0, 30.0]])
        result = xyxy_to_xywh(xyxy)
        assert result.shape == (2, 4)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    @pytest.mark.parametrize(
        ("xywh", "expected"),
        [
            pytest.param(np.array([5.0, 10.0, 10.0, 20.0]), np.array([0.0, 0.0, 10.0, 20.0]), id="origin-rect"),
            pytest.param(np.array([0.0, 0.0, 4.0, 8.0]), np.array([-2.0, -4.0, 2.0, 4.0]), id="neg-coords"),
        ],
    )
    def test_xywh_to_xyxy(self, xywh: np.ndarray, expected: np.ndarray) -> None:
        """xywh_to_xyxy converts a single 1-D (cx, cy, w, h) box to xyxy."""
        result = xywh_to_xyxy(xywh)
        assert result.shape == (4,)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_xywh_to_xyxy_batch(self) -> None:
        """xywh_to_xyxy converts a 2-D batch of (cx, cy, w, h) boxes element-wise."""
        xywh = np.array([[1.0, 1.0, 2.0, 2.0], [20.0, 35.0, 20.0, 30.0]])
        expected = np.array([[0.0, 0.0, 2.0, 2.0], [10.0, 20.0, 30.0, 50.0]])
        result = xywh_to_xyxy(xywh)
        assert result.shape == (2, 4)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    @pytest.mark.parametrize(
        "xyxy",
        [
            pytest.param(np.array([0.0, 0.0, 10.0, 20.0]), id="origin-rect"),
            pytest.param(np.array([10.0, 20.0, 30.0, 50.0]), id="offset-rect"),
            pytest.param(np.array([-2.0, -4.0, 2.0, 4.0]), id="neg-coords"),
        ],
    )
    def test_roundtrip(self, xyxy: np.ndarray) -> None:
        """xyxy → xywh → xyxy round-trip recovers the original box without drift."""
        xywh = xyxy_to_xywh(xyxy)
        recovered = xywh_to_xyxy(xywh)
        np.testing.assert_array_almost_equal(recovered, xyxy, decimal=6)


class TestXCYCSRConversion:
    """xyxy ↔ xcycsr (center-x, center-y, scale, aspect) conversions and round-trip."""

    @pytest.mark.parametrize(
        ("xyxy", "expected"),
        [
            pytest.param(
                np.array([0.0, 0.0, 1.0, 1.0]),
                np.array([0.5, 0.5, 1.0, 1.0]),
                id="unit-square",
            ),
            pytest.param(
                np.array([10.0, 20.0, 12.0, 24.0]),
                np.array([11.0, 22.0, 8.0, 0.5]),
                id="rect-2x4",
            ),
            pytest.param(
                np.array([0.0, 0.0, 100.0, 50.0]),
                np.array([50.0, 25.0, 5000.0, 2.0]),
                id="wide-rect",
            ),
            pytest.param(
                np.array([5.0, 5.0, 15.0, 55.0]),
                np.array([10.0, 30.0, 500.0, 0.2]),
                id="tall-rect",
            ),
            pytest.param(
                np.array([-5.0, -5.0, 5.0, 5.0]),
                np.array([0.0, 0.0, 100.0, 1.0]),
                id="neg-coords",
            ),
            pytest.param(
                np.array([0.0, 0.0, 0.001, 0.001]),
                np.array([0.0005, 0.0005, 0.000001, 0.999001]),
                id="sub-pixel",
            ),
            pytest.param(
                np.array([0.0, 0.0, 10000.0, 10000.0]),
                np.array([5000.0, 5000.0, 100000000.0, 1.0]),
                id="large-box",
            ),
        ],
    )
    def test_xyxy_to_xcycsr(self, xyxy: np.ndarray, expected: np.ndarray) -> None:
        """xyxy_to_xcycsr converts a 1-D xyxy box to (cx, cy, scale, aspect)."""
        result = xyxy_to_xcycsr(xyxy)
        assert result.shape == (4,)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_xyxy_to_xcycsr_zero_height(self) -> None:
        """A zero-height box yields a finite xcycsr (no NaN/Inf) with scale=0."""
        xyxy = np.array([0.0, 0.0, 10.0, 0.0])
        result = xyxy_to_xcycsr(xyxy)
        assert np.isfinite(result).all()
        assert result[2] == 0.0

    def test_xyxy_to_xcycsr_2d_input(self) -> None:
        """A 1-row 2-D xyxy input keeps its 2-D shape after conversion."""
        xyxy = np.array([[0.0, 0.0, 1.0, 1.0]])
        result = xyxy_to_xcycsr(xyxy)
        assert result.shape == (1, 4)
        np.testing.assert_array_almost_equal(result[0], np.array([0.5, 0.5, 1.0, 1.0]), decimal=5)

    def test_xyxy_to_xcycsr_empty(self) -> None:
        """An empty (0, 4) xyxy batch returns an empty (0, 4) xcycsr batch."""
        xyxy = np.zeros((0, 4))
        result = xyxy_to_xcycsr(xyxy)
        assert result.shape == (0, 4)

    @pytest.mark.parametrize(
        ("xcycsr", "expected"),
        [
            pytest.param(
                np.array([0.5, 0.5, 1.0, 1.0]),
                np.array([0.0, 0.0, 1.0, 1.0]),
                id="unit-square",
            ),
            pytest.param(
                np.array([11.0, 22.0, 8.0, 0.5]),
                np.array([10.0, 20.0, 12.0, 24.0]),
                id="rect-2x4",
            ),
            pytest.param(
                np.array([50.0, 25.0, 5000.0, 2.0]),
                np.array([0.0, 0.0, 100.0, 50.0]),
                id="wide-box",
            ),
            pytest.param(
                np.array([10.0, 30.0, 500.0, 0.2]),
                np.array([5.0, 5.0, 15.0, 55.0]),
                id="tall-box",
            ),
            pytest.param(
                np.array([0.0, 0.0, 100.0, 1.0]),
                np.array([-5.0, -5.0, 5.0, 5.0]),
                id="center-at-origin",
            ),
            pytest.param(
                np.array([0.0005, 0.0005, 0.000001, 1.0]),
                np.array([0.0, 0.0, 0.001, 0.001]),
                id="very-small",
            ),
            pytest.param(
                np.array([5000.0, 5000.0, 100000000.0, 1.0]),
                np.array([0.0, 0.0, 10000.0, 10000.0]),
                id="very-large",
            ),
        ],
    )
    def test_xcycsr_to_xyxy(self, xcycsr: np.ndarray, expected: np.ndarray) -> None:
        """xcycsr_to_xyxy converts a 1-D (cx, cy, scale, aspect) box back to xyxy."""
        result = xcycsr_to_xyxy(xcycsr)
        assert result.shape == (4,)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    @pytest.mark.parametrize(
        "xcycsr",
        [
            pytest.param(np.array([10.0, 20.0, 0.0, 1.0]), id="zero-scale"),
            pytest.param(np.array([10.0, 20.0, 100.0, 0.0]), id="zero-aspect"),
        ],
    )
    def test_xcycsr_to_xyxy_degenerate_collapses_to_point(self, xcycsr: np.ndarray) -> None:
        """Zero scale or zero aspect-ratio decodes to a finite point box at (cx, cy)."""
        result = xcycsr_to_xyxy(xcycsr)
        assert np.isfinite(result).all()
        np.testing.assert_array_almost_equal(result, [10.0, 20.0, 10.0, 20.0])

    def test_xcycsr_to_xyxy_batch_mixed_degenerate(self) -> None:
        """Batch path handles a mix of normal and degenerate boxes without error."""
        xcycsr = np.array(
            [
                [10.0, 20.0, 100.0, 1.0],  # normal box
                [10.0, 20.0, 0.0, 1.0],  # zero scale
                [10.0, 20.0, 100.0, 0.0],  # zero aspect
            ]
        )
        result = xcycsr_to_xyxy(xcycsr)
        assert np.isfinite(result).all()
        assert result.shape == (3, 4)
        np.testing.assert_array_almost_equal(result[0], [5.0, 15.0, 15.0, 25.0])
        np.testing.assert_array_almost_equal(result[1], [10.0, 20.0, 10.0, 20.0])
        np.testing.assert_array_almost_equal(result[2], [10.0, 20.0, 10.0, 20.0])

    def test_xcycsr_to_xyxy_negative_scale(self) -> None:
        """A negative-scale xcycsr decodes with NaN entries (sqrt of negative)."""
        xcycsr = np.array([10.0, 20.0, -100.0, 1.0])
        result = xcycsr_to_xyxy(xcycsr)
        assert np.isnan(result).any()

    def test_xcycsr_to_xyxy_batch_negative_scale(self) -> None:
        """Batch path propagates NaN for negative-scale inputs (sqrt of negative)."""
        xcycsr = np.array([[10.0, 20.0, -100.0, 1.0]])
        result = xcycsr_to_xyxy(xcycsr)
        assert np.isnan(result).any()

    def test_xcycsr_to_xyxy_2d_input(self) -> None:
        """A 1-row 2-D xcycsr input keeps its 2-D shape after conversion."""
        xcycsr = np.array([[0.5, 0.5, 1.0, 1.0]])
        result = xcycsr_to_xyxy(xcycsr)
        assert result.shape == (1, 4)
        np.testing.assert_array_almost_equal(result[0], np.array([0.0, 0.0, 1.0, 1.0]), decimal=5)

    def test_xcycsr_to_xyxy_empty(self) -> None:
        """An empty (0, 4) xcycsr batch returns an empty (0, 4) xyxy batch."""
        xcycsr = np.zeros((0, 4))
        result = xcycsr_to_xyxy(xcycsr)
        assert result.shape == (0, 4)

    @pytest.mark.parametrize(
        "xyxy",
        [
            pytest.param(np.array([0.0, 0.0, 1.0, 1.0]), id="unit-square"),
            pytest.param(np.array([10.0, 20.0, 30.0, 50.0]), id="offset-rect"),
            pytest.param(np.array([100.0, 200.0, 150.0, 210.0]), id="medium-rect"),
            pytest.param(np.array([-10.0, -20.0, 10.0, 20.0]), id="neg-coords"),
            pytest.param(np.array([0.0, 0.0, 0.01, 0.01]), id="sub-pixel"),
            pytest.param(np.array([0.0, 0.0, 1000.0, 500.0]), id="wide-large"),
        ],
    )
    def test_roundtrip(self, xyxy: np.ndarray) -> None:
        """xyxy → xcycsr → xyxy round-trip recovers the original 1-D box."""
        xcycsr = xyxy_to_xcycsr(xyxy)
        recovered = xcycsr_to_xyxy(xcycsr)
        np.testing.assert_array_almost_equal(recovered, xyxy, decimal=5)

    def test_roundtrip_2d(self) -> None:
        """xyxy → xcycsr → xyxy round-trip preserves the original 2-D shape."""
        xyxy = np.array([[0.0, 0.0, 10.0, 10.0]])
        xcycsr = xyxy_to_xcycsr(xyxy)
        recovered = xcycsr_to_xyxy(xcycsr)
        assert recovered.shape == (1, 4)
        np.testing.assert_array_almost_equal(recovered, xyxy, decimal=5)

    def test_roundtrip_degenerate_is_lossy(self) -> None:
        """xyxy → xcycsr → xyxy is lossy for zero-area boxes; asserts the known result."""
        xyxy = np.array([0.0, 0.0, 10.0, 0.0])
        xcycsr = xyxy_to_xcycsr(xyxy)
        recovered = xcycsr_to_xyxy(xcycsr)
        # scale = w*h = 0 erases size information; width collapses and cannot be recovered.
        np.testing.assert_array_almost_equal(recovered, [5.0, 0.0, 5.0, 0.0], decimal=5)
