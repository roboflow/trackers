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
            (np.array([0.0, 0.0, 10.0, 20.0]), np.array([5.0, 10.0, 10.0, 20.0])),
            (np.array([-2.0, -4.0, 2.0, 4.0]), np.array([0.0, 0.0, 4.0, 8.0])),
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
            (np.array([5.0, 10.0, 10.0, 20.0]), np.array([0.0, 0.0, 10.0, 20.0])),
            (np.array([0.0, 0.0, 4.0, 8.0]), np.array([-2.0, -4.0, 2.0, 4.0])),
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
            np.array([0.0, 0.0, 10.0, 20.0]),
            np.array([10.0, 20.0, 30.0, 50.0]),
            np.array([-2.0, -4.0, 2.0, 4.0]),
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
            # Unit square at origin
            (
                np.array([0.0, 0.0, 1.0, 1.0]),
                np.array([0.5, 0.5, 1.0, 1.0]),
            ),
            # Rectangle 2x4 at (10, 20)
            (
                np.array([10.0, 20.0, 12.0, 24.0]),
                np.array([11.0, 22.0, 8.0, 0.5]),
            ),
            # Wide rectangle (aspect ratio > 1)
            (
                np.array([0.0, 0.0, 100.0, 50.0]),
                np.array([50.0, 25.0, 5000.0, 2.0]),
            ),
            # Tall rectangle (aspect ratio < 1)
            (
                np.array([5.0, 5.0, 15.0, 55.0]),
                np.array([10.0, 30.0, 500.0, 0.2]),
            ),
            # Negative coordinates (box crossing origin)
            (
                np.array([-5.0, -5.0, 5.0, 5.0]),
                np.array([0.0, 0.0, 100.0, 1.0]),
            ),
            # Very small box (sub-pixel) - aspect ratio affected by epsilon protection
            (
                np.array([0.0, 0.0, 0.001, 0.001]),
                np.array([0.0005, 0.0005, 0.000001, 0.999001]),
            ),
            # Very large box
            (
                np.array([0.0, 0.0, 10000.0, 10000.0]),
                np.array([5000.0, 5000.0, 100000000.0, 1.0]),
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
        np.testing.assert_array_almost_equal(
            result[0], np.array([0.5, 0.5, 1.0, 1.0]), decimal=5
        )

    def test_xyxy_to_xcycsr_empty(self) -> None:
        """An empty (0, 4) xyxy batch returns an empty (0, 4) xcycsr batch."""
        xyxy = np.zeros((0, 4))
        result = xyxy_to_xcycsr(xyxy)
        assert result.shape == (0, 4)

    @pytest.mark.parametrize(
        ("xcycsr", "expected"),
        [
            # Unit square at (0.5, 0.5)
            (
                np.array([0.5, 0.5, 1.0, 1.0]),
                np.array([0.0, 0.0, 1.0, 1.0]),
            ),
            # Rectangle at (11, 22) with area=8, ratio=0.5
            (
                np.array([11.0, 22.0, 8.0, 0.5]),
                np.array([10.0, 20.0, 12.0, 24.0]),
            ),
            # Wide box
            (
                np.array([50.0, 25.0, 5000.0, 2.0]),
                np.array([0.0, 0.0, 100.0, 50.0]),
            ),
            # Tall box
            (
                np.array([10.0, 30.0, 500.0, 0.2]),
                np.array([5.0, 5.0, 15.0, 55.0]),
            ),
            # Center at origin
            (
                np.array([0.0, 0.0, 100.0, 1.0]),
                np.array([-5.0, -5.0, 5.0, 5.0]),
            ),
            # Very small box
            (
                np.array([0.0005, 0.0005, 0.000001, 1.0]),
                np.array([0.0, 0.0, 0.001, 0.001]),
            ),
            # Very large box
            (
                np.array([5000.0, 5000.0, 100000000.0, 1.0]),
                np.array([0.0, 0.0, 10000.0, 10000.0]),
            ),
        ],
    )
    def test_xcycsr_to_xyxy(self, xcycsr: np.ndarray, expected: np.ndarray) -> None:
        """xcycsr_to_xyxy converts a 1-D (cx, cy, scale, aspect) box back to xyxy."""
        result = xcycsr_to_xyxy(xcycsr)
        assert result.shape == (4,)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_xcycsr_to_xyxy_zero_scale(self) -> None:
        """Zero-scale xcycsr decodes with NaN y-coords; x-center is preserved."""
        xcycsr = np.array([10.0, 20.0, 0.0, 1.0])
        result = xcycsr_to_xyxy(xcycsr)
        assert result[0] == result[2] == 10.0
        assert np.isnan(result[1]) and np.isnan(result[3])

    def test_xcycsr_to_xyxy_zero_aspect(self) -> None:
        """A zero aspect-ratio xcycsr decodes to a non-finite (NaN/Inf) box."""
        xcycsr = np.array([10.0, 20.0, 100.0, 0.0])
        result = xcycsr_to_xyxy(xcycsr)
        assert np.isnan(result).any() or np.isinf(result).any()

    def test_xcycsr_to_xyxy_negative_scale(self) -> None:
        """A negative-scale xcycsr decodes with NaN entries (sqrt of negative)."""
        xcycsr = np.array([10.0, 20.0, -100.0, 1.0])
        result = xcycsr_to_xyxy(xcycsr)
        assert np.isnan(result).any()

    def test_xcycsr_to_xyxy_2d_input(self) -> None:
        """A 1-row 2-D xcycsr input keeps its 2-D shape after conversion."""
        xcycsr = np.array([[0.5, 0.5, 1.0, 1.0]])
        result = xcycsr_to_xyxy(xcycsr)
        assert result.shape == (1, 4)
        np.testing.assert_array_almost_equal(
            result[0], np.array([0.0, 0.0, 1.0, 1.0]), decimal=5
        )

    def test_xcycsr_to_xyxy_empty(self) -> None:
        """An empty (0, 4) xcycsr batch returns an empty (0, 4) xyxy batch."""
        xcycsr = np.zeros((0, 4))
        result = xcycsr_to_xyxy(xcycsr)
        assert result.shape == (0, 4)

    @pytest.mark.parametrize(
        "xyxy",
        [
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([10.0, 20.0, 30.0, 50.0]),
            np.array([100.0, 200.0, 150.0, 210.0]),
            np.array([-10.0, -20.0, 10.0, 20.0]),
            np.array([0.0, 0.0, 0.01, 0.01]),
            np.array([0.0, 0.0, 1000.0, 500.0]),
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
