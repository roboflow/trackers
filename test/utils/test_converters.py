# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from trackers.utils.converters import xcycsr_to_xyxy, xyxy_to_xcycsr


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
def test_xyxy_to_xcycsr_single_box(xyxy: np.ndarray, expected: np.ndarray) -> None:
    result = xyxy_to_xcycsr(xyxy)
    assert result.shape == (4,)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_xyxy_to_xcycsr_zero_height_returns_finite() -> None:
    xyxy = np.array([0.0, 0.0, 10.0, 0.0])
    result = xyxy_to_xcycsr(xyxy)
    assert np.isfinite(result).all()
    assert result[2] == 0.0


def test_xyxy_to_xcycsr_single_row_2d_returns_2d() -> None:
    xyxy = np.array([[0.0, 0.0, 1.0, 1.0]])
    result = xyxy_to_xcycsr(xyxy)
    assert result.shape == (1, 4)
    np.testing.assert_array_almost_equal(
        result[0], np.array([0.5, 0.5, 1.0, 1.0]), decimal=5
    )


def test_xyxy_to_xcycsr_empty_batch_returns_empty() -> None:
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
def test_xcycsr_to_xyxy_single_box(xcycsr: np.ndarray, expected: np.ndarray) -> None:
    result = xcycsr_to_xyxy(xcycsr)
    assert result.shape == (4,)
    np.testing.assert_array_almost_equal(result, expected, decimal=4)


def test_xcycsr_to_xyxy_zero_scale_produces_nan() -> None:
    xcycsr = np.array([10.0, 20.0, 0.0, 1.0])
    result = xcycsr_to_xyxy(xcycsr)
    assert result[0] == result[2] == 10.0
    assert np.isnan(result[1]) and np.isnan(result[3])


def test_xcycsr_to_xyxy_zero_aspect_ratio_produces_nan() -> None:
    xcycsr = np.array([10.0, 20.0, 100.0, 0.0])
    result = xcycsr_to_xyxy(xcycsr)
    assert np.isnan(result).any() or np.isinf(result).any()


def test_xcycsr_to_xyxy_negative_scale_produces_nan() -> None:
    xcycsr = np.array([10.0, 20.0, -100.0, 1.0])
    result = xcycsr_to_xyxy(xcycsr)
    assert np.isnan(result).any()


def test_xcycsr_to_xyxy_single_row_2d_returns_2d() -> None:
    xcycsr = np.array([[0.5, 0.5, 1.0, 1.0]])
    result = xcycsr_to_xyxy(xcycsr)
    assert result.shape == (1, 4)
    np.testing.assert_array_almost_equal(
        result[0], np.array([0.0, 0.0, 1.0, 1.0]), decimal=5
    )


def test_xcycsr_to_xyxy_empty_batch_returns_empty() -> None:
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
def test_xyxy_xcycsr_roundtrip(xyxy: np.ndarray) -> None:
    xcycsr = xyxy_to_xcycsr(xyxy)
    recovered = xcycsr_to_xyxy(xcycsr)
    np.testing.assert_array_almost_equal(recovered, xyxy, decimal=5)


def test_xyxy_xcycsr_roundtrip_preserves_2d_shape() -> None:
    xyxy = np.array([[0.0, 0.0, 10.0, 10.0]])
    xcycsr = xyxy_to_xcycsr(xyxy)
    recovered = xcycsr_to_xyxy(xcycsr)
    assert recovered.shape == (1, 4)
    np.testing.assert_array_almost_equal(recovered, xyxy, decimal=5)
