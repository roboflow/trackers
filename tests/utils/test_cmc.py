# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from trackers.utils.cmc import CMC, CMCConfig

CMC_METHODS = ["sparseOptFlow", "orb", "sift", "ecc"]
TINY_FRAME_SHAPES = [(1, 1), (1, 3), (3, 1)]


@pytest.mark.parametrize("method", CMC_METHODS)
@pytest.mark.parametrize("frame_shape", TINY_FRAME_SHAPES)
def test_cmc_downscale_tiny_frame_first_call_returns_identity(
    method: Literal["sparseOptFlow", "orb", "sift", "ecc"],
    frame_shape: tuple[int, int],
) -> None:
    """Test that the first tiny frame still initializes CMC with identity."""
    cfg = CMCConfig(method=method, downscale=2)
    cmc = CMC(cfg)
    frame = np.zeros((*frame_shape, 3), dtype=np.uint8)

    H = cmc.estimate(frame)

    assert H.shape == (2, 3)
    np.testing.assert_array_equal(H, np.eye(2, 3, dtype=np.float32))


@pytest.mark.parametrize("method", CMC_METHODS)
@pytest.mark.parametrize("frame_shape", TINY_FRAME_SHAPES)
def test_cmc_downscale_tiny_frame_second_call_returns_identity(
    method: Literal["sparseOptFlow", "orb", "sift", "ecc"],
    frame_shape: tuple[int, int],
) -> None:
    """Test that repeating the same tiny frame still returns identity."""
    cfg = CMCConfig(method=method, downscale=2)
    cmc = CMC(cfg)
    frame = np.zeros((*frame_shape, 3), dtype=np.uint8)

    cmc.estimate(frame)
    H = cmc.estimate(frame)

    assert H.shape == (2, 3)
    np.testing.assert_array_equal(H, np.eye(2, 3, dtype=np.float32))


@pytest.mark.parametrize("method", CMC_METHODS)
def test_cmc_recovers_after_tiny_frame_first_call_returns_identity(
    method: Literal["sparseOptFlow", "orb", "sift", "ecc"],
) -> None:
    """Test that the first normal frame still initializes CMC with identity."""
    cfg = CMCConfig(method=method, downscale=2)
    cmc = CMC(cfg)

    normal_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    normal_frame[16:48, 16:48] = 255

    H = cmc.estimate(normal_frame)

    assert H.shape == (2, 3)
    np.testing.assert_array_equal(H, np.eye(2, 3, dtype=np.float32))


@pytest.mark.parametrize("method", CMC_METHODS)
@pytest.mark.parametrize("frame_shape", TINY_FRAME_SHAPES)
def test_cmc_recovers_after_tiny_frame_tiny_call_returns_identity(
    method: Literal["sparseOptFlow", "orb", "sift", "ecc"],
    frame_shape: tuple[int, int],
) -> None:
    """Test that a tiny frame after initialization still returns identity."""
    cfg = CMCConfig(method=method, downscale=2)
    cmc = CMC(cfg)

    normal_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    normal_frame[16:48, 16:48] = 255
    tiny_frame = np.zeros((*frame_shape, 3), dtype=np.uint8)

    cmc.estimate(normal_frame)
    H = cmc.estimate(tiny_frame)

    assert H.shape == (2, 3)
    np.testing.assert_array_equal(H, np.eye(2, 3, dtype=np.float32))


@pytest.mark.parametrize("method", CMC_METHODS)
@pytest.mark.parametrize("frame_shape", TINY_FRAME_SHAPES)
def test_cmc_recovers_after_tiny_frame_followup_call_returns_identity(
    method: Literal["sparseOptFlow", "orb", "sift", "ecc"],
    frame_shape: tuple[int, int],
) -> None:
    """Test that a normal frame after a tiny frame still returns identity."""
    cfg = CMCConfig(method=method, downscale=2)
    cmc = CMC(cfg)

    normal_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    normal_frame[16:48, 16:48] = 255
    tiny_frame = np.zeros((*frame_shape, 3), dtype=np.uint8)

    cmc.estimate(normal_frame)
    cmc.estimate(tiny_frame)
    H = cmc.estimate(normal_frame)

    assert H.shape == (2, 3)
    np.testing.assert_array_equal(H, np.eye(2, 3, dtype=np.float32))
