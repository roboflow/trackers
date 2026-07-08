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


@pytest.mark.parametrize("method", ["sparseOptFlow", "orb", "sift", "ecc"])
@pytest.mark.parametrize("frame_shape", [(1, 1), (1, 3), (3, 1)])
def test_cmc_downscale_tiny_frames(
    method: Literal["sparseOptFlow", "orb", "sift", "ecc"],
    frame_shape: tuple[int, int],
) -> None:
    """Test that tiny frames below the downscale factor still produce identity transforms."""
    cfg = CMCConfig(method=method, downscale=2)
    cmc = CMC(cfg)
    frame = np.zeros((*frame_shape, 3), dtype=np.uint8)
    expected = np.eye(2, 3, dtype=np.float32)

    H1 = cmc.estimate(frame)
    assert H1.shape == (2, 3)
    np.testing.assert_array_equal(H1, expected)

    H2 = cmc.estimate(frame)
    assert H2.shape == (2, 3)
    np.testing.assert_array_equal(H2, expected)


@pytest.mark.parametrize("method", ["sparseOptFlow", "orb", "sift", "ecc"])
def test_cmc_recovers_after_tiny_frame(method: Literal["sparseOptFlow", "orb", "sift", "ecc"]) -> None:
    """Test that a tiny frame does not poison the next normal estimate."""
    cfg = CMCConfig(method=method, downscale=2)
    cmc = CMC(cfg)
    expected = np.eye(2, 3, dtype=np.float32)

    normal_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    normal_frame[16:48, 16:48] = 255
    tiny_frame = np.zeros((1, 1, 3), dtype=np.uint8)

    H1 = cmc.estimate(normal_frame)
    assert H1.shape == (2, 3)
    np.testing.assert_array_equal(H1, expected)

    H2 = cmc.estimate(tiny_frame)
    assert H2.shape == (2, 3)
    np.testing.assert_array_equal(H2, expected)

    H3 = cmc.estimate(normal_frame)
    assert H3.shape == (2, 3)
    np.testing.assert_array_equal(H3, expected)
