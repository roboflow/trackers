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

    affine_mtx = cmc.estimate(frame)

    assert affine_mtx.shape == (2, 3)
    np.testing.assert_array_equal(affine_mtx, np.eye(2, 3, dtype=np.float32))


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
    affine_mtx = cmc.estimate(frame)

    assert affine_mtx.shape == (2, 3)
    np.testing.assert_array_equal(affine_mtx, np.eye(2, 3, dtype=np.float32))


@pytest.mark.parametrize("method", CMC_METHODS)
def test_cmc_recovers_after_tiny_frame_first_call_returns_identity(
    method: Literal["sparseOptFlow", "orb", "sift", "ecc"],
) -> None:
    """Test that the first normal frame still initializes CMC with identity."""
    cfg = CMCConfig(method=method, downscale=2)
    cmc = CMC(cfg)

    normal_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    normal_frame[16:48, 16:48] = 255

    affine_mtx = cmc.estimate(normal_frame)

    assert affine_mtx.shape == (2, 3)
    np.testing.assert_array_equal(affine_mtx, np.eye(2, 3, dtype=np.float32))


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
    affine_mtx = cmc.estimate(tiny_frame)

    assert affine_mtx.shape == (2, 3)
    np.testing.assert_array_equal(affine_mtx, np.eye(2, 3, dtype=np.float32))


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
    affine_mtx = cmc.estimate(normal_frame)

    assert affine_mtx.shape == (2, 3)
    np.testing.assert_array_equal(affine_mtx, np.eye(2, 3, dtype=np.float32))


@pytest.mark.parametrize("method", CMC_METHODS)
def test_cmc_survives_resolution_change(
    method: Literal["sparseOptFlow", "orb", "sift", "ecc"],
) -> None:
    """No CMC method crashes when the frame size changes mid-stream.

    A source that renegotiates resolution (e.g. an RTSP camera after a
    reconnect, or switching clips without a reset) feeds consecutive frames of
    different sizes. sparseOptFlow feeds both into calcOpticalFlowPyrLK, which
    asserts they share the same size, so it used to crash; the others already
    coped. Every method must return a finite affine matrix.
    """
    rng = np.random.default_rng(0)
    small = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    large = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    cmc = CMC(CMCConfig(method=method, downscale=2))
    cmc.estimate(small)
    affine_mtx = cmc.estimate(large)  # resolution changed between the two frames

    assert affine_mtx.shape == (2, 3)
    assert np.all(np.isfinite(affine_mtx))


def test_cmc_sparse_optflow_returns_identity_on_resolution_change() -> None:
    """sparseOptFlow re-inits and returns identity when the frame size changes.

    This is the guarded path: the cached previous frame no longer matches the
    new frame size, so the optical-flow step is skipped for that frame.
    """
    rng = np.random.default_rng(0)
    small = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    large = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    cmc = CMC(CMCConfig(method="sparseOptFlow", downscale=2))
    cmc.estimate(small)
    affine_mtx = cmc.estimate(large)

    np.testing.assert_array_equal(affine_mtx, np.eye(2, 3, dtype=np.float32))


def test_cmc_sparse_optflow_preserves_nonzero_status_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sparse optical flow treats every nonzero status as a good match."""
    points = np.array(
        [[[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[2.0, 2.0]]],
        dtype=np.float32,
    )
    matched = points + np.array([2.0, 3.0], dtype=np.float32)
    status = np.array([[255], [2], [3], [4], [5]], dtype=np.uint8)
    expected_affine = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype=np.float64)

    monkeypatch.setattr(
        "trackers.utils.cmc.cv2.goodFeaturesToTrack",
        lambda frame, mask=None, **kwargs: points.copy(),
    )
    monkeypatch.setattr(
        "trackers.utils.cmc.cv2.calcOpticalFlowPyrLK",
        lambda previous, current, previous_points, next_points: (matched, status, None),
    )
    monkeypatch.setattr(
        "trackers.utils.cmc.cv2.estimateAffinePartial2D",
        lambda previous_points, current_points, method: (expected_affine, None),
    )

    cmc = CMC(CMCConfig(method="sparseOptFlow", downscale=1))
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    cmc.estimate(frame)
    affine_mtx = cmc.estimate(frame)

    np.testing.assert_array_equal(affine_mtx, expected_affine.astype(np.float32))


@pytest.mark.parametrize("method", CMC_METHODS)
def test_cmc_recovers_after_resolution_change(
    method: Literal["sparseOptFlow", "orb", "sift", "ecc"],
) -> None:
    """After a resolution change CMC keeps estimating on the new size."""
    rng = np.random.default_rng(1)
    small = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    large_a = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    large_b = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    cmc = CMC(CMCConfig(method=method, downscale=2))
    cmc.estimate(small)
    cmc.estimate(large_a)  # change: re-syncs on the new size
    affine_mtx = cmc.estimate(large_b)  # same new size again: no crash, finite output

    assert affine_mtx.shape == (2, 3)
    assert np.all(np.isfinite(affine_mtx))
