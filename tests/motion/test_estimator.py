# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from trackers.motion.estimator import MotionEstimator
from trackers.motion.transformation import (
    CoordinatesTransformation,
    HomographyTransformation,
    IdentityTransformation,
)


def _noise_frame(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def _translated_correspondences() -> tuple[np.ndarray, np.ndarray]:
    """Return point sets related by a pure translation, enough for findHomography to succeed."""
    previous = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 2.0], [2.0, 7.0]],
        dtype=np.float32,
    )
    return previous, previous + np.array([3.0, -2.0], dtype=np.float32)


def test_motion_estimator_survives_resolution_change() -> None:
    """A frame size change mid-stream returns a transformation instead of crashing.

    calcOpticalFlowPyrLK asserts both frames share the same size, so when a source renegotiates resolution between two
    consecutive frames the estimator must re-sync the reference frame rather than crash.
    """
    estimator = MotionEstimator()
    estimator.update(_noise_frame(480, 640, seed=1))
    transform = estimator.update(_noise_frame(720, 1280, seed=2))  # resolution changed

    assert isinstance(transform, CoordinatesTransformation)
    point = np.array([[100.0, 100.0]], dtype=np.float32)
    assert np.all(np.isfinite(transform.abs_to_rel(point)))


def test_motion_estimator_recovers_after_resolution_change() -> None:
    """After a resolution change the estimator keeps working on the new size."""
    estimator = MotionEstimator()
    estimator.update(_noise_frame(480, 640, seed=1))
    estimator.update(_noise_frame(720, 1280, seed=2))  # change: re-syncs
    transform = estimator.update(_noise_frame(720, 1280, seed=3))  # same new size

    assert isinstance(transform, CoordinatesTransformation)
    point = np.array([[100.0, 100.0]], dtype=np.float32)
    assert np.all(np.isfinite(transform.abs_to_rel(point)))


def test_motion_estimator_resets_frame_on_resolution_change() -> None:
    """A resolution change re-baselines instead of carrying stale-scale coordinates.

    The accumulated homography lives in the previous resolution's pixel space, so returning it after a size change would
    hand back coordinates in the wrong scale. The estimator must reset the reference frame to identity.
    """
    estimator = MotionEstimator()
    estimator.update(_noise_frame(480, 640, seed=1))
    # simulate motion accumulated in the old resolution's pixel space
    estimator._accumulated_homography = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 30.0], [0.0, 0.0, 1.0]])

    transform = estimator.update(_noise_frame(960, 1280, seed=2))  # resolution changed

    point = np.array([[0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(transform.abs_to_rel(point), point)  # re-baselined, not 50/30


def test_estimate_homography_normalizes_accumulated_scale() -> None:
    """Chained homographies are renormalized so the projective scale stays pinned at 1.

    Each `update` multiplies the accumulator by the frame-to-frame homography. Without renormalizing, the overall scale
    compounds every frame and eventually drives the accumulator into a numerically degenerate state.
    """
    estimator = MotionEstimator()
    estimator._accumulated_homography = np.eye(3) * 4.0  # scale left over from earlier chaining
    previous, current = _translated_correspondences()

    transform = estimator._estimate_homography(previous, current)

    assert isinstance(transform, HomographyTransformation)
    assert transform.homography_matrix[2, 2] == pytest.approx(1.0)
    assert estimator._accumulated_homography[2, 2] == pytest.approx(1.0)


def test_estimate_homography_rebaselines_degenerate_accumulator() -> None:
    """A degenerate accumulator re-baselines to identity instead of raising."""
    estimator = MotionEstimator()
    estimator._accumulated_homography = np.zeros((3, 3))
    previous, current = _translated_correspondences()

    transform = estimator._estimate_homography(previous, current)

    assert isinstance(transform, IdentityTransformation)
    np.testing.assert_allclose(estimator._accumulated_homography, np.eye(3))
