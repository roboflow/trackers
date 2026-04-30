# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for CMC (Camera Motion Compensation) — all four methods."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import pytest

from trackers.core.botsort.cmc import CMC, CMCConfig
from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.botsort.tracklet import BoTSORTTracklet

# All supported CMC methods.
ALL_METHODS = ["sparseOptFlow", "orb", "sift", "ecc"]


def _bgr_frame(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    """Random-noise BGR frame with dense texture suitable for feature tracking."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _is_near_identity(H: np.ndarray, atol: float = 1e-3) -> bool:
    return bool(np.allclose(H, np.eye(2, 3, dtype=np.float32), atol=atol))


def _make_cmc(method: str, **kwargs: object) -> CMC:
    """Instantiate CMC, skipping the test if the method is unavailable in OpenCV."""
    try:
        return CMC(CMCConfig(method=method, **kwargs))  # type: ignore[arg-type]
    except (cv2.error, AttributeError) as exc:
        pytest.skip(f"CMC method {method!r} unavailable in this OpenCV build: {exc}")
    raise AssertionError("unreachable")


# ==========================================================================
# Output shape and dtype
# ==========================================================================


@pytest.mark.parametrize("method", ALL_METHODS)
def test_cmc_output_is_2x3_float32(method: str) -> None:
    """estimate() must always return a (2, 3) float32 matrix."""
    cmc = _make_cmc(method)
    frame1 = _bgr_frame(seed=0)
    frame2 = _bgr_frame(seed=1)

    for frame in (frame1, frame2):
        H = cmc.estimate(frame)
        assert H.shape == (2, 3), f"[{method}] Expected shape (2,3), got {H.shape}"
        assert H.dtype == np.float32, f"[{method}] Expected float32, got {H.dtype}"


# ==========================================================================
# First-frame returns identity
# ==========================================================================


@pytest.mark.parametrize("method", ALL_METHODS)
def test_cmc_first_frame_returns_identity(method: str) -> None:
    """On the first frame there is no previous frame, so identity is returned."""
    cmc = _make_cmc(method)
    H = cmc.estimate(_bgr_frame(seed=42))
    assert _is_near_identity(H), f"[{method}] Expected identity on first frame:\n{H}"


# ==========================================================================
# None frame returns identity
# ==========================================================================


@pytest.mark.parametrize("method", ALL_METHODS)
def test_cmc_none_frame_returns_identity(method: str) -> None:
    """Passing None as frame_bgr must return identity without raising."""
    cmc = _make_cmc(method)
    H = cmc.estimate(cast(np.ndarray, None))
    assert H.shape == (2, 3)
    assert _is_near_identity(H), f"[{method}] Expected identity for None frame:\n{H}"


# ==========================================================================
# Reset restores first-frame behaviour
# ==========================================================================


@pytest.mark.parametrize("method", ALL_METHODS)
def test_cmc_reset_returns_identity_on_next_call(method: str) -> None:
    """After reset(), the very next estimate() call returns identity."""
    cmc = _make_cmc(method)
    cmc.estimate(_bgr_frame(seed=0))  # initialise
    cmc.estimate(_bgr_frame(seed=1))  # second frame

    cmc.reset()
    assert not cmc._initialized, f"[{method}] CMC must be uninitialized after reset"

    H = cmc.estimate(_bgr_frame(seed=2))
    assert _is_near_identity(H), f"[{method}] Expected identity after reset:\n{H}"


# ==========================================================================
# Identical frames yield near-identity transform
# ==========================================================================


@pytest.mark.parametrize("method", ALL_METHODS)
def test_cmc_identical_frames_near_identity(method: str) -> None:
    """Two identical frames should produce near-zero translation."""
    cmc = _make_cmc(method)
    frame = _bgr_frame(seed=99)

    cmc.estimate(frame)  # first frame: stores state
    H = cmc.estimate(frame)  # same frame again

    assert np.all(np.isfinite(H)), f"[{method}] H must be finite for identical frames"
    assert abs(H[0, 2]) < 3.0 and abs(H[1, 2]) < 3.0, (
        f"[{method}] Expected near-zero translation for identical frames:\n{H}"
    )


# ==========================================================================
# Non-trivial motion is handled gracefully
# ==========================================================================


@pytest.mark.parametrize("method", ALL_METHODS)
def test_cmc_different_frames_returns_finite_h(method: str) -> None:
    """Two clearly different frames must return a finite (2,3) matrix."""
    cmc = _make_cmc(method, downscale=1)
    frame1 = _bgr_frame(seed=0)
    frame2 = _bgr_frame(seed=1234)  # completely different content

    cmc.estimate(frame1)
    H = cmc.estimate(frame2)

    assert H.shape == (2, 3), f"[{method}] Bad shape: {H.shape}"
    assert np.all(np.isfinite(H)), f"[{method}] H must contain only finite values"


# ==========================================================================
# Downscale: still returns valid transforms
# ==========================================================================


@pytest.mark.parametrize("method", ALL_METHODS)
def test_cmc_downscale_produces_valid_output(method: str) -> None:
    """downscale>1 must still return a finite (2,3) float32 transform."""
    cmc_ds1 = _make_cmc(method, downscale=1)
    cmc_ds2 = _make_cmc(method, downscale=2)

    frame1 = _bgr_frame(seed=0)
    frame2 = _bgr_frame(seed=1)

    cmc_ds1.estimate(frame1)
    H1 = cmc_ds1.estimate(frame2)

    cmc_ds2.estimate(frame1)
    H2 = cmc_ds2.estimate(frame2)

    for label, H in [("downscale=1", H1), ("downscale=2", H2)]:
        assert H.shape == (2, 3), f"[{method}/{label}] Bad shape: {H.shape}"
        assert H.dtype == np.float32, f"[{method}/{label}] Bad dtype: {H.dtype}"
        assert np.all(np.isfinite(H)), (
            f"[{method}/{label}] H must contain only finite values"
        )


# ==========================================================================
# BoTSORTTracker.apply_cmc_batch()
# ==========================================================================


def test_botsort_tracker_apply_cmc_batch_matches_single() -> None:
    """Batch CMC must match per-track apply_cmc for a single track."""
    bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
    H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float32)

    single = BoTSORTTracklet(bbox)
    batched = BoTSORTTracklet(bbox)
    tracker = BoTSORTTracker()
    tracker.tracks = [batched]

    single.apply_cmc(H)
    tracker.apply_cmc_batch(H)

    np.testing.assert_allclose(
        single.state_estimator.kf.x,
        batched.state_estimator.kf.x,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        single.state_estimator.kf.P,
        batched.state_estimator.kf.P,
        atol=1e-6,
    )


def test_botsort_tracker_apply_cmc_batch_multiple_tracklets() -> None:
    """Batch CMC applies the same transform to every tracklet."""
    bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
    H = np.array([[1.0, 0.0, 8.0], [0.0, 1.0, -2.0]], dtype=np.float32)

    singles = [BoTSORTTracklet(bbox) for _ in range(3)]
    batched = [BoTSORTTracklet(bbox) for _ in range(3)]
    tracker = BoTSORTTracker()
    tracker.tracks = batched

    for t in singles:
        t.apply_cmc(H)
    tracker.apply_cmc_batch(H)

    for s, b in zip(singles, batched):
        np.testing.assert_allclose(
            s.state_estimator.kf.x, b.state_estimator.kf.x, atol=1e-6
        )


def test_botsort_tracker_apply_cmc_batch_none_is_noop() -> None:
    """apply_cmc_batch with H=None must not change any tracklet state."""
    bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
    tracklet = BoTSORTTracklet(bbox)
    tracker = BoTSORTTracker()
    tracker.tracks = [tracklet]
    state_before = tracklet.state_estimator.kf.x.copy()

    tracker.apply_cmc_batch(None)

    np.testing.assert_array_equal(tracklet.state_estimator.kf.x, state_before)


def test_botsort_tracker_apply_cmc_batch_empty_list_is_noop() -> None:
    """apply_cmc_batch with an empty list must not raise."""
    H = np.eye(2, 3, dtype=np.float32)
    tracker = BoTSORTTracker()
    tracker.tracks = []
    tracker.apply_cmc_batch(H)  # must not raise
