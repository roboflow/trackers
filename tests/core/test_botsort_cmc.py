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

from trackers.core.botsort.tracklet import BoTSORTTracklet
from trackers.utils.cmc import CMC, CMCConfig
from trackers.utils.state_representations import XYXYStateEstimator

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


def _xyxy_tracklet(bbox: np.ndarray) -> BoTSORTTracklet:
    """Build a BoTSORTTracklet whose state is XYXYStateEstimator."""
    return BoTSORTTracklet(bbox, state_estimator_class=XYXYStateEstimator)


class TestCMCEstimateAcrossMethods:
    """CMC.estimate() contract — parametrised over every supported method.

    All tests instantiate CMC via `_make_cmc(method)` (skips when the method is
    unavailable in the local OpenCV build) and feed it `_bgr_frame(...)` noise
    frames; assertions cover the documented `estimate()` return contract:
    shape (2, 3), dtype float32, identity on degenerate inputs, finite values
    on real inputs.
    """

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_output_is_2x3_float32(self, method: str) -> None:
        """estimate() must always return a (2, 3) float32 matrix."""
        cmc = _make_cmc(method)
        frame1 = _bgr_frame(seed=0)
        frame2 = _bgr_frame(seed=1)

        for frame in (frame1, frame2):
            H = cmc.estimate(frame)
            assert H.shape == (2, 3), f"[{method}] Expected shape (2,3), got {H.shape}"
            assert H.dtype == np.float32, f"[{method}] Expected float32, got {H.dtype}"

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_first_frame_returns_identity(self, method: str) -> None:
        """On the first frame there is no previous frame, so identity is returned."""
        cmc = _make_cmc(method)
        H = cmc.estimate(_bgr_frame(seed=42))
        assert _is_near_identity(H), f"[{method}] Expected identity on first frame:\n{H}"

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_none_frame_returns_identity(self, method: str) -> None:
        """Passing None as frame_bgr must return identity without raising."""
        cmc = _make_cmc(method)
        H = cmc.estimate(cast(np.ndarray, None))
        assert H.shape == (2, 3)
        assert _is_near_identity(H), f"[{method}] Expected identity for None frame:\n{H}"

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_reset_returns_identity_on_next_call(self, method: str) -> None:
        """After reset(), the very next estimate() call returns identity."""
        cmc = _make_cmc(method)
        cmc.estimate(_bgr_frame(seed=0))  # initialise
        cmc.estimate(_bgr_frame(seed=1))  # second frame

        cmc.reset()
        assert not cmc._initialized, f"[{method}] CMC must be uninitialized after reset"

        H = cmc.estimate(_bgr_frame(seed=2))
        assert _is_near_identity(H), f"[{method}] Expected identity after reset:\n{H}"

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_identical_frames_near_identity(self, method: str) -> None:
        """Two identical frames should produce near-zero translation."""
        cmc = _make_cmc(method)
        frame = _bgr_frame(seed=99)

        cmc.estimate(frame)  # first frame: stores state
        H = cmc.estimate(frame)  # same frame again

        assert np.all(np.isfinite(H)), f"[{method}] H must be finite for identical frames"
        assert abs(H[0, 2]) < 3.0 and abs(H[1, 2]) < 3.0, (
            f"[{method}] Expected near-zero translation for identical frames:\n{H}"
        )

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_different_frames_returns_finite_h(self, method: str) -> None:
        """Two clearly different frames must return a finite (2,3) matrix."""
        cmc = _make_cmc(method, downscale=1)
        frame1 = _bgr_frame(seed=0)
        frame2 = _bgr_frame(seed=1234)  # completely different content

        cmc.estimate(frame1)
        H = cmc.estimate(frame2)

        assert H.shape == (2, 3), f"[{method}] Bad shape: {H.shape}"
        assert np.all(np.isfinite(H)), f"[{method}] H must contain only finite values"

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_downscale_produces_valid_output(self, method: str) -> None:
        """downscale>1 must still return a finite (2,3) float32 transform."""
        cmc_ds1 = _make_cmc(method, downscale=1)
        cmc_ds2 = _make_cmc(method, downscale=2)

        frame1 = _bgr_frame(seed=0)
        frame2 = _bgr_frame(seed=1)

        cmc_ds1.estimate(frame1)
        H1 = cmc_ds1.estimate(frame2)

        cmc_ds2.estimate(frame1)
        H2 = cmc_ds2.estimate(frame2)

        assert H1.shape == (2, 3)
        assert H1.dtype == np.float32
        assert np.all(np.isfinite(H1))
        assert H2.shape == (2, 3)
        assert H2.dtype == np.float32
        assert np.all(np.isfinite(H2))


class TestCMCApplyBatch:
    """`CMC.apply_batch` against center-state tracklets.

    All tests build default `BoTSORTTracklet(bbox)` instances (center-state
    estimator) and verify the batch entry-point against the per-track
    `apply_cmc` contract (equivalence, multi-tracklet, no-op cases).
    """

    def test_matches_single(self) -> None:
        """Batch CMC must match per-track apply_cmc for a single track."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
        H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float32)

        single = BoTSORTTracklet(bbox)
        batched = BoTSORTTracklet(bbox)

        single.apply_cmc(H)
        CMC.apply_batch(H, [batched])

        np.testing.assert_allclose(
            single.state_estimator.kf.state,
            batched.state_estimator.kf.state,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            single.state_estimator.kf.state_covariance,
            batched.state_estimator.kf.state_covariance,
            atol=1e-6,
        )

    def test_multiple_tracklets(self) -> None:
        """Batch CMC applies the same transform to every tracklet."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
        H = np.array([[1.0, 0.0, 8.0], [0.0, 1.0, -2.0]], dtype=np.float32)

        singles = [BoTSORTTracklet(bbox) for _ in range(3)]
        batched = [BoTSORTTracklet(bbox) for _ in range(3)]

        for t in singles:
            t.apply_cmc(H)
        CMC.apply_batch(H, batched)

        for s, b in zip(singles, batched):
            np.testing.assert_allclose(s.state_estimator.kf.state, b.state_estimator.kf.state, atol=1e-6)

    def test_none_is_noop(self) -> None:
        """CMC.apply_batch with H=None must not change any tracklet state."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
        tracklet = BoTSORTTracklet(bbox)
        state_before = tracklet.state_estimator.kf.state.copy()

        CMC.apply_batch(None, [tracklet])

        np.testing.assert_array_equal(tracklet.state_estimator.kf.state, state_before)

    def test_empty_list_is_noop(self) -> None:
        """CMC.apply_batch with an empty list must not raise."""
        H = np.eye(2, 3, dtype=np.float32)
        CMC.apply_batch(H, [])  # must not raise

    def test_mixed_state_list_raises(self) -> None:
        """Heterogeneous state estimator types must raise TypeError immediately."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
        H = np.eye(2, 3, dtype=np.float32)

        xcycwh_track = BoTSORTTracklet(bbox)
        xyxy_track = _xyxy_tracklet(bbox)

        with pytest.raises(TypeError, match="homogeneous"):
            CMC.apply_batch(H, [xcycwh_track, xyxy_track])


class TestCMCApplyToXYXY:
    """Direct unit tests on `CMC.warp_xyxy_corners(x1, y1, x2, y2, R, t=None)`.

    Each test passes raw 1-D NumPy arrays for the four corner channels and a
    2x2 `R` (and optional 1-D `t`), then asserts on the four return arrays.
    No tracklet, no Kalman state — just the pure helper contract.
    """

    def test_identity_translation_only(self) -> None:
        """Identity R with non-zero t must shift all corners by t."""
        x1 = np.array([10.0])
        y1 = np.array([20.0])
        x2 = np.array([50.0])
        y2 = np.array([80.0])
        R = np.eye(2)
        t = np.array([5.0, -3.0])

        nx1, ny1, nx2, ny2 = CMC.warp_xyxy_corners(x1, y1, x2, y2, R, t)

        np.testing.assert_allclose([nx1[0], ny1[0], nx2[0], ny2[0]], [15.0, 17.0, 55.0, 77.0])

    def test_translation_none_skips_offset(self) -> None:
        """When t=None, only the linear part R is applied (velocity contract)."""
        x1 = np.array([0.0])
        y1 = np.array([0.0])
        x2 = np.array([1.0])
        y2 = np.array([2.0])
        R = np.eye(2)

        nx1, ny1, nx2, ny2 = CMC.warp_xyxy_corners(x1, y1, x2, y2, R, None)

        np.testing.assert_allclose([nx1[0], ny1[0], nx2[0], ny2[0]], [0.0, 0.0, 1.0, 2.0])

    def test_90deg_rotation_recovers_axis_aligned_box(self) -> None:
        """90° rotation of [0,0,2,4] gives axis-aligned enclosing [-4,0,0,2]."""
        x1 = np.array([0.0])
        y1 = np.array([0.0])
        x2 = np.array([2.0])
        y2 = np.array([4.0])
        R = np.array([[0.0, -1.0], [1.0, 0.0]])  # +90° rotation

        nx1, ny1, nx2, ny2 = CMC.warp_xyxy_corners(x1, y1, x2, y2, R)

        # Original corners (0,0),(2,0),(2,4),(0,4) → rotated (0,0),(0,2),(-4,2),(-4,0)
        np.testing.assert_allclose([nx1[0], ny1[0], nx2[0], ny2[0]], [-4.0, 0.0, 0.0, 2.0])

    def test_reflection_y_axis_swaps_x_endpoints(self) -> None:
        """Reflection across y-axis: x→-x, output preserves min<max ordering."""
        x1 = np.array([10.0])
        y1 = np.array([20.0])
        x2 = np.array([50.0])
        y2 = np.array([80.0])
        R = np.array([[-1.0, 0.0], [0.0, 1.0]])  # reflect across y-axis

        nx1, ny1, nx2, ny2 = CMC.warp_xyxy_corners(x1, y1, x2, y2, R)

        # Original x-range [10, 50] reflected → [-50, -10]; y unchanged
        assert nx1[0] < nx2[0], f"x ordering broken: {nx1[0]} >= {nx2[0]}"
        assert ny1[0] < ny2[0], f"y ordering broken: {ny1[0]} >= {ny2[0]}"
        np.testing.assert_allclose([nx1[0], ny1[0], nx2[0], ny2[0]], [-50.0, 20.0, -10.0, 80.0])

    def test_zero_size_box_yields_zero_size_output(self) -> None:
        """A degenerate (point) box stays a point under any affine transform."""
        x1 = np.array([5.0])
        y1 = np.array([5.0])
        x2 = np.array([5.0])
        y2 = np.array([5.0])
        R = np.array([[0.0, -1.0], [1.0, 0.0]])
        t = np.array([1.0, 2.0])

        nx1, ny1, nx2, ny2 = CMC.warp_xyxy_corners(x1, y1, x2, y2, R, t)

        assert nx1[0] == nx2[0] and ny1[0] == ny2[0]

    def test_negative_coordinates_preserved(self) -> None:
        """Negative input coordinates must transform without sign-related artifacts."""
        x1 = np.array([-50.0])
        y1 = np.array([-30.0])
        x2 = np.array([-10.0])
        y2 = np.array([20.0])
        R = np.eye(2)
        t = np.array([100.0, 100.0])

        nx1, ny1, nx2, ny2 = CMC.warp_xyxy_corners(x1, y1, x2, y2, R, t)

        np.testing.assert_allclose([nx1[0], ny1[0], nx2[0], ny2[0]], [50.0, 70.0, 90.0, 120.0])

    def test_batch_input_matches_per_element_loop(self) -> None:
        """Vectorised batch call must equal applying the helper per element."""
        x1 = np.array([0.0, 10.0, -5.0])
        y1 = np.array([0.0, 20.0, -2.0])
        x2 = np.array([2.0, 30.0, 5.0])
        y2 = np.array([4.0, 40.0, 7.0])
        R = np.array([[0.0, -1.0], [1.0, 0.0]])
        t = np.array([1.0, 2.0])

        bnx1, bny1, bnx2, bny2 = CMC.warp_xyxy_corners(x1, y1, x2, y2, R, t)
        expected = [
            CMC.warp_xyxy_corners(
                np.array([x1[i]]),
                np.array([y1[i]]),
                np.array([x2[i]]),
                np.array([y2[i]]),
                R,
                t,
            )
            for i in range(3)
        ]
        np.testing.assert_allclose(bnx1, [e[0][0] for e in expected])
        np.testing.assert_allclose(bny1, [e[1][0] for e in expected])
        np.testing.assert_allclose(bnx2, [e[2][0] for e in expected])
        np.testing.assert_allclose(bny2, [e[3][0] for e in expected])


class TestXYXYCovarianceUpdate:
    """Covariance (P) update contract for XYXY-state tracklets under CMC.

    All tests build an `_xyxy_tracklet([10, 20, 50, 80])`, snapshot
    `kf.state_covariance`, construct an H matrix from a chosen 2x2 R block, apply CMC
    (single via `tracklet.apply_cmc`, batch via `tracker.apply_cmc_batch`),
    then assert P either propagates as `A @ P @ A.T` (axis-aligned R) or
    is left untouched (cross-axis R).
    """

    @pytest.mark.parametrize(
        "R_off",
        [
            np.array([[1.0, 0.0], [0.0, 1.0]]),  # identity (axis-aligned)
            np.array([[1.0, 0.0], [0.0, -1.0]]),  # axis-flip (axis-aligned)
            np.array([[2.0, 0.0], [0.0, 0.5]]),  # axis-aligned scale
        ],
    )
    def test_axis_aligned_updates_p(self, R_off: np.ndarray) -> None:
        """Axis-aligned R must propagate into P via A @ P @ A.T."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
        H = np.zeros((2, 3), dtype=np.float32)
        H[:2, :2] = R_off
        tracklet = _xyxy_tracklet(bbox)
        P_before = tracklet.state_estimator.kf.state_covariance.copy()

        tracklet.apply_cmc(H)

        A = np.eye(8, dtype=np.float64)
        A[0:2, 0:2] = R_off
        A[2:4, 2:4] = R_off
        A[4:6, 4:6] = R_off
        A[6:8, 6:8] = R_off
        expected_P = A @ P_before @ A.T
        np.testing.assert_allclose(tracklet.state_estimator.kf.state_covariance, expected_P, atol=1e-9)

    @pytest.mark.parametrize(
        "R_cross",
        [
            np.array([[0.0, -1.0], [1.0, 0.0]]),  # 90° rotation
            np.array([[1.0, 0.5], [0.0, 1.0]]),  # x-shear
            np.array([[1.0, 0.0], [0.5, 1.0]]),  # y-shear
            np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]]),  # small rot
        ],
    )
    def test_cross_axis_freezes_p(self, R_cross: np.ndarray) -> None:
        """Cross-axis R must leave P untouched (A=None branch)."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
        H = np.zeros((2, 3), dtype=np.float32)
        H[:2, :2] = R_cross
        tracklet = _xyxy_tracklet(bbox)
        P_before = tracklet.state_estimator.kf.state_covariance.copy()

        tracklet.apply_cmc(H)

        np.testing.assert_array_equal(tracklet.state_estimator.kf.state_covariance, P_before)

    @pytest.mark.parametrize(
        "R_cross",
        [
            np.array([[0.0, -1.0], [1.0, 0.0]]),
            np.array([[1.0, 0.5], [0.0, 1.0]]),
        ],
    )
    def test_cross_axis_freezes_p_batch_path(self, R_cross: np.ndarray) -> None:
        """Batch path must also freeze P when R has cross-axis terms."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
        H = np.zeros((2, 3), dtype=np.float32)
        H[:2, :2] = R_cross
        tracklet = _xyxy_tracklet(bbox)
        P_before = tracklet.state_estimator.kf.state_covariance.copy()

        CMC.apply_batch(H, [tracklet])

        np.testing.assert_array_equal(tracklet.state_estimator.kf.state_covariance, P_before)

    def test_axis_aligned_updates_p_batch_path(self) -> None:
        """Batch path must update P when R is axis-aligned."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
        R = np.array([[1.0, 0.0], [0.0, -1.0]])
        H = np.zeros((2, 3), dtype=np.float32)
        H[:2, :2] = R
        tracklet = _xyxy_tracklet(bbox)
        P_before = tracklet.state_estimator.kf.state_covariance.copy()

        CMC.apply_batch(H, [tracklet])

        A = np.eye(8, dtype=np.float64)
        A[0:2, 0:2] = R
        A[2:4, 2:4] = R
        A[4:6, 4:6] = R
        A[6:8, 6:8] = R
        expected_P = A @ P_before @ A.T
        np.testing.assert_allclose(tracklet.state_estimator.kf.state_covariance, expected_P, atol=1e-9)


class TestXYXYAxisAlignedTolerance:
    """Boundary tests on the `atol=1e-6` axis-aligned classifier in CMC.

    Both tests build the same `_xyxy_tracklet` and an R whose cross-axis
    residual sits just below or just above 1e-6, then verify which branch
    (P-update vs P-freeze) `tracklet.apply_cmc(H)` selected by inspecting
    the post-call `kf.state_covariance`.
    """

    def test_residual_below_atol_treated_as_axis_aligned(self) -> None:
        """Cross-axis residual below 1e-6 takes axis-aligned (P-update) branch."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
        R = np.array([[1.0, 5e-7], [5e-7, 1.0]])  # within atol=1e-6
        H = np.zeros((2, 3), dtype=np.float32)
        H[:2, :2] = R
        tracklet = _xyxy_tracklet(bbox)
        P_before = tracklet.state_estimator.kf.state_covariance.copy()

        tracklet.apply_cmc(H)

        # Branch was taken → P was multiplied by A, so it must differ slightly
        # from the original (or remain near-equal but not strictly identical).
        # Stronger contract: P matches A @ P_before @ A.T (not the freeze case).
        A = np.eye(8, dtype=np.float64)
        A[0:2, 0:2] = R
        A[2:4, 2:4] = R
        A[4:6, 4:6] = R
        A[6:8, 6:8] = R
        expected_P = A @ P_before @ A.T
        np.testing.assert_allclose(tracklet.state_estimator.kf.state_covariance, expected_P, atol=1e-9)

    def test_residual_above_atol_treated_as_cross_axis(self) -> None:
        """Cross-axis residual above 1e-6 must take the freeze branch."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
        R = np.array([[1.0, 1e-3], [0.0, 1.0]])  # above atol=1e-6
        H = np.zeros((2, 3), dtype=np.float32)
        H[:2, :2] = R
        tracklet = _xyxy_tracklet(bbox)
        P_before = tracklet.state_estimator.kf.state_covariance.copy()

        tracklet.apply_cmc(H)

        np.testing.assert_array_equal(tracklet.state_estimator.kf.state_covariance, P_before)


@pytest.mark.parametrize(
    "R",
    [
        np.array([[0.0, -1.0], [1.0, 0.0]]),  # rotation
        np.array([[-1.0, 0.0], [0.0, 1.0]]),  # reflection
        np.array([[1.0, 0.5], [0.0, 1.0]]),  # shear
        np.array([[2.0, 0.0], [0.0, 0.5]]),  # axis-aligned scale
    ],
)
def test_xyxy_batch_matches_single_under_non_translation_R(R: np.ndarray) -> None:
    """CMC.apply_batch and per-track apply_cmc must agree for any 2x2 R on XYXY state."""
    bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
    H = np.array([[R[0, 0], R[0, 1], 7.0], [R[1, 0], R[1, 1], -2.0]], dtype=np.float32)
    single = _xyxy_tracklet(bbox)
    batched = _xyxy_tracklet(bbox)

    single.apply_cmc(H)
    CMC.apply_batch(H, [batched])

    np.testing.assert_allclose(single.state_estimator.kf.state, batched.state_estimator.kf.state, atol=1e-9)
    np.testing.assert_allclose(
        single.state_estimator.kf.state_covariance, batched.state_estimator.kf.state_covariance, atol=1e-9
    )


def test_xyxy_apply_cmc_90deg_rotation_state_is_axis_aligned() -> None:
    """After 90° rotation, the post-CMC XYXY state must remain a valid box."""
    bbox = np.array([0.0, 0.0, 2.0, 4.0], dtype=np.float32)
    R = np.array([[0.0, -1.0], [1.0, 0.0]])  # +90° rotation about origin
    H = np.zeros((2, 3), dtype=np.float32)
    H[:2, :2] = R
    tracklet = _xyxy_tracklet(bbox)

    tracklet.apply_cmc(H)

    x = tracklet.state_estimator.kf.state.reshape(-1)
    # Original corners (0,0),(2,0),(2,4),(0,4) rotate to (0,0),(0,2),(-4,2),(-4,0)
    # → enclosing box [-4, 0, 0, 2]
    np.testing.assert_allclose([x[0], x[1], x[2], x[3]], [-4.0, 0.0, 0.0, 2.0], atol=1e-9)
    assert x[0] < x[2], "x1 must remain < x2"
    assert x[1] < x[3], "y1 must remain < y2"


def test_xyxy_velocity_rotates_without_translation() -> None:
    """Velocity entries (state[4:8]) must transform via R only, not R+t."""
    bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
    tracklet = _xyxy_tracklet(bbox)
    # Inject a known non-zero velocity quad: vx1=1, vy1=0, vx2=0, vy2=1
    tracklet.state_estimator.kf.state[4, 0] = 1.0
    tracklet.state_estimator.kf.state[5, 0] = 0.0
    tracklet.state_estimator.kf.state[6, 0] = 0.0
    tracklet.state_estimator.kf.state[7, 0] = 1.0
    R = np.array([[0.0, -1.0], [1.0, 0.0]])  # +90° rotation
    H = np.zeros((2, 3), dtype=np.float32)
    H[:2, :2] = R
    H[:2, 2] = np.array([100.0, 200.0])  # large t — must NOT bleed into velocity

    tracklet.apply_cmc(H)

    # Velocity corners (1,0),(0,0),(0,1),(1,1) rotate to (0,1),(0,0),(-1,0),(-1,1)
    # → enclosing velocity box [-1, 0, 0, 1] (no translation applied)
    v = tracklet.state_estimator.kf.state.reshape(-1)[4:8]
    np.testing.assert_allclose(v, [-1.0, 0.0, 0.0, 1.0], atol=1e-9)


class TestCMCApplyBatchAdversarial:
    """Adversarial H inputs to CMC.apply_batch.

    Covers wrong-shape H (should raise) and non-finite H values
    (should propagate to state without raising).
    """

    def test_wrong_shape_h_raises(self) -> None:
        """H with (2,2) shape must raise when the translation column is accessed."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
        track = BoTSORTTracklet(bbox)
        H_bad = np.eye(2, 2, dtype=np.float32)

        with pytest.raises((IndexError, ValueError)):
            CMC.apply_batch(H_bad, [track])

    @pytest.mark.parametrize(
        "fill_val",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "inf", "-inf"],
    )
    def test_non_finite_h_propagates_to_state(self, fill_val: float) -> None:
        """Non-finite H values must propagate into track state without raising."""
        bbox = np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32)
        track = BoTSORTTracklet(bbox)
        H_bad = np.full((2, 3), fill_val, dtype=np.float32)

        CMC.apply_batch(H_bad, [track])  # must not raise

        state = track.state_estimator.kf.state.reshape(-1)
        assert not np.all(np.isfinite(state)), "Non-finite H must propagate to track state"
