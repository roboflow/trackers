# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for conditional camera motion compensation (CMC).

Checks the ORB candidate-match threshold and XCYCWH state/covariance transformation from the original McByte++, together
with invalid-transform and per-track rejection safeguards in the Trackers adaptation.
"""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from trackers.core.mcbyteplusplus.tracklet import McBytePlusPlusTracklet
from trackers.utils.conditional_cmc import CMCMethod, ConditionalCMC, ConditionalCMCConfig
from trackers.utils.state_representations import BaseStateEstimator, XCYCSRStateEstimator, XYXYStateEstimator

METHODS: list[CMCMethod] = ["orb", "sift", "sparseOptFlow", "ecc"]


@pytest.fixture
def image() -> np.ndarray:
    """Textured, repeatable background with enough features for every method."""
    rng = np.random.default_rng(12)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(300):
        x, y = rng.integers([10, 10], [630, 470])
        color = tuple(int(value) for value in rng.integers(40, 256, 3))
        cv2.circle(frame, (int(x), int(y)), int(rng.integers(3, 12)), color, -1)
    return frame


def translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Move a known background without loading a model or a dataset."""
    return cv2.warpAffine(
        image,
        np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]]),
        (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_REFLECT_101,
    )


@pytest.mark.parametrize("method", METHODS)
def test_estimates_consecutive_motion_in_original_image_pixels(method: CMCMethod, image: np.ndarray) -> None:
    """Each method must track successive image pairs and undo its downscale."""
    cmc = ConditionalCMC(ConditionalCMCConfig(method=method, downscale=2))
    np.testing.assert_array_equal(cmc.estimate(image), np.eye(2, 3))
    for step in (1, 2):
        affine = cmc.estimate(translate(image, step * 4, step * 2))
        assert affine.dtype == np.float32
        np.testing.assert_allclose(affine[:, :2], np.eye(2), atol=0.02)
        np.testing.assert_allclose(affine[:, 2], [4, 2], atol=0.8)


@pytest.mark.parametrize("method", METHODS)
def test_reset_missing_images_and_resolution_changes(method: CMCMethod, image: np.ndarray) -> None:
    """None defers; a new resolution or sequence must not reuse old features."""
    cmc = ConditionalCMC(ConditionalCMCConfig(method=method, downscale=2))
    cmc.estimate(image)
    # A missing image returns identity while retaining the last usable reference.
    np.testing.assert_array_equal(cmc.estimate(None), np.eye(2, 3))
    np.testing.assert_allclose(cmc.estimate(translate(image, 4, 2))[:, 2], [4, 2], atol=0.8)
    # Even resolutions that floor-divide to the same size require re-seeding.
    resized = np.pad(translate(image, 8, 4), ((0, 1), (0, 1), (0, 0)), mode="edge")
    np.testing.assert_array_equal(cmc.estimate(resized), np.eye(2, 3))
    # Start with a nonzero count so reset must actively clear it.
    cmc.frames_failed = 1
    cmc.reset()
    assert cmc.frames_failed == 0
    # Keep the resolution unchanged so automatic re-seeding cannot hide a broken reset.
    np.testing.assert_array_equal(cmc.estimate(translate(resized, 4, 2)), np.eye(2, 3))
    # The second image after reset must recover motion relative to the new reference.
    np.testing.assert_allclose(cmc.estimate(translate(resized, 8, 4))[:, 2], [4, 2], atol=0.8)


@pytest.mark.parametrize("method", METHODS)
def test_feature_poor_and_tiny_frames_allow_recovery(method: CMCMethod, image: np.ndarray) -> None:
    """Unavailable image evidence produces identity and leaves a recoverable estimator."""
    cmc = ConditionalCMC(ConditionalCMCConfig(method=method, downscale=2))
    # Seed a reference, lose features, encounter an undersized image, then seed a fresh reference.
    for frame in (image, np.zeros_like(image), np.zeros((1, 3, 3), dtype=np.uint8), image):
        np.testing.assert_array_equal(cmc.estimate(frame), np.eye(2, 3))
    # Two usable images must restore estimation of the known (4, 2)-pixel motion.
    np.testing.assert_allclose(cmc.estimate(translate(image, 4, 2))[:, 2], [4, 2], atol=0.8)


@pytest.mark.parametrize("method", METHODS)
def test_opencv_failure_clears_reference_then_recovers(
    method: CMCMethod, image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted estimate must not span the failed frame on the next call."""
    cmc = ConditionalCMC(ConditionalCMCConfig(method=method, downscale=2))
    cmc.estimate(image)
    name = {"orb": "knnMatch", "sift": "knnMatch", "sparseOptFlow": "calcOpticalFlowPyrLK", "ecc": "findTransformECC"}[
        method
    ]

    def fail(*args, **kwargs):
        raise cv2.error("synthetic estimation failure")

    # Force the method-specific OpenCV call to fail for one frame only.
    with monkeypatch.context() as scoped:
        if method in ("orb", "sift"):
            scoped.setattr(cmc, "matcher", SimpleNamespace(knnMatch=fail))
        else:
            scoped.setattr(cv2, name, fail)
        np.testing.assert_array_equal(cmc.estimate(translate(image, 4, 2)), np.eye(2, 3))
    assert cmc.frames_failed == 1
    # With OpenCV restored, the next frame seeds a new reference instead of matching stale history.
    np.testing.assert_array_equal(cmc.estimate(translate(image, 8, 4)), np.eye(2, 3))
    # Recover only the next (4, 2)-pixel step, without spanning the failed frame.
    np.testing.assert_allclose(cmc.estimate(translate(image, 12, 6))[:, 2], [4, 2], atol=0.8)


def feature_inputs(cmc: ConditionalCMC, count: int = 12) -> SimpleNamespace:
    """Inject descriptor correspondences to exercise precise failure boundaries."""
    points = [cv2.KeyPoint(float(20 + i * 3), float(40 + (i % 3) * 3), 1) for i in range(count)]
    payload = SimpleNamespace(points=points, descriptors=np.zeros((count, 32), dtype=np.uint8))
    cmc.detector = SimpleNamespace(detect=lambda image, mask: payload.points)
    cmc.extractor = SimpleNamespace(compute=lambda image, points: (payload.points, payload.descriptors))
    cmc.matcher = SimpleNamespace(
        knnMatch=lambda *args, **kwargs: [
            (cv2.DMatch(i, i, 0.0), cv2.DMatch(i, (i + 1) % count, 10.0)) for i in range(count)
        ]
    )
    return payload


def test_incomplete_knn_results_are_skipped(image: np.ndarray) -> None:
    """ORB/SIFT share the matching path; one neighbor cannot support its two-neighbor ratio test."""
    cmc = ConditionalCMC()
    payload = feature_inputs(cmc)
    cmc.estimate(image)
    payload.descriptors = np.ones_like(payload.descriptors)
    cmc.matcher = SimpleNamespace(knnMatch=lambda *args, **kwargs: [(cv2.DMatch(0, 0, 0.0),)])
    np.testing.assert_array_equal(cmc.estimate(image), np.eye(2, 3))
    np.testing.assert_array_equal(cmc._prev_desc, payload.descriptors)


@pytest.mark.parametrize(
    "minimum_matches,count,expected_fits",
    [
        pytest.param(None, 11, 0, id="default-below"),
        pytest.param(None, 12, 1, id="default-at-minimum"),
        pytest.param(15, 14, 0, id="custom-below"),
        pytest.param(15, 15, 1, id="custom-at-minimum"),
    ],
)
def test_candidate_match_minimum(
    minimum_matches: int | None, count: int, expected_fits: int, image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the original McByte++ default and custom candidate thresholds before affine fitting."""
    config = (
        ConditionalCMCConfig() if minimum_matches is None else ConditionalCMCConfig(minimum_matches=minimum_matches)
    )
    cmc = ConditionalCMC(config)
    feature_inputs(cmc, count)
    calls = []

    def fit(*args, **kwargs):
        calls.append(args)
        return np.eye(2, 3), None

    monkeypatch.setattr(cv2, "estimateAffinePartial2D", fit)
    cmc.estimate(image)
    cmc.estimate(image)
    assert len(calls) == expected_fits


@pytest.mark.parametrize(
    "affine,expected_failures",
    [pytest.param(None, 0, id="unavailable"), pytest.param(np.eye(2, 3) * 4, 1, id="rejected")],
)
def test_unsuccessful_fit_keeps_current_reference(
    affine: np.ndarray | None, expected_failures: int, image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unavailable and rejected fits both advance features, but only rejected transforms count as failures."""
    cmc = ConditionalCMC()
    payload = feature_inputs(cmc)
    cmc.estimate(image)
    payload.descriptors = np.ones_like(payload.descriptors)
    monkeypatch.setattr(cv2, "estimateAffinePartial2D", lambda *args, **kwargs: (affine, None))
    np.testing.assert_array_equal(cmc.estimate(image), np.eye(2, 3))
    np.testing.assert_array_equal(cmc._prev_desc, payload.descriptors)
    assert cmc.frames_failed == expected_failures


def test_detection_exclusion_clips_to_image_edges(image: np.ndarray) -> None:
    """A partly off-screen box must still exclude its entire visible region."""
    # Include the full image so the ROI cannot hide an off-by-one clipping error.
    cmc = ConditionalCMC(ConditionalCMCConfig(roi_min_frac=0.0, roi_max_frac=1.0))
    # Inspect the feature-detection mask: 0 excludes pixels; 255 leaves them available.
    masks = []

    def capture_mask(frame, mask):
        masks.append(mask.copy())
        return []

    cmc.detector = SimpleNamespace(detect=capture_mask)
    cmc.extractor = SimpleNamespace(compute=lambda *args: ([], None))
    # Default downscale=4 produces a 120x160 mask; the boxes cross its left and bottom/right edges.
    cmc.estimate(image, np.array([[-20.0, 20.0, 80.0, 80.0], [600.0, 400.0, 800.0, 600.0]]))
    assert not masks[0][5:20, :20].any()
    # This region includes the final row and column of the downscaled image.
    assert not masks[0][100:, 150:].any()
    assert masks[0][-1, 40] == 255  # An unoccluded border pixel remains available.


def test_unexpected_errors_propagate(image: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    """Programming errors must not be mislabeled as an expected OpenCV fallback."""
    cmc = ConditionalCMC()

    def broken(*args, **kwargs):
        raise RuntimeError("broken implementation")

    monkeypatch.setattr(cmc, "_estimate_feature_affine", broken)
    with pytest.raises(RuntimeError, match="broken implementation"):
        cmc.estimate(image)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_uses_the_same_configured_transform_guard(
    method: CMCMethod, image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No estimator may bypass conditional acceptance or its configured bounds."""
    cmc = ConditionalCMC(ConditionalCMCConfig(method=method, maximum_scale=1.5))
    name = {
        "orb": "_estimate_feature_affine",
        "sift": "_estimate_feature_affine",
        "sparseOptFlow": "_estimate_sparse_optflow",
        "ecc": "_estimate_ecc",
    }[method]
    monkeypatch.setattr(cmc, name, lambda *args: np.eye(2, 3) * 2)
    np.testing.assert_array_equal(cmc.estimate(image), np.eye(2, 3))
    assert cmc.frames_failed == 1


def test_zoom_updates_size_and_covariance() -> None:
    """A valid zoom must transform size and covariance as in the original McByte++."""
    track = McBytePlusPlusTracklet(np.array([10.0, 20.0, 50.0, 120.0]))
    # Nonzero centre and size velocities make missed velocity scaling detectable.
    track.state_estimator.kf.state[4:, 0] = [2.0, -1.0, 0.5, -0.25]
    before = track.state_estimator.kf.state.copy()
    covariance = track.state_estimator.kf.state_covariance.copy()
    ConditionalCMC().apply_batch(np.array([[1.2, 0.0, 5.0], [0.0, 1.2, -3.0]]), [track])
    # A 1.2x zoom scales all state components; translation affects only the centre.
    expected = before * 1.2
    expected[:2, 0] += [5.0, -3.0]
    np.testing.assert_allclose(track.state_estimator.kf.state, expected)
    # Covariance scales by 1.2 squared: P' = M @ P @ M.T with M = 1.2 * I.
    np.testing.assert_allclose(track.state_estimator.kf.state_covariance, covariance * 1.2**2)


def test_rotation_rejects_only_the_invalid_track() -> None:
    """Keep the original McByte++ size/covariance mapping and reject invalid candidates in Trackers."""
    # Both boxes start valid: 100x100 for the square and 40x100 for the narrow track.
    square = McBytePlusPlusTracklet(np.array([10.0, 20.0, 110.0, 120.0]))
    narrow = McBytePlusPlusTracklet(np.array([10.0, 20.0, 50.0, 120.0]))
    # State layout: [xc, yc, w, h, vx, vy, vw, vh]. Nonzero velocities expose missing velocity transformations.
    square.state_estimator.kf.state[4:, 0] = [4.0, -2.0, 1.0, -0.5]
    narrow.state_estimator.kf.state[4:, 0] = [1.0, 2.0, -0.25, 0.5]
    # Unequal variances make rotation change uncertainty rather than leave it invariant.
    square.state_estimator.kf.state_covariance = np.diag([4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0])
    narrow_state = narrow.state_estimator.kf.state.copy()
    narrow_covariance = narrow.state_estimator.kf.state_covariance.copy()
    root3 = np.sqrt(3.0)
    rotation = np.array([[root3 / 2, -0.5], [0.5, root3 / 2]])
    affine = np.column_stack((rotation, [5.0, -3.0]))
    # The original McByte++ size approximation rotates (w, h) directly.
    # Candidate widths become ~36.60 (square) and ~-15.36 (narrow), so only the narrow candidate must be rejected.
    ConditionalCMC().apply_batch(affine, [square, narrow])
    # Hand-computed 30-degree rotation; translation applies only to the centre.
    expected = np.array(
        [
            30 * root3 - 30,
            35 * root3 + 27,
            50 * root3 - 50,
            50 * root3 + 50,
            2 * root3 + 1,
            2 - root3,
            root3 / 2 + 0.25,
            0.5 - root3 / 4,
        ]
    )
    # For each independent pair (a, b), rotated variances are (3a+b)/4 and
    # (a+3b)/4, with cross-covariance sqrt(3)*(a-b)/4.
    expected_covariance = np.diag([5.25, 7.75, 18.25, 22.75, 39.25, 45.75, 68.25, 76.75])
    for offset, difference in zip((0, 2, 4, 6), (-5, -9, -13, -17)):
        cross_covariance = root3 * difference / 4
        expected_covariance[offset, offset + 1] = cross_covariance
        expected_covariance[offset + 1, offset] = cross_covariance
    # Allow harmless floating-point rounding in the accepted state and covariance.
    np.testing.assert_allclose(square.state_estimator.kf.state.ravel(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(square.state_estimator.kf.state_covariance, expected_covariance, rtol=1e-12, atol=1e-12)
    # Reject the narrow candidate as a whole: neither its state nor its covariance may change.
    np.testing.assert_array_equal(narrow.state_estimator.kf.state, narrow_state)
    np.testing.assert_array_equal(narrow.state_estimator.kf.state_covariance, narrow_covariance)


def test_covariance_overflow_does_not_commit_a_valid_mean() -> None:
    """Mean and covariance must be accepted together even if only covariance fails."""
    track = McBytePlusPlusTracklet(np.array([10.0, 20.0, 50.0, 120.0]))
    track.state_estimator.kf.state_covariance[:] = np.eye(8) * 1e308
    before = track.state_estimator.kf.state.copy(), track.state_estimator.kf.state_covariance.copy()
    ConditionalCMC().apply_batch(np.eye(2, 3) * 2, [track])
    np.testing.assert_array_equal(track.state_estimator.kf.state, before[0])
    np.testing.assert_array_equal(track.state_estimator.kf.state_covariance, before[1])


@pytest.mark.parametrize(
    "affine",
    [
        None,
        np.eye(3),
        np.full((2, 3), np.nan),
        np.array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]),
        np.array([[3.1, 0.0, 0.0], [0.0, 3.1, 0.0]]),
    ],
)
def test_invalid_global_warp_changes_no_state(affine: np.ndarray | None) -> None:
    """Global transform rejection preserves the complete prediction and uncertainty."""
    track = McBytePlusPlusTracklet(np.array([10.0, 20.0, 50.0, 120.0]))
    before = track.state_estimator.kf.state.copy(), track.state_estimator.kf.state_covariance.copy()
    ConditionalCMC().apply_batch(affine, [track])
    np.testing.assert_array_equal(track.state_estimator.kf.state, before[0])
    np.testing.assert_array_equal(track.state_estimator.kf.state_covariance, before[1])


@pytest.mark.parametrize(
    "representation,expected_box,expected_variances",
    [
        (XYXYStateEstimator, [22.5, 34.0, 82.5, 234.0], [2.25, 8.0, 6.75, 16.0, 11.25, 24.0, 15.75, 32.0]),
        (XCYCSRStateEstimator, [32.5, 84.0, 72.5, 184.0], [2.25, 8.0, 3.0, 4.0, 11.25, 24.0, 7.0]),
    ],
)
def test_other_state_representations_keep_their_warp_rules(
    representation: type[BaseStateEstimator], expected_box: list[float], expected_variances: list[float]
) -> None:
    """Translation and axis-aligned scaling must respect the XYXY and XCYCSR geometry/covariance layouts."""
    track = McBytePlusPlusTracklet(np.array([10.0, 20.0, 50.0, 120.0]), representation)
    # Distinct variances expose missing or incorrect scaling of individual state dimensions.
    covariance = np.diag(np.arange(1, track.state_estimator.kf.dim_x + 1, dtype=np.float64))
    track.state_estimator.kf.state_covariance = covariance.copy()
    cmc = ConditionalCMC()
    # Pure translation changes position but leaves uncertainty unchanged.
    cmc.apply_batch(np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]]), [track])
    np.testing.assert_allclose(track.get_state_bbox(), [15.0, 17.0, 55.0, 117.0])
    np.testing.assert_array_equal(track.state_estimator.kf.state_covariance, covariance)
    # XYXY scales the corners; XCYCSR scales the centre but retains area/aspect ratio.
    cmc.apply_batch(np.array([[1.5, 0.0, 0.0], [0.0, 2.0, 0.0]]), [track])
    np.testing.assert_allclose(track.get_state_bbox(), expected_box)
    # Only transformed dimensions get their variances multiplied by the corresponding squared scale.
    np.testing.assert_allclose(track.state_estimator.kf.state_covariance, np.diag(expected_variances), atol=1e-12)


@pytest.mark.parametrize("translation", [(0.0, 0.0), (5.0, -3.0)], ids=["identity", "translation"])
def test_xyxy_identity_and_translation_preserve_corner_motion(translation: tuple[float, float]) -> None:
    """Camera translation must not reverse a shrinking box or mix velocities between tracks."""
    tracks = [
        McBytePlusPlusTracklet(np.array([10.0, 20.0, 50.0, 80.0]), XYXYStateEstimator),
        McBytePlusPlusTracklet(np.array([100.0, 200.0, 160.0, 290.0]), XYXYStateEstimator),
    ]
    # The first box shrinks; the second expands. Corner velocities have no min/max ordering requirement.
    tracks[0].state_estimator.kf.state[4:, 0] = [1.0, 2.0, -1.0, -2.0]
    tracks[1].state_estimator.kf.state[4:, 0] = [-2.0, -3.0, 4.0, 5.0]
    before = np.stack([track.state_estimator.kf.state[:, 0].copy() for track in tracks])
    covariances = [track.state_estimator.kf.state_covariance.copy() for track in tracks]
    affine = np.column_stack((np.eye(2), translation))
    ConditionalCMC().apply_batch(affine, tracks)
    expected = before.copy()
    expected[:, :4] += np.tile(translation, 2)
    for track, state, covariance in zip(tracks, expected, covariances):
        np.testing.assert_array_equal(track.state_estimator.kf.state[:, 0], state)
        np.testing.assert_array_equal(track.state_estimator.kf.state_covariance, covariance)
        # Check the next real Kalman prediction as well as the immediate CMC result.
        track.state_estimator.predict()
        np.testing.assert_allclose(track.get_state_bbox(), state[:4] + state[4:])


@pytest.mark.parametrize(
    "affine,expected_box,expected_velocity",
    [
        pytest.param(
            [[0.0, -1.0, 5.0], [1.0, 0.0, -3.0]],
            [-75.0, 7.0, -15.0, 47.0],
            [4.0, 1.0, -2.0, -3.0],
            id="quarter-turn",
        ),
        pytest.param(
            [[1.0, 0.5, 5.0], [-0.25, 1.0, -3.0]],
            [25.0, 4.5, 95.0, 74.5],
            [2.0, 2.75, -5.0, -4.25],
            id="shear",
        ),
        pytest.param(
            [[-1.0, 0.0, 60.0], [0.0, 1.0, 0.0]],
            [10.0, 20.0, 50.0, 80.0],
            [3.0, 2.0, -1.0, -4.0],
            id="reflection",
        ),
    ],
)
def test_xyxy_edge_velocities_follow_transformed_corners(
    affine: list[list[float]], expected_box: list[float], expected_velocity: list[float]
) -> None:
    """Each enclosing edge keeps the velocity of its spatial corner, even when axes mix or reverse."""
    track = McBytePlusPlusTracklet(np.array([10.0, 20.0, 50.0, 80.0]), XYXYStateEstimator)
    track.state_estimator.kf.state[4:, 0] = [1.0, 2.0, -3.0, -4.0]
    # Hand-computed expectations deliberately differ from sorting the transformed velocities.
    ConditionalCMC().apply_batch(np.array(affine), [track])
    np.testing.assert_allclose(track.get_state_bbox(), expected_box)
    np.testing.assert_allclose(track.state_estimator.kf.state[4:, 0], expected_velocity)


@pytest.mark.parametrize(
    "settings",
    [
        {"minimum_matches": 4},
        {"minimum_matches": 12.5},
        {"minimum_scale": 0.0},
        {"maximum_scale": float("inf")},
        {"minimum_scale": 1.1},
    ],
)
def test_invalid_acceptance_settings_fail_early(settings: dict) -> None:
    """Invalid thresholds should fail during construction, before any image processing."""
    with pytest.raises(ValueError):
        ConditionalCMC(ConditionalCMCConfig(**settings))
