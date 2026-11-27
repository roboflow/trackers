# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Covariance contract for XYXY Kalman states under camera-motion compensation.

The XYXY warp reorders corners whenever an axis is reflected or a box arrives inverted (``x2 < x1``): the new left edge
is the old right edge mapped through the transform, and its velocity comes along with it. The covariance must describe
that same warped state, so its index blocks are relabelled with exactly the same permutation and scaled by the same
linear factors. Every oracle here is written out by hand from literal index and scale arrays, or derived from the state
warp alone, and never from the covariance code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from trackers.core.botsort.tracklet import BoTSORTTracklet
from trackers.utils.cmc import CMC
from trackers.utils.state_representations import XYXYStateEstimator

#: Upright box (x1 < x2, y1 < y2) with distinct velocities on every coordinate.
_UPRIGHT = [10.0, 20.0, 50.0, 80.0, 1.0, 2.0, -3.0, 0.5]
#: Same box with its x corners stored inverted (x1 > x2), as a long prediction of a shrinking box can leave it.
_INVERTED_X = [50.0, 20.0, 10.0, 80.0, 1.0, 2.0, -3.0, 0.5]
#: Same box with both axes stored inverted.
_INVERTED_XY = [50.0, 80.0, 10.0, 20.0, 1.0, 2.0, -3.0, 0.5]

#: State index each warped slot takes its value from, for every corner relabelling an axis-aligned warp can produce.
_KEEP = [0, 1, 2, 3, 4, 5, 6, 7]
_SWAP_X = [2, 1, 0, 3, 6, 5, 4, 7]
_SWAP_Y = [0, 3, 2, 1, 4, 7, 6, 5]
_SWAP_XY = [2, 3, 0, 1, 6, 7, 4, 5]


def _asymmetric_covariance() -> np.ndarray:
    """Fixed dense SPD covariance whose left/right and top/bottom variances and cross terms all differ."""
    seed = (np.arange(64, dtype=np.float64).reshape(8, 8) * 7.0 % 11.0 - 5.0) / 5.0 + np.eye(8)
    return seed @ seed.T + np.eye(8)


def _affine(linear: list[list[float]], shift: list[float], dtype: type = np.float64) -> np.ndarray:
    """Stack a 2x2 linear part and a translation into the 2x3 matrix ``CMC.estimate`` returns."""
    return np.hstack([np.array(linear), np.array(shift)[:, np.newaxis]]).astype(dtype)


def _warp_batch(
    states: list[np.ndarray], covariances: list[np.ndarray], affine: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Warp several XYXY tracklets in one ``CMC.apply_batch`` call and return copies of their states/covariances."""
    tracklets = [BoTSORTTracklet(np.array(_UPRIGHT[:4]), state_estimator_class=XYXYStateEstimator) for _ in states]
    for tracklet, state, covariance in zip(tracklets, states, covariances, strict=True):
        tracklet.state_estimator.kf.state = np.array(state, dtype=np.float64).reshape(-1, 1)
        tracklet.state_estimator.kf.state_covariance = covariance.copy()
    CMC.apply_batch(affine, tracklets)
    return (
        [trk.state_estimator.kf.state.reshape(-1).copy() for trk in tracklets],
        [trk.state_estimator.kf.state_covariance.copy() for trk in tracklets],
    )


def _warp(state: np.ndarray, covariance: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp one XYXY tracklet through ``CMC.apply_batch``."""
    states, covariances = _warp_batch([state], [covariance], affine)
    return states[0], covariances[0]


def _relabelled_covariance(covariance: np.ndarray, source: list[int], scale: np.ndarray) -> np.ndarray:
    """Covariance of ``y`` with ``y[i] = scale[i] * x[source[i]]``, written entry by entry from ``cov(x)``."""
    return covariance[np.ix_(source, source)] * np.outer(scale, scale)


def _state_warp_jacobian(state: np.ndarray, affine: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Finite-difference Jacobian of the CMC state warp alone; the covariance passed along is never inspected.

    The XYXY warp is affine in the state as long as no corner pair changes order, so a unit step on boxes tens of pixels
    wide recovers the linear part up to float64 rounding.
    """
    base, _ = _warp(state, np.eye(8), affine)
    columns = [(_warp(state + step * np.eye(8)[k], np.eye(8), affine)[0] - base) / step for k in range(8)]
    return np.stack(columns, axis=1)


_RELABEL_CASES = [
    pytest.param(_UPRIGHT, [[-1.0, 0.0], [0.0, 1.0]], [300.0, 0.0], _SWAP_X, [-1.0, 1.0], id="x-reflection"),
    pytest.param(_UPRIGHT, [[1.0, 0.0], [0.0, -1.0]], [0.0, 200.0], _SWAP_Y, [1.0, -1.0], id="y-reflection"),
    pytest.param(_UPRIGHT, [[-1.0, 0.0], [0.0, -1.0]], [300.0, 200.0], _SWAP_XY, [-1.0, -1.0], id="point-reflection"),
    pytest.param(_UPRIGHT, [[-2.0, 0.0], [0.0, 0.5]], [0.0, 0.0], _SWAP_X, [-2.0, 0.5], id="reflection-and-scale"),
    pytest.param(_INVERTED_X, [[1.5, 0.0], [0.0, 1.0]], [5.0, -3.0], _SWAP_X, [1.5, 1.0], id="scale-of-inverted-box"),
    pytest.param(_INVERTED_XY, [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], _SWAP_XY, [1.0, 1.0], id="identity-inverted-box"),
    pytest.param(_INVERTED_X, [[-1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], _KEEP, [-1.0, 1.0], id="reflect-inverted-box"),
    pytest.param(_UPRIGHT, [[2.0, 0.0], [0.0, 0.5]], [7.0, 4.0], _KEEP, [2.0, 0.5], id="scale-of-upright-box"),
]


class TestXYXYCovarianceFollowsCornerRelabelling:
    """An axis-aligned CMC step must move each corner's uncertainty to the slot the corner itself moves to.

    Each case gives a stored state, an axis-aligned affine transform, the literal source index of every warped slot and
    the per-axis linear factor. The state assertion pins the relabelling the warp applies; the covariance assertion
    requires the covariance of that same relabelled random vector.
    """

    @pytest.mark.parametrize(("state", "linear", "shift", "source", "axis_scale"), _RELABEL_CASES)
    def test_state_is_relabelled_and_scaled(
        self,
        state: list[float],
        linear: list[list[float]],
        shift: list[float],
        source: list[int],
        axis_scale: list[float],
    ) -> None:
        """Every warped state slot equals its source slot times the axis factor, plus the shift on positions.

        This pins down the relabelling the state warp performs, so the covariance test below checks against the labels
        the filter state actually uses rather than against an assumed convention.
        """
        scale = np.tile(axis_scale, 4)
        offset = np.concatenate([np.tile(shift, 2), np.zeros(4)])

        warped_state, _ = _warp(np.array(state), _asymmetric_covariance(), _affine(linear, shift))

        np.testing.assert_allclose(warped_state, scale * np.array(state)[source] + offset, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(("state", "linear", "shift", "source", "axis_scale"), _RELABEL_CASES)
    def test_covariance_is_relabelled_with_the_state(
        self,
        state: list[float],
        linear: list[list[float]],
        shift: list[float],
        source: list[int],
        axis_scale: list[float],
    ) -> None:
        """The warped covariance equals the covariance of the relabelled state, cross terms included.

        With unequal left/right variances and dense cross covariance, a warp that only scales the original index blocks
        leaves the old left-edge uncertainty on the new left edge, which is the old right edge, and every later Kalman
        gain for that edge is computed from the wrong variance.
        """
        covariance = _asymmetric_covariance()

        _, warped_covariance = _warp(np.array(state), covariance, _affine(linear, shift))

        expected = _relabelled_covariance(covariance, source, np.tile(axis_scale, 4))
        np.testing.assert_allclose(warped_covariance, expected, rtol=1e-12, atol=1e-12)

    def test_batch_relabels_each_tracklet_by_its_own_corner_order(self) -> None:
        """One reflection over a mixed batch swaps the upright track's corners and keeps the inverted track's corners.

        Whether a reflection swaps the corner labels depends on how each box is stored, so the batched warp must decide
        the relabelling per tracklet rather than once for the whole batch.
        """
        upright_covariance = _asymmetric_covariance()
        inverted_covariance = _asymmetric_covariance()[::-1, ::-1].copy()
        scale = np.tile([-1.0, 1.0], 4)

        _, (warped_upright, warped_inverted) = _warp_batch(
            [np.array(_UPRIGHT), np.array(_INVERTED_X)],
            [upright_covariance, inverted_covariance],
            _affine([[-1.0, 0.0], [0.0, 1.0]], [300.0, 0.0]),
        )

        np.testing.assert_allclose(warped_upright, _relabelled_covariance(upright_covariance, _SWAP_X, scale))
        np.testing.assert_allclose(warped_inverted, _relabelled_covariance(inverted_covariance, _KEEP, scale))


class TestXYXYCovarianceMatchesStateWarp:
    """The covariance update must be ``J P J.T`` for the Jacobian ``J`` of the state warp itself.

    ``J`` is measured by finite differences of the warped state only, so this holds the covariance and the state warp to
    one linear map for every axis-aligned transform, including float32 transforms whose near-zero cross-axis residuals
    are negative and therefore pick the opposite corner on the other axis.
    """

    @pytest.mark.parametrize(
        ("state", "affine"),
        [
            pytest.param(_UPRIGHT, _affine([[-1.0, 0.0], [0.0, 1.0]], [300.0, 0.0]), id="x-reflection"),
            pytest.param(_UPRIGHT, _affine([[-1.5, 0.0], [0.0, -0.5]], [9.0, 4.0]), id="point-reflection-and-scale"),
            pytest.param(_INVERTED_XY, _affine([[0.8, 0.0], [0.0, 1.2]], [-2.0, 6.0]), id="scale-of-inverted-box"),
            pytest.param(
                _UPRIGHT,
                _affine([[0.999, -5e-7], [4e-7, 1.001]], [3.5, -2.25], dtype=np.float32),
                id="float32-near-axis-aligned-negative-residual",
            ),
            pytest.param(
                _INVERTED_X,
                _affine([[-1.0, 5e-7], [-5e-7, 1.0]], [120.0, 0.0], dtype=np.float32),
                id="float32-reflection-of-inverted-box",
            ),
        ],
    )
    def test_covariance_is_conjugated_by_state_warp_jacobian(self, state: list[float], affine: np.ndarray) -> None:
        """The warped covariance equals ``J P J.T`` with ``J`` measured from the state warp alone.

        Any disagreement means the covariance describes a different random vector than the one the filter now holds.
        """
        covariance = _asymmetric_covariance()
        jacobian = _state_warp_jacobian(np.array(state), affine)

        _, warped_covariance = _warp(np.array(state), covariance, affine)

        np.testing.assert_allclose(warped_covariance, jacobian @ covariance @ jacobian.T, rtol=1e-9, atol=1e-9)


class TestXYXYNoOpTransforms:
    """Transforms that move no corner label must leave the upright track's covariance bit-for-bit unchanged."""

    @pytest.mark.parametrize(
        "affine",
        [
            pytest.param(np.eye(2, 3, dtype=np.float32), id="identity"),
            pytest.param(_affine([[1.0, 0.0], [0.0, 1.0]], [12.5, -7.0], dtype=np.float32), id="pure-translation"),
        ],
    )
    def test_covariance_is_untouched(self, affine: np.ndarray) -> None:
        """Identity and pure translation keep every covariance entry exactly as it was on an upright box.

        These are the transforms CMC returns on a static camera or a pure pan, so any rounding introduced here would
        drift the filter on every frame.
        """
        covariance = _asymmetric_covariance()

        _, warped_covariance = _warp(np.array(_UPRIGHT), covariance, affine)

        np.testing.assert_array_equal(warped_covariance, covariance)

    def test_identity_leaves_state_untouched(self) -> None:
        """The identity transform keeps every position and velocity of an upright box exactly as it was.

        Together with the covariance check this makes the identity a true no-op on the full Gaussian state.
        """
        state = np.array(_UPRIGHT)

        warped_state, _ = _warp(state, _asymmetric_covariance(), np.eye(2, 3, dtype=np.float32))

        np.testing.assert_array_equal(warped_state, state)
