# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for coordinate transformations."""

from __future__ import annotations

import numpy as np
import pytest

from trackers.motion.transformation import HomographyTransformation, IdentityTransformation

TRANSLATION_MATRIX = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]])


def test_identity_transformation_returns_points_unchanged() -> None:
    """IdentityTransformation is a no-op in both directions."""
    points = np.array([[100.0, 200.0], [300.0, 400.0]])
    transformation = IdentityTransformation()

    np.testing.assert_allclose(transformation.abs_to_rel(points), points)
    np.testing.assert_allclose(transformation.rel_to_abs(points), points)


def test_homography_transformation_roundtrip() -> None:
    """abs_to_rel followed by rel_to_abs recovers the original points."""
    points = np.array([[100.0, 200.0], [300.0, 400.0]])
    transformation = HomographyTransformation(TRANSLATION_MATRIX)

    relative = transformation.abs_to_rel(points)

    np.testing.assert_allclose(relative, points + np.array([10.0, 20.0]))
    np.testing.assert_allclose(transformation.rel_to_abs(relative), points)


def test_homography_transformation_rejects_wrong_shape() -> None:
    """A matrix that is not 3x3 is rejected."""
    with pytest.raises(ValueError, match="must be 3x3"):
        HomographyTransformation(np.eye(2))


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param(np.zeros((3, 3)), id="all-zero"),
        pytest.param(np.ones((3, 3)), id="rank-one"),
        pytest.param(np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), id="duplicate-row"),
    ],
)
def test_homography_transformation_rejects_singular_matrix(matrix: np.ndarray) -> None:
    """A singular matrix raises ValueError rather than surfacing a bare LinAlgError."""
    with pytest.raises(ValueError, match="singular"):
        HomographyTransformation(matrix)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="inf"),
    ],
)
def test_homography_transformation_rejects_non_finite_matrix(value: float) -> None:
    """A matrix carrying NaN or infinity is rejected before it can poison every transform."""
    matrix = TRANSLATION_MATRIX.copy()
    matrix[0, 2] = value

    with pytest.raises(ValueError, match="finite"):
        HomographyTransformation(matrix)
