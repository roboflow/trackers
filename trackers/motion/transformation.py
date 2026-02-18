# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CoordinatesTransformation(ABC):
    """Abstract base class for coordinate transformations.

    Subclasses implement specific transformation types that convert points between
    absolute (world) and relative (frame) coordinates.
    """

    @abstractmethod
    def abs_to_rel(self, points: np.ndarray) -> np.ndarray:
        """Transform points from absolute (world) to relative (frame) coordinates.

        Args:
            points: Array of shape `(N, 2)` containing `(x, y)` coordinates
                in absolute/world space.

        Returns:
            Array of shape `(N, 2)` containing `(x, y)` coordinates
            in relative/frame space.
        """
        pass

    @abstractmethod
    def rel_to_abs(self, points: np.ndarray) -> np.ndarray:
        """Transform points from relative (frame) to absolute (world) coordinates.

        Args:
            points: Array of shape `(N, 2)` containing `(x, y)` coordinates
                in relative/frame space.

        Returns:
            Array of shape `(N, 2)` containing `(x, y)` coordinates
            in absolute/world space.
        """
        pass


class IdentityTransformation(CoordinatesTransformation):
    """No-op transformation where absolute and relative coordinates are identical.

    Used for the first frame (before any camera motion is detected) or when
    motion estimation fails.
    """

    def abs_to_rel(self, points: np.ndarray) -> np.ndarray:
        """Return points unchanged."""
        return np.atleast_2d(points).copy()

    def rel_to_abs(self, points: np.ndarray) -> np.ndarray:
        """Return points unchanged."""
        return np.atleast_2d(points).copy()


class HomographyTransformation(CoordinatesTransformation):
    """Full perspective transformation using a 3x3 homography matrix.

    Supports rotation, translation, scaling, and perspective changes.
    This is the most general transformation type, suitable for any camera motion.

    The homography matrix maps points from the first frame (absolute) to the
    current frame (relative). The inverse matrix is computed automatically
    for the reverse transformation.

    Args:
        homography_matrix: 3x3 homography matrix that transforms points from
            absolute (first frame) coordinates to relative (current frame)
            coordinates.

    Raises:
        ValueError: If the matrix is not 3x3.

    Example:
        ```python
        import numpy as np

        from trackers import HomographyTransformation

        homography_matrix = np.array([
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 20.0],
            [0.0, 0.0, 1.0],
        ])
        transform = HomographyTransformation(homography_matrix)

        world_points = np.array([[100, 200], [300, 400]])
        frame_points = transform.abs_to_rel(world_points)
        ```
    """

    def __init__(self, homography_matrix: np.ndarray) -> None:
        self.homography_matrix = np.array(homography_matrix, dtype=np.float64)
        if self.homography_matrix.shape != (3, 3):
            raise ValueError(
                f"Homography matrix must be 3x3, got {self.homography_matrix.shape}"
            )
        self.inverse_homography_matrix = np.linalg.inv(self.homography_matrix)

    def _transform_points(self, points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply homography transformation to points.

        Args:
            points: Array of shape `(N, 2)` containing `(x, y)` coordinates.
            matrix: 3x3 homography matrix to apply.

        Returns:
            Transformed points of shape `(N, 2)`.
        """
        points = np.atleast_2d(points)
        if len(points) == 0:
            return points

        ones = np.ones((len(points), 1))
        homogeneous_points = np.hstack((points[:, :2], ones))
        transformed = homogeneous_points @ matrix.T

        # Normalize: (x', y', w') -> (x'/w', y'/w')
        # Points with w <= 0 or very small w are numerically unstable
        # (they map through infinity in projective space)
        homogeneous_scale = transformed[:, 2:3]
        min_scale = 1e-4
        homogeneous_scale = np.where(
            np.abs(homogeneous_scale) < min_scale,
            np.sign(homogeneous_scale + 1e-10) * min_scale,
            homogeneous_scale,
        )
        return transformed[:, :2] / homogeneous_scale

    def abs_to_rel(self, points: np.ndarray) -> np.ndarray:
        """Transform from absolute (world) to relative (frame) coordinates."""
        return self._transform_points(points, self.homography_matrix)

    def rel_to_abs(self, points: np.ndarray) -> np.ndarray:
        """Transform from relative (frame) to absolute (world) coordinates."""
        return self._transform_points(points, self.inverse_homography_matrix)
