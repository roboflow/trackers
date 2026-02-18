# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from trackers.motion.transformation import (
    CoordinatesTransformation,
    HomographyTransformation,
    IdentityTransformation,
)

_OPTICAL_FLOW_WINDOW_SIZE = (21, 21)
_OPTICAL_FLOW_MAX_PYRAMID_LEVEL = 3
_OPTICAL_FLOW_MAX_ITERATIONS = 30
_OPTICAL_FLOW_EPSILON = 0.01
_MIN_POINTS_FOR_HOMOGRAPHY = 4


class MotionEstimator:
    """Estimates camera motion between consecutive video frames.

    Uses sparse optical flow (Lucas-Kanade) to track feature points and computes
    the geometric transformation between frames. Accumulates transformations to
    maintain a consistent world coordinate system relative to the first frame.

    Args:
        max_points: Maximum number of feature points to track. More points
            increase accuracy but reduce speed. Default: 200.
        min_distance: Minimum distance between detected feature points.
            Larger values spread points more evenly. Default: 15.
        block_size: Size of the averaging block for corner detection.
            Larger values make detection less sensitive to noise. Default: 3.
        quality_level: Quality threshold for corner detection (0-1).
            Higher values detect fewer, stronger corners. Default: 0.01.
        ransac_reproj_threshold: RANSAC inlier threshold in pixels.
            Points with reprojection error below this are considered inliers.
            Default: 3.0.

    Example:
        ```python
        import cv2
        from trackers.motion import MotionEstimator

        estimator = MotionEstimator()

        cap = cv2.VideoCapture("video.mp4")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get transformation for this frame
            coord_transform = estimator.update(frame)

            # Transform trajectory points from world to frame coordinates
            frame_points = coord_transform.abs_to_rel(world_trajectory)
        ```
    """

    def __init__(
        self,
        max_points: int = 200,
        min_distance: int = 15,
        block_size: int = 3,
        quality_level: float = 0.01,
        ransac_reproj_threshold: float = 3.0,
    ) -> None:
        self.max_points = max_points
        self.min_distance = min_distance
        self.block_size = block_size
        self.quality_level = quality_level
        self.ransac_reproj_threshold = ransac_reproj_threshold

        self._optical_flow_params = dict(
            winSize=_OPTICAL_FLOW_WINDOW_SIZE,
            maxLevel=_OPTICAL_FLOW_MAX_PYRAMID_LEVEL,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                _OPTICAL_FLOW_MAX_ITERATIONS,
                _OPTICAL_FLOW_EPSILON,
            ),
        )

        self._previous_grayscale: Optional[np.ndarray] = None
        self._previous_features: Optional[np.ndarray] = None
        self._accumulated_homography: np.ndarray = np.eye(3, dtype=np.float64)

    def update(self, frame: np.ndarray) -> CoordinatesTransformation:
        """Process a new frame and return the coordinate transformation.

        The returned transformation converts between:
        - Absolute coordinates: Position relative to the first frame
        - Relative coordinates: Position in the current frame

        Args:
            frame: Current video frame (BGR or grayscale).

        Returns:
            `CoordinatesTransformation` for converting between absolute and
                relative coordinates. Returns `IdentityTransformation` for the
                first frame or if motion estimation fails.
        """
        if len(frame.shape) == 3:
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = frame.copy()

        if self._previous_grayscale is None:
            self._previous_grayscale = grayscale
            self._previous_features = self._find_features(grayscale)
            return IdentityTransformation()

        current_features = self._find_features(grayscale)

        if (
            self._previous_features is None
            or len(self._previous_features) < _MIN_POINTS_FOR_HOMOGRAPHY
        ):
            self._previous_grayscale = grayscale
            self._previous_features = current_features
            return self._get_current_transformation()

        tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._previous_grayscale,
            grayscale,
            self._previous_features,
            None,
            **self._optical_flow_params,
        )

        if tracked_points is None:
            self._previous_grayscale = grayscale
            self._previous_features = current_features
            return self._get_current_transformation()

        status = status.flatten()
        valid_previous = self._previous_features[status == 1]
        valid_current = tracked_points[status == 1]

        if len(valid_previous) < _MIN_POINTS_FOR_HOMOGRAPHY:
            self._previous_grayscale = grayscale
            self._previous_features = current_features
            return self._get_current_transformation()

        transform = self._estimate_homography(valid_previous, valid_current)

        self._previous_grayscale = grayscale
        self._previous_features = current_features

        if transform is None:
            return self._get_current_transformation()

        return transform

    def _find_features(self, grayscale: np.ndarray) -> Optional[np.ndarray]:
        """Detect good features to track in the grayscale image.

        Args:
            grayscale: Grayscale image.

        Returns:
            Array of feature points of shape `(N, 1, 2)`, or `None` if no
            features found.
        """
        return cv2.goodFeaturesToTrack(
            grayscale,
            maxCorners=self.max_points,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
        )

    def _estimate_homography(
        self, previous_points: np.ndarray, current_points: np.ndarray
    ) -> Optional[CoordinatesTransformation]:
        """Estimate homography transformation between point sets.

        Args:
            previous_points: Points from previous frame.
            current_points: Corresponding points in current frame.

        Returns:
            `HomographyTransformation` or `None` if estimation fails.
        """
        previous_points_2d = previous_points.reshape(-1, 2)
        current_points_2d = current_points.reshape(-1, 2)

        homography_matrix, _ = cv2.findHomography(
            previous_points_2d,
            current_points_2d,
            cv2.RANSAC,
            self.ransac_reproj_threshold,
        )

        if homography_matrix is None:
            return None

        # H_total = H_current @ H_previous gives transformation from frame 0
        self._accumulated_homography = homography_matrix @ self._accumulated_homography

        return HomographyTransformation(self._accumulated_homography)

    def _get_current_transformation(self) -> CoordinatesTransformation:
        """Get the current accumulated transformation.

        Returns:
            The accumulated transformation, or IdentityTransformation
            if no motion has been estimated yet.
        """
        if np.allclose(self._accumulated_homography, np.eye(3)):
            return IdentityTransformation()
        return HomographyTransformation(self._accumulated_homography)

    def reset(self) -> None:
        """Reset the estimator state.

        Call this when starting a new video or when you want to reset the
        world coordinate system to the current frame.
        """
        self._previous_grayscale = None
        self._previous_features = None
        self._accumulated_homography = np.eye(3, dtype=np.float64)
