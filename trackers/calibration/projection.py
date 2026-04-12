# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.calibration.types import CalibrationFrame


def _as_points(
    points: np.ndarray | list[list[float]] | list[tuple[float, float]],
) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        if array.shape[0] != 2:
            raise ValueError("Expected a single point with shape (2,)")
        return array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {array.shape}")
    return array


def invert_homography(homography: np.ndarray) -> np.ndarray:
    """Invert a 3x3 homography matrix."""
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 homography matrix, got {matrix.shape}")
    return np.linalg.inv(matrix)


def apply_homography(
    points: np.ndarray | list[list[float]] | list[tuple[float, float]],
    homography: np.ndarray,
) -> np.ndarray:
    """Project 2D points through a homography matrix."""
    points_array = _as_points(points)
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 homography matrix, got {matrix.shape}")

    homogeneous = np.concatenate(
        [points_array, np.ones((points_array.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = homogeneous @ matrix.T
    scale = transformed[:, 2:3]
    safe_scale = np.where(np.abs(scale) < 1e-9, 1e-9, scale)
    return transformed[:, :2] / safe_scale


def project_image_points_to_pitch(
    points: np.ndarray | list[list[float]] | list[tuple[float, float]],
    calibration: CalibrationFrame,
) -> np.ndarray:
    """Project image-space points into pitch metric coordinates."""
    if calibration.image_to_pitch is None:
        raise ValueError("Calibration frame is missing image_to_pitch homography")
    return apply_homography(points, calibration.image_to_pitch)


def project_pitch_points_to_image(
    points: np.ndarray | list[list[float]] | list[tuple[float, float]],
    calibration: CalibrationFrame,
) -> np.ndarray:
    """Project pitch metric coordinates back into image space."""
    if calibration.pitch_to_image is None:
        raise ValueError("Calibration frame is missing pitch_to_image homography")
    return apply_homography(points, calibration.pitch_to_image)


def bottom_center_from_xyxy(boxes: np.ndarray | list[list[float]]) -> np.ndarray:
    """Return bottom-center anchor points for xyxy boxes."""
    boxes_array = np.asarray(boxes, dtype=np.float64)
    if boxes_array.ndim != 2 or boxes_array.shape[1] != 4:
        raise ValueError(f"Expected boxes with shape (N, 4), got {boxes_array.shape}")
    return np.column_stack(
        (
            (boxes_array[:, 0] + boxes_array[:, 2]) / 2.0,
            boxes_array[:, 3],
        )
    )


def bottom_center_from_xywh(boxes: np.ndarray | list[list[float]]) -> np.ndarray:
    """Return bottom-center anchor points for xywh boxes."""
    boxes_array = np.asarray(boxes, dtype=np.float64)
    if boxes_array.ndim != 2 or boxes_array.shape[1] != 4:
        raise ValueError(f"Expected boxes with shape (N, 4), got {boxes_array.shape}")
    return np.column_stack(
        (
            boxes_array[:, 0] + (boxes_array[:, 2] / 2.0),
            boxes_array[:, 1] + boxes_array[:, 3],
        )
    )
