# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trackers.calibration.types import PitchDimensions


def _as_points(points: np.ndarray | list[list[float]] | list[tuple[float, float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        if array.shape[0] != 2:
            raise ValueError("Expected a single point with shape (2,)")
        return array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {array.shape}")
    return array


@dataclass(frozen=True)
class PitchModel:
    """Canonical pitch coordinate system."""

    dimensions: PitchDimensions = field(default_factory=PitchDimensions)

    def metric_to_normalized(self, points: np.ndarray | list[list[float]]) -> np.ndarray:
        metric_points = _as_points(points)
        normalized = metric_points.copy()
        normalized[:, 0] /= self.dimensions.length_m
        normalized[:, 1] /= self.dimensions.width_m
        return normalized

    def normalized_to_metric(self, points: np.ndarray | list[list[float]]) -> np.ndarray:
        normalized_points = _as_points(points)
        metric = normalized_points.copy()
        metric[:, 0] *= self.dimensions.length_m
        metric[:, 1] *= self.dimensions.width_m
        return metric

    def contains_metric_points(
        self,
        points: np.ndarray | list[list[float]],
        *,
        tolerance_m: float = 0.0,
    ) -> np.ndarray:
        metric_points = _as_points(points)
        return (
            (metric_points[:, 0] >= -tolerance_m)
            & (metric_points[:, 0] <= self.dimensions.length_m + tolerance_m)
            & (metric_points[:, 1] >= -tolerance_m)
            & (metric_points[:, 1] <= self.dimensions.width_m + tolerance_m)
        )
