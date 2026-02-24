# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from trackers.utils.converters import xcycsr_to_xyxy, xyxy_to_xcycsr
from trackers.utils.kalman_filter import KalmanFilter


class StateRepresentation(Enum):
    """Kalman filter state representation for bounding boxes.

    Attributes:
        XCYCSR: Center-based representation with 7 state variables:
            `x_center`, `y_center` (box center), `scale` (area), `aspect_ratio`
            (width/height), and velocities `vx`, `vy`, `vs`. Aspect ratio is
            treated as constant (no velocity term). Used in original SORT and
            OC-SORT papers.
        XYXY: Corner-based representation with 8 state variables:
            `x1`, `y1` (top-left corner), `x2`, `y2` (bottom-right corner),
            and velocities `vx1`, `vy1`, `vx2`, `vy2` for each coordinate.
            More direct representation, potentially better for non-rigid objects.
    """

    XCYCSR = "xcycsr"
    XYXY = "xyxy"


class BaseStateEstimator(ABC):
    """Abstract Kalman filter with a specific bounding box state representation.

    Wraps a `KalmanFilter` and provides a unified interface for
    bounding-box tracking regardless of the internal state encoding.
    Subclasses configure the filter dimensions, matrices, noise, and
    handle conversions between `[x1, y1, x2, y2]` bboxes and the
    internal state/measurement vectors.

    Attributes:
        kf: The underlying Kalman filter instance.
    """

    def __init__(self, bbox: np.ndarray) -> None:
        """Initialise the filter with the first detection.

        Args:
            bbox: First detection `[x1, y1, x2, y2]`.
        """
        self.kf: KalmanFilter = self._create_filter(bbox)

    @abstractmethod
    def _create_filter(self, bbox: np.ndarray) -> KalmanFilter:
        """Create and configure a Kalman filter for *bbox*.

        Args:
            bbox: First detection `[x1, y1, x2, y2]`.

        Returns:
            A fully configured `KalmanFilter`.
        """

    @abstractmethod
    def bbox_to_measurement(self, bbox: np.ndarray) -> np.ndarray:
        """Convert an `[x1, y1, x2, y2]` bbox to a measurement vector.

        Args:
            bbox: Bounding box `[x1, y1, x2, y2]`.

        Returns:
            Measurement vector suitable for `KalmanFilter.update`.
        """

    @abstractmethod
    def state_to_bbox(self) -> np.ndarray:
        """Extract an `[x1, y1, x2, y2]` bbox from the current filter state.

        Returns:
            Bounding box `[x1, y1, x2, y2]`.
        """

    @abstractmethod
    def clamp_velocity(self) -> None:
        """Clamp velocity components to prevent degenerate predictions.

        Called before `predict` to ensure physical plausibility
        (e.g. non-negative scale). Modifies the filter state in-place.
        """

    def predict(self) -> None:
        """Run the Kalman filter prediction step."""
        self.clamp_velocity()
        self.kf.predict()

    def update(self, bbox: np.ndarray | None) -> None:
        """Update the filter with a new observation.

        Args:
            bbox: Bounding box `[x1, y1, x2, y2]` or `None` when no
                observation is available.
        """
        if bbox is not None:
            self.kf.update(self.bbox_to_measurement(bbox))
        else:
            self.kf.update(None)

    def get_state(self) -> dict:
        """Snapshot the filter state for later restoration (e.g. ORU freeze).

        Returns:
            Opaque state dictionary.
        """
        return self.kf.get_state()

    def set_state(self, state: dict) -> None:
        """Restore a previously saved filter state.

        Args:
            state: Dictionary from `get_state`.
        """
        self.kf.set_state(state)


class XCYCSRStateEstimator(BaseStateEstimator):
    """Center-based Kalman filter with 7 state dimensions and 4 measurements.

    State vector contains `x_center`, `y_center` (box center), `scale` (area),
    `aspect_ratio` (width/height), and velocities `vx`, `vy`, `vs`. Aspect ratio
    is treated as constant (no velocity term), which works well for rigid objects
    that maintain their shape. Matches the representation used in the original
    SORT and OC-SORT papers.
    """

    def _create_filter(self, bbox: np.ndarray) -> KalmanFilter:
        kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition: constant velocity model
        kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],  # aspect ratio: no velocity
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Measurement function: observe (x, y, s, r) from state
        kf.H = np.eye(4, 7, dtype=np.float64)

        # Noise tuning (from OC-SORT paper)
        kf.R[2:, 2:] *= 10.0
        kf.P[4:, 4:] *= 1000.0  # high uncertainty for velocities
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01

        # Initialise state with first observation
        kf.x[:4] = xyxy_to_xcycsr(bbox).reshape((4, 1))

        return kf

    def bbox_to_measurement(self, bbox: np.ndarray) -> np.ndarray:
        return xyxy_to_xcycsr(bbox)

    def state_to_bbox(self) -> np.ndarray:
        return xcycsr_to_xyxy(self.kf.x[:4].reshape((4,)))

    def clamp_velocity(self) -> None:
        # If predicted scale would go negative, zero out scale velocity
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0


class XYXYStateEstimator(BaseStateEstimator):
    """Corner-based Kalman filter with 8 state dimensions and 4 measurements.

    State vector contains `x1`, `y1` (top-left corner), `x2`, `y2` (bottom-right
    corner), and independent velocities `vx1`, `vy1`, `vx2`, `vy2` for each
    coordinate. This allows the box shape to change over time, which may be
    better suited for non-rigid or deformable objects.
    """

    def _create_filter(self, bbox: np.ndarray) -> KalmanFilter:
        kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition: constant velocity for all coordinates
        kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],  # x1 += vx1
                [0, 1, 0, 0, 0, 1, 0, 0],  # y1 += vy1
                [0, 0, 1, 0, 0, 0, 1, 0],  # x2 += vx2
                [0, 0, 0, 1, 0, 0, 0, 1],  # y2 += vy2
                [0, 0, 0, 0, 1, 0, 0, 0],  # vx1
                [0, 0, 0, 0, 0, 1, 0, 0],  # vy1
                [0, 0, 0, 0, 0, 0, 1, 0],  # vx2
                [0, 0, 0, 0, 0, 0, 0, 1],  # vy2
            ],
            dtype=np.float64,
        )

        # Measurement function: observe (x1, y1, x2, y2) from state
        kf.H = np.eye(4, 8, dtype=np.float64)

        # Noise tuning (similar scaling to XCYCSR version)
        kf.R *= 1.0  # measurement noise
        kf.P[4:, 4:] *= 1000.0  # high uncertainty for velocities
        kf.P *= 10.0
        kf.Q[4:, 4:] *= 0.01

        # Initialise state with first observation (direct XYXY)
        kf.x[:4] = bbox.reshape((4, 1))

        return kf

    def bbox_to_measurement(self, bbox: np.ndarray) -> np.ndarray:
        return bbox

    def state_to_bbox(self) -> np.ndarray:
        return self.kf.x[:4].reshape((4,))

    def clamp_velocity(self) -> None:
        # No clamping needed for XYXY representation
        pass


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

_REPR_MAP: dict[StateRepresentation, type[BaseStateEstimator]] = {
    StateRepresentation.XCYCSR: XCYCSRStateEstimator,
    StateRepresentation.XYXY: XYXYStateEstimator,
}


def create_state_estimator(
    state_repr: StateRepresentation,
    bbox: np.ndarray,
) -> BaseStateEstimator:
    """Create a state estimator for the given state representation.

    Args:
        state_repr: The desired representation. Ex: StateRepresentation.XCYCSR
        bbox: First detection `[x1, y1, x2, y2]`.

    Returns:
        An initialised `BaseStateEstimator` wrapping a configured
        estimator.

    Raises:
        ValueError: If *state_repr* is not recognised.
    """
    cls = _REPR_MAP.get(state_repr, None)
    if cls is None:
        raise ValueError(
            f"Unknown state representation: {state_repr!r}. "
            f"Available: {list(_REPR_MAP.keys())}"
        )
    return cls(bbox)
