# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Kalman filter state representations for bounding box tracking.

Provides pluggable state representations that define how bounding boxes are
encoded into a Kalman filter state vector. Each representation handles:
- Filter creation with appropriate dimensions and noise tuning
- Converting an ``[x1, y1, x2, y2]`` bbox to a measurement vector
- Converting the state vector back to an ``[x1, y1, x2, y2]`` bbox
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from trackers.utils.converters import xcycsr_to_xyxy, xyxy_to_xcycsr
from trackers.utils.kalman_filter import KalmanFilter


class StateRepresentation(Enum):
    """Kalman filter state representation for bounding boxes.

    XCYCSR: Center-based (x_center, y_center, scale, aspect_ratio, vx, vy, vs)
        - 7 state variables, aspect ratio is constant (no velocity)
        - Used in original SORT/OC-SORT papers

    XYXY: Corner-based (x1, y1, x2, y2, vx1, vy1, vx2, vy2)
        - 8 state variables, all coordinates have velocities
        - More direct representation, potentially better for non-rigid objects
    """

    XCYCSR = "xcycsr"
    XYXY = "xyxy"


class BaseStateRepresentation(ABC):
    """Abstract base for Kalman filter state representations.

    Subclasses define how bounding boxes map to/from a Kalman filter state
    vector, and how the filter matrices (F, H, R, P, Q) are configured.
    """

    @abstractmethod
    def create_filter(self, initial_bbox: np.ndarray) -> KalmanFilter:
        """Create and initialise a Kalman filter for *initial_bbox*.

        Args:
            initial_bbox: First detection ``[x1, y1, x2, y2]``.

        Returns:
            A fully configured :class:`KalmanFilter`.
        """

    @abstractmethod
    def bbox_to_measurement(self, bbox: np.ndarray) -> np.ndarray:
        """Convert an ``[x1, y1, x2, y2]`` bbox to a measurement vector.

        Args:
            bbox: Bounding box ``[x1, y1, x2, y2]``.

        Returns:
            Measurement vector suitable for :meth:`KalmanFilter.update`.
        """

    @abstractmethod
    def state_to_bbox(self, kf: KalmanFilter) -> np.ndarray:
        """Extract an ``[x1, y1, x2, y2]`` bbox from the filter state.

        Args:
            kf: The Kalman filter instance.

        Returns:
            Bounding box ``[x1, y1, x2, y2]``.
        """

    @abstractmethod
    def clamp_velocity(self, kf: KalmanFilter) -> None:
        """Clamp velocity components to prevent degenerate predictions.

        Called before :meth:`KalmanFilter.predict` to ensure physical
        plausibility (e.g. non-negative scale).

        Args:
            kf: The Kalman filter instance (modified in-place).
        """


class XCYCSRStateRepresentation(BaseStateRepresentation):
    """Center-based state: ``[x_c, y_c, scale, ratio, vx, vy, vs]``.

    7 state dimensions, 4 measurement dimensions.
    Aspect ratio is treated as constant (no velocity term).
    Matches the representation used in the original SORT and OC-SORT papers.
    """

    def create_filter(self, initial_bbox: np.ndarray) -> KalmanFilter:
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
        kf.x[:4] = xyxy_to_xcycsr(initial_bbox).reshape((4, 1))

        return kf

    def bbox_to_measurement(self, bbox: np.ndarray) -> np.ndarray:
        return xyxy_to_xcycsr(bbox)

    def state_to_bbox(self, kf: KalmanFilter) -> np.ndarray:
        return xcycsr_to_xyxy(kf.x[:4].reshape((4,)))

    def clamp_velocity(self, kf: KalmanFilter) -> None:
        # If predicted scale would go negative, zero out scale velocity
        if (kf.x[6] + kf.x[2]) <= 0:
            kf.x[6] *= 0.0


class XYXYStateRepresentation(BaseStateRepresentation):
    """Corner-based state: ``[x1, y1, x2, y2, vx1, vy1, vx2, vy2]``.

    8 state dimensions, 4 measurement dimensions.
    All four coordinates carry their own velocity term.
    """

    def create_filter(self, initial_bbox: np.ndarray) -> KalmanFilter:
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
        kf.x[:4] = initial_bbox.reshape((4, 1))

        return kf

    def bbox_to_measurement(self, bbox: np.ndarray) -> np.ndarray:
        return bbox

    def state_to_bbox(self, kf: KalmanFilter) -> np.ndarray:
        return kf.x[:4].reshape((4,))

    def clamp_velocity(self, kf: KalmanFilter) -> None:
        # No clamping needed for XYXY representation
        pass


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

_REPR_MAP: dict[str, type[BaseStateRepresentation]] = {
    'xcycsr': XCYCSRStateRepresentation,
    'xyxy': XYXYStateRepresentation,
}


def get_state_representation(
    state_repr: str,
) -> BaseStateRepresentation:
    """Return a :class:`BaseStateRepresentation` instance for *state_repr*.

    Args:
        state_repr (str): The desired representation enum value.

    Returns:
        An instance of the matching state representation class.

    Raises:
        ValueError: If *state_repr* is not recognised.
    """
    cls = _REPR_MAP.get(state_repr)
    if cls is None:
        raise ValueError(
            f"Unknown state representation: {state_repr!r}. "
            f"Available: {list(_REPR_MAP.keys())}"
        )
    return cls()
