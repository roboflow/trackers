# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from trackers.utils.converters import (
    xcycsr_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_xcycsr,
    xyxy_to_xywh,
)
from trackers.utils.kalman_filter import KalmanFilter
from trackers.utils.motion_models import KalmanMotionModel, init_constant_velocity_filter


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
        XCYCWH: Center-based representation with 8 state variables:
            `x_center`, `y_center` (box center), `w` (width), `h` (height),
            and velocities `vx`, `vy`, `vw`, `vh`. Unlike XCYCSR, both width
            and height carry independent velocity terms. Commonly used with
            BoT-SORT-style tracking, where callers may refresh scale-aware
            process and measurement noise separately based on the current w/h.
    """

    XCYCSR = "xcycsr"
    XYXY = "xyxy"
    XCYCWH = "xcycwh"


class BaseStateEstimator(ABC):
    """Abstract Kalman filter with a specific bounding box state representation.

    Wraps a `KalmanFilter` and provides a unified interface for
    bounding-box tracking regardless of the internal state encoding.
    Subclasses configure the filter dimensions, matrices, noise, and
    handle conversions between `[x1, y1, x2, y2]` bboxes and the
    internal state/measurement vectors.

    Variable frame rate: pass a larger ``frame_step`` to ``predict()`` after a
    gap. See ``docs/learn/dynamic-frame-rate.md``.

    Note:
        Noise matrices (R, Q, P) are not configured in ``_create_filter``
        and default to identity matrices. Callers must configure them via
        ``set_kf_covariances`` after construction for accurate tracking.
        Tracklet classes (``SORTTracklet``, ``ByteTrackTracklet``,
        ``OCSORTTracklet``) do this automatically via ``_configure_noise()``.
        If you instantiate a state estimator directly, call
        ``set_kf_covariances`` before the first ``predict``/``update``.

    Attributes:
        kf: The underlying Kalman filter instance.
        motion: Builds ``F`` and ``Q`` before each predict.
    """

    _POS_IDX: np.ndarray
    _VEL_IDX: np.ndarray

    def __init__(self, bbox: np.ndarray) -> None:
        """Initialise the filter with the first detection.

        Args:
            bbox: First detection `[x1, y1, x2, y2]`.
        """
        self.kf: KalmanFilter = self._create_filter(bbox)
        self.motion: KalmanMotionModel = KalmanMotionModel.from_filter(
            self.kf,
            self._POS_IDX,
            self._VEL_IDX,
        )

    def _init_cv_filter(self, dim_x: int, measurement: np.ndarray) -> KalmanFilter:
        """Create a constant-velocity filter with ``F(1)``, ``H = I``, and initial state."""
        return init_constant_velocity_filter(
            dim_x=dim_x,
            dim_z=4,
            pos_idx=self._POS_IDX,
            vel_idx=self._VEL_IDX,
            measurement=measurement,
        )

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

    def predict(self, frame_step: float = 1.0) -> None:
        """Predict one Kalman step, scaling F and Q by frame_step.

        Args:
            frame_step: Elapsed time in frame units; ``1.0`` = one nominal
                frame. Pass a larger value after a gap between updates so the
                filter extrapolates further and widens process noise accordingly.
        """
        self.clamp_velocity()
        self.motion.apply(self.kf, frame_step)
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
        self.motion.reset_cache()

    def set_kf_covariances(
        self,
        measurement_noise: np.ndarray | None = None,
        process_noise: np.ndarray | None = None,
        state_covariance: np.ndarray | None = None,
    ) -> None:
        """Set Kalman noise matrices.

        ``process_noise`` controls how much the state may drift per predict.
        Tracklets set it in ``_configure_noise()`` for the one-frame case.
        When passed here, the motion model stores it as the reference used at
        ``frame_step=1.0`` and as the starting point for gap scaling.

        Args:
            measurement_noise: Measurement noise covariance (trust in detections).
            process_noise: Process noise covariance (drift between detections).
            state_covariance: Initial state uncertainty.
        """
        if measurement_noise is not None:
            expected_shape = (self.kf.dim_z, self.kf.dim_z)
            if measurement_noise.shape != expected_shape:
                raise ValueError(f"measurement_noise must have shape {expected_shape}; got {measurement_noise.shape}.")
            self.kf.measurement_noise = measurement_noise
        if process_noise is not None:
            expected_shape = (self.kf.dim_x, self.kf.dim_x)
            if process_noise.shape != expected_shape:
                raise ValueError(f"process_noise must have shape {expected_shape}; got {process_noise.shape}.")
            self.kf.process_noise = process_noise
            self.motion.calibrate_from_process_noise(process_noise)
        if state_covariance is not None:
            expected_shape = (self.kf.dim_x, self.kf.dim_x)
            if state_covariance.shape != expected_shape:
                raise ValueError(f"state_covariance must have shape {expected_shape}; got {state_covariance.shape}.")
            self.kf.state_covariance = state_covariance


class XCYCSRStateEstimator(BaseStateEstimator):
    """Center-based Kalman filter with 7 state dimensions and 4 measurements.

    State vector contains `x_center`, `y_center` (box center), `scale` (area),
    `aspect_ratio` (width/height), and velocities `vx`, `vy`, `vs`. Aspect ratio
    is treated as constant (no velocity term), which works well for rigid objects
    that maintain their shape. Matches the representation used in the original
    SORT and OC-SORT papers.
    """

    # State layout: [xc, yc, s, r, vx, vy, vs]
    _POS_IDX = np.array([0, 1, 2], dtype=np.int64)
    _VEL_IDX = np.array([4, 5, 6], dtype=np.int64)

    def _create_filter(self, bbox: np.ndarray) -> KalmanFilter:
        return self._init_cv_filter(7, xyxy_to_xcycsr(bbox))

    def bbox_to_measurement(self, bbox: np.ndarray) -> np.ndarray:
        return xyxy_to_xcycsr(bbox)

    def state_to_bbox(self) -> np.ndarray:
        return xcycsr_to_xyxy(self.kf.state[:4].reshape((4,)))

    def clamp_velocity(self) -> None:
        if (self.kf.state[6] + self.kf.state[2]) <= 0:
            self.kf.state[6] = 0.0


class XCYCWHStateEstimator(BaseStateEstimator):
    """Center-width-height Kalman filter with 8 state dims and 4 measurements.

    State vector contains `x_center`, `y_center` (box center), `w` (width),
    `h` (height), and velocities `vx`, `vy`, `vw`, `vh`.  Unlike
    `XCYCSRStateEstimator`, both width and height have independent velocity
    terms and can change freely.

    This estimator only provides the coordinate-transform and filter-layout
    logic (F, H, conversions).  Noise tuning (Q, R, P) and any dynamic
    noise refresh are the responsibility of the tracklet that owns the
    estimator — exactly like `XYXYStateEstimator` and `XCYCSRStateEstimator`.
    """

    # State layout: [xc, yc, w, h, vx, vy, vw, vh]
    _POS_IDX = np.array([0, 1, 2, 3], dtype=np.int64)
    _VEL_IDX = np.array([4, 5, 6, 7], dtype=np.int64)

    def _create_filter(self, bbox: np.ndarray) -> KalmanFilter:
        return self._init_cv_filter(8, xyxy_to_xywh(bbox))

    def bbox_to_measurement(self, bbox: np.ndarray) -> np.ndarray:
        return xyxy_to_xywh(bbox)

    def state_to_bbox(self) -> np.ndarray:
        return xywh_to_xyxy(self.kf.state[:4].reshape((4,)))

    def clamp_velocity(self) -> None:
        pass


class XYXYStateEstimator(BaseStateEstimator):
    """Corner-based Kalman filter with 8 state dimensions and 4 measurements.

    State vector contains `x1`, `y1` (top-left corner), `x2`, `y2` (bottom-right
    corner), and independent velocities `vx1`, `vy1`, `vx2`, `vy2` for each
    coordinate. This allows the box shape to change over time, which may be
    better suited for non-rigid or deformable objects.
    """

    # State layout: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    _POS_IDX = np.array([0, 1, 2, 3], dtype=np.int64)
    _VEL_IDX = np.array([4, 5, 6, 7], dtype=np.int64)

    def _create_filter(self, bbox: np.ndarray) -> KalmanFilter:
        return self._init_cv_filter(8, bbox)

    def bbox_to_measurement(self, bbox: np.ndarray) -> np.ndarray:
        return bbox

    def state_to_bbox(self) -> np.ndarray:
        return self.kf.state[:4].reshape((4,))

    def clamp_velocity(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

_REPR_MAP: dict[StateRepresentation, type[BaseStateEstimator]] = {
    StateRepresentation.XCYCSR: XCYCSRStateEstimator,
    StateRepresentation.XYXY: XYXYStateEstimator,
    StateRepresentation.XCYCWH: XCYCWHStateEstimator,
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
        raise ValueError(f"Unknown state representation: {state_repr!r}. Available: {list(_REPR_MAP.keys())}")
    return cls(bbox)
