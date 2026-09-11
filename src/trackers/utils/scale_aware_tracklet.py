# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.utils.base_tracklet import BaseTracklet
from trackers.utils.converters import xyxy_to_xywh
from trackers.utils.matrix import _diagonal_matrix
from trackers.utils.state_representations import (
    XCYCSRStateEstimator,
    XCYCWHStateEstimator,
    XYXYStateEstimator,
)


class ScaleAwareNoiseTracklet(BaseTracklet):
    """Base tracklet whose Kalman noise scales with the tracked box size.

    Sizing happens at two different cadences, so filter uncertainty tracks object scale:

    - ``P`` is sized **once**, from the first detection's width / height, via ``_configure_initial_noise``.
    - ``Q`` is rebuilt from the current state's width / height on every ``predict``.
    - ``R`` is rebuilt from the current state's width / height on every ``update``.

    Shared by ``BoTSORTTracklet`` and ``McByteTracklet``, which differ only in their ``update`` / ``predict`` /
    ``apply_cmc`` surface — not in this machinery.

    Subclasses still supply ``update``, ``predict`` and ``get_state_bbox`` (abstract on ``BaseTracklet``).
    """

    # Noise sigma constants (scale-aware noise).
    _SIGMA_P: float = 0.05
    _SIGMA_V: float = 0.00625
    _SIGMA_M: float = 0.05

    def _configure_initial_noise(self, bbox: np.ndarray) -> None:
        """Set initial P, Q, R based on the first detection's size."""
        measurement = xyxy_to_xywh(bbox)
        w, h = float(measurement[2]), float(measurement[3])
        self._set_scale_aware_noise(w, h)

    def _build_process_noise(self, w: float, h: float) -> np.ndarray:
        """Build the scale-aware process-noise covariance (Q) for the given box size."""
        sp, sv = self._SIGMA_P, self._SIGMA_V
        if isinstance(self.state_estimator, XCYCSRStateEstimator):
            s = np.sqrt(max(w * h, 1e-6))
            return _diagonal_matrix(
                [
                    (sp * w) ** 2,
                    (sp * h) ** 2,
                    (sp * s) ** 2,
                    (sp * 1.0) ** 2,
                    (sv * w) ** 2,
                    (sv * h) ** 2,
                    (sv * s) ** 2,
                ]
            )
        return _diagonal_matrix(
            [
                (sp * w) ** 2,
                (sp * h) ** 2,
                (sp * w) ** 2,
                (sp * h) ** 2,
                (sv * w) ** 2,
                (sv * h) ** 2,
                (sv * w) ** 2,
                (sv * h) ** 2,
            ]
        )

    def _build_measurement_noise(self, w: float, h: float) -> np.ndarray:
        """Build the scale-aware measurement-noise covariance (R) for the given box size."""
        sm = self._SIGMA_M
        if isinstance(self.state_estimator, XCYCSRStateEstimator):
            s = np.sqrt(max(w * h, 1e-6))
            return _diagonal_matrix([(sm * w) ** 2, (sm * h) ** 2, (sm * s) ** 2, (sm * 1.0) ** 2])
        return _diagonal_matrix([(sm * w) ** 2, (sm * h) ** 2, (sm * w) ** 2, (sm * h) ** 2])

    def _set_scale_aware_noise(self, w: float, h: float) -> None:
        """Set the initial Q, R and P from the first detection's size."""
        sp, sv = self._SIGMA_P, self._SIGMA_V
        Q = self._build_process_noise(w, h)
        R = self._build_measurement_noise(w, h)

        if isinstance(self.state_estimator, XCYCSRStateEstimator):
            s = np.sqrt(max(w * h, 1e-6))
            state_covariance = _diagonal_matrix(
                [
                    (2 * sp * w) ** 2,
                    (2 * sp * h) ** 2,
                    (2 * sp * s) ** 2,
                    (2 * sp * 1.0) ** 2,
                    (10 * sv * w) ** 2,
                    (10 * sv * h) ** 2,
                    (10 * sv * s) ** 2,
                ]
            )
        else:
            state_covariance = _diagonal_matrix(
                [
                    (2 * sp * w) ** 2,
                    (2 * sp * h) ** 2,
                    (2 * sp * w) ** 2,
                    (2 * sp * h) ** 2,
                    (10 * sv * w) ** 2,
                    (10 * sv * h) ** 2,
                    (10 * sv * w) ** 2,
                    (10 * sv * h) ** 2,
                ]
            )
        self.state_estimator.set_kf_covariances(measurement_noise=R, process_noise=Q, state_covariance=state_covariance)

    def _current_wh(self) -> tuple[float, float]:
        """Return the current box width/height, clamped away from zero.

        The ``XCYCWHStateEstimator`` path reconstructs each pair of corners before subtracting them. Reading the stored
        width and height directly would skip the center-dependent rounding performed by ``state_to_bbox()``. The fast
        path applies only when the estimator is exactly ``XCYCWHStateEstimator`` and its state has dtype ``float64``.
        Subclasses and states with other dtypes fall back to ``state_to_bbox()`` so overridden conversion behavior and
        the decoder's float64 corner arithmetic is preserved.
        """
        state = self.state_estimator.kf.state
        if type(self.state_estimator) is XCYCWHStateEstimator and state.dtype == np.float64:
            x_center = state[0, 0]
            y_center = state[1, 0]
            half_w = state[2, 0] * 0.5
            half_h = state[3, 0] * 0.5
            w = max(float((x_center + half_w) - (x_center - half_w)), 1e-3)
            h = max(float((y_center + half_h) - (y_center - half_h)), 1e-3)
            return w, h

        bbox = self.state_estimator.state_to_bbox()
        w = max(float(bbox[2] - bbox[0]), 1e-3)
        h = max(float(bbox[3] - bbox[1]), 1e-3)
        return w, h

    def _refresh_process_noise_from_state(self) -> None:
        """Recompute Q (process noise) from the current bbox size.

        Only ``predict()`` reads ``process_noise`` (see ``KalmanFilter.predict``), so this must not be called from
        ``update()`` — it would be overwritten by the next ``predict()`` before ever being read.
        """
        w, h = self._current_wh()
        self.state_estimator.set_kf_covariances(process_noise=self._build_process_noise(w, h))

    def _refresh_measurement_noise_from_state(self) -> None:
        """Recompute R (measurement noise) from the current bbox size.

        Only ``update()`` reads ``measurement_noise`` (see ``KalmanFilter.update``), so this must not be called from
        ``predict()`` — it would be overwritten by the next ``update()`` before ever being read.
        """
        w, h = self._current_wh()
        self.state_estimator.set_kf_covariances(measurement_noise=self._build_measurement_noise(w, h))

    @staticmethod
    def _clamp_xyxy_state(kf_x: np.ndarray) -> None:
        """Ensure XYXY state keeps valid box corners."""
        if kf_x[2, 0] <= kf_x[0, 0]:
            kf_x[2, 0] = kf_x[0, 0] + 1e-3
        if kf_x[3, 0] <= kf_x[1, 0]:
            kf_x[3, 0] = kf_x[1, 0] + 1e-3

    @staticmethod
    def _clamp_xcycwh_state(kf_x: np.ndarray) -> None:
        """Ensure XCYCWH state keeps positive width and height."""
        kf_x[2, 0] = max(kf_x[2, 0], 1e-3)
        kf_x[3, 0] = max(kf_x[3, 0], 1e-3)

    @staticmethod
    def _clamp_xcycsr_state(kf_x: np.ndarray) -> None:
        """Ensure XCYCSR state keeps positive scale and aspect ratio."""
        kf_x[2, 0] = max(kf_x[2, 0], 1e-3)
        kf_x[3, 0] = max(kf_x[3, 0], 1e-3)

    def _clamp_state_bbox(self) -> None:
        """Clamp geometric components based on active state representation."""
        kf_x = self.state_estimator.kf.state
        if isinstance(self.state_estimator, XYXYStateEstimator):
            self._clamp_xyxy_state(kf_x)
        elif isinstance(self.state_estimator, XCYCWHStateEstimator):
            self._clamp_xcycwh_state(kf_x)
        elif isinstance(self.state_estimator, XCYCSRStateEstimator):
            self._clamp_xcycsr_state(kf_x)
