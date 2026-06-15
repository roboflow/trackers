# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.utils.base_tracklet import BaseTracklet
from trackers.utils.cmc import CMC
from trackers.utils.converters import xyxy_to_xywh
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCSRStateEstimator,
    XCYCWHStateEstimator,
    XYXYStateEstimator,
)


class BoTSORTTracklet(BaseTracklet):
    """Tracklet for the BoT-SORT tracker.

    Uses ``XCYCWHStateEstimator`` (center + width/height) by default,
    mirroring the original BoT-SORT Kalman filter model.

    * **Scale-aware noise**: ``Q``, ``R`` and the initial ``P`` are computed
      from the current width / height of the tracked object each frame, so
      that uncertainty scales with object size.
    * **Width / height clamping** after every predict and update step.
    * ``predict()`` increments ``time_since_update``: unmatched tracks are
      never explicitly fed ``update(None)``.
    * ``number_of_successful_updates`` counts every successful measurement
      update (never reset on a miss).
    * ``apply_cmc(H)`` applies a 2x3 affine camera-motion transform to the
      internal Kalman state and covariance.
    """

    # Noise sigma constants (scale-aware noise for BoT-SORT)
    _SIGMA_P: float = 0.05
    _SIGMA_V: float = 0.00625
    _SIGMA_M: float = 0.05

    def __init__(
        self,
        initial_bbox: np.ndarray,
        state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator,
    ) -> None:
        super().__init__(initial_bbox, state_estimator_class)
        self._configure_initial_noise(initial_bbox)
        # Count initial bbox as first successful update so that
        # number_of_successful_updates starts at 1.
        self.number_of_successful_updates = 1

    def _configure_initial_noise(self, bbox: np.ndarray) -> None:
        """Set initial P, Q, R based on the first detection's size."""
        measurement = xyxy_to_xywh(bbox)
        w, h = float(measurement[2]), float(measurement[3])
        self._set_scale_aware_noise(w, h, initial=True)

    def _set_scale_aware_noise(self, w: float, h: float, *, initial: bool = False) -> None:
        sp, sv, sm = self._SIGMA_P, self._SIGMA_V, self._SIGMA_M

        if isinstance(self.state_estimator, XCYCSRStateEstimator):
            s = np.sqrt(max(w * h, 1e-6))
            Q = np.diag(
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
            R = np.diag(
                [
                    (sm * w) ** 2,
                    (sm * h) ** 2,
                    (sm * s) ** 2,
                    (sm * 1.0) ** 2,
                ]
            )
        else:
            Q = np.diag(
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
            R = np.diag(
                [
                    (sm * w) ** 2,
                    (sm * h) ** 2,
                    (sm * w) ** 2,
                    (sm * h) ** 2,
                ]
            )

        if initial:
            if isinstance(self.state_estimator, XCYCSRStateEstimator):
                s = np.sqrt(max(w * h, 1e-6))
                P = np.diag(
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
                P = np.diag(
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
            self.state_estimator.set_kf_covariances(R=R, Q=Q, P=P)
        else:
            self.state_estimator.set_kf_covariances(R=R, Q=Q)

    def _refresh_noise_from_state(self) -> None:
        """Recompute Q and R from the current bbox size."""
        bbox = self.state_estimator.state_to_bbox()
        w = max(float(bbox[2] - bbox[0]), 1e-3)
        h = max(float(bbox[3] - bbox[1]), 1e-3)
        self._set_scale_aware_noise(w, h)

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
        kf_x = self.state_estimator.kf.x
        if isinstance(self.state_estimator, XYXYStateEstimator):
            self._clamp_xyxy_state(kf_x)
        elif isinstance(self.state_estimator, XCYCWHStateEstimator):
            self._clamp_xcycwh_state(kf_x)
        elif isinstance(self.state_estimator, XCYCSRStateEstimator):
            self._clamp_xcycsr_state(kf_x)

    def update(self, bbox: np.ndarray) -> None:
        """Update tracklet with a new observation.

        In the BoT-SORT flow **only matched tracks** call ``update(bbox)``
        with an actual bounding box.  Unmatched tracks simply skip
        ``update`` (their ``time_since_update`` is incremented in
        ``predict`` instead).
        """
        self._refresh_noise_from_state()
        self.state_estimator.update(bbox)
        self._clamp_state_bbox()
        self.time_since_update = 0
        self.number_of_successful_updates += 1

    def predict(self) -> np.ndarray:
        """Predict the next bounding-box position.

        Increments ``time_since_update`` to track how many frames have
        elapsed since the last matched measurement — this replaces the
        ``update(None)`` call used in ByteTrack/SORT.
        """
        self._refresh_noise_from_state()
        self.state_estimator.predict()
        self._clamp_state_bbox()
        self.age += 1
        self.time_since_update += 1
        return self.state_estimator.state_to_bbox()

    def get_state_bbox(self) -> np.ndarray:
        """Return the current bounding-box estimate in xyxy format."""
        return self.state_estimator.state_to_bbox()

    def apply_cmc(self, H: np.ndarray | None) -> None:
        """Apply a 2x3 affine camera-motion transform **in place**.

        Delegates to :meth:`CMC.apply_batch` with ``[self]`` as the
        tracklet list. See that method for full documentation of the
        transform convention, state-representation handling, and covariance
        update rules.

        Args:
            H: 2x3 affine transform matrix. If ``None``, this is a no-op.

        Examples:
            >>> import numpy as np
            >>> bbox = np.array([10.0, 20.0, 50.0, 80.0])
            >>> tracklet = BoTSORTTracklet(bbox)
            >>> H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float32)
            >>> tracklet.apply_cmc(H)
            >>> tracklet.apply_cmc(None)  # no-op
        """
        CMC.apply_batch(H, [self])
