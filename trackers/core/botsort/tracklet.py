# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trackers.utils.base_tracklet import BaseTracklet
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCWHStateEstimator,
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

    count_id: int = 0

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
        measurement = XCYCWHStateEstimator.xyxy_to_xywh(bbox)
        w, h = float(measurement[2]), float(measurement[3])
        self._set_scale_aware_noise(w, h, initial=True)

    def _set_scale_aware_noise(
        self, w: float, h: float, *, initial: bool = False
    ) -> None:
        sp, sv, sm = self._SIGMA_P, self._SIGMA_V, self._SIGMA_M

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
        """Recompute Q and R from the current w/h in the Kalman state."""
        kf = self.state_estimator.kf
        w = max(float(kf.x[2, 0]), 1e-3)
        h = max(float(kf.x[3, 0]), 1e-3)
        self._set_scale_aware_noise(w, h)

    @staticmethod
    def _clamp_wh(kf_x: np.ndarray) -> None:
        """Ensure width and height stay positive."""
        kf_x[2, 0] = max(kf_x[2, 0], 1e-3)
        kf_x[3, 0] = max(kf_x[3, 0], 1e-3)

    def update(self, bbox: np.ndarray) -> None:
        """Update tracklet with a new observation.

        In the BoT-SORT flow **only matched tracks** call ``update(bbox)``
        with an actual bounding box.  Unmatched tracks simply skip
        ``update`` (their ``time_since_update`` is incremented in
        ``predict`` instead).
        """
        self._refresh_noise_from_state()
        self.state_estimator.update(bbox)
        self._clamp_wh(self.state_estimator.kf.x)
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
        self._clamp_wh(self.state_estimator.kf.x)
        self.age += 1
        self.time_since_update += 1
        return self.state_estimator.state_to_bbox()

    def get_state_bbox(self) -> np.ndarray:
        """Return the current bounding-box estimate in xyxy format."""
        return self.state_estimator.state_to_bbox()

    def apply_cmc(self, H: np.ndarray | None) -> None:
        """Apply a 2x3 affine camera-motion transform **in place**.

        The transform follows the convention ``x' = R @ x + t`` where
        ``R = H[:2, :2]`` and ``t = H[:2, 2]``.

        For the XCYCWH state ``[xc, yc, w, h, vxc, vyc, vw, vh]``:
          * Centre position ``[xc, yc]``  → ``R @ [xc, yc] + t``
          * Centre velocity ``[vxc, vyc]`` → ``R @ [vxc, vyc]``
          * Width / height and their velocities are **not** transformed.

        The covariance ``P`` is updated as ``P = A @ P @ A.T`` where ``A``
        embeds ``R`` in the position and velocity blocks.
        """
        if H is None:
            return

        kf = self.state_estimator.kf
        R = H[:2, :2].astype(np.float64)
        t = H[:2, 2].astype(np.float64)

        x = kf.x.reshape(-1)
        x[0:2] = R @ x[0:2] + t
        x[4:6] = R @ x[4:6]
        kf.x = x.reshape(-1, 1)

        A = np.eye(kf.x.shape[0], dtype=np.float64)
        A[0:2, 0:2] = R
        A[4:6, 4:6] = R
        kf.P = A @ kf.P @ A.T

    @staticmethod
    def apply_cmc_batch(
        tracklets: Sequence[BoTSORTTracklet], H: np.ndarray | None
    ) -> None:
        """Apply a 2x3 affine camera-motion transform to all tracklets at once.

        Vectorised replacement for calling :meth:`apply_cmc` in a loop.
        State vectors are stacked into a single ``(N, dim)`` matrix and
        covariance matrices into ``(N, dim, dim)`` so that the rotation,
        translation and covariance transforms are pure batch numpy ops.

        Args:
            tracklets: Sequence of tracklets to transform **in place**.
            H: 2x3 affine transform ``[R | t]``.
        """
        if H is None or len(tracklets) == 0:
            return

        R = H[:2, :2].astype(np.float64)
        t = H[:2, 2].astype(np.float64)

        dim = tracklets[0].state_estimator.kf.x.shape[0]

        # Stack states (N, dim) and covariances (N, dim, dim)
        states = np.array([trk.state_estimator.kf.x.reshape(-1) for trk in tracklets])
        Ps = np.array([trk.state_estimator.kf.P for trk in tracklets])

        # Batch-transform centre positions: x' = x @ R.T + t
        states[:, 0:2] = states[:, 0:2] @ R.T + t
        # Batch-transform centre velocities: v' = v @ R.T
        states[:, 4:6] = states[:, 4:6] @ R.T

        # Build 8x8 rotation-embedding matrix once
        A = np.eye(dim, dtype=np.float64)
        A[0:2, 0:2] = R
        A[4:6, 4:6] = R

        # Batch covariance: P' = A @ P @ A.T  ->  (8,8) @ (N,8,8) @ (8,8)
        Ps = A @ Ps @ A.T

        # Write back
        for i, trk in enumerate(tracklets):
            trk.state_estimator.kf.x = states[i].reshape(-1, 1)
            trk.state_estimator.kf.P = Ps[i]
