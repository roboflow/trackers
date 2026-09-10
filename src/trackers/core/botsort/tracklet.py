# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.utils.cmc import CMC
from trackers.utils.predict_timing import FIXED_RATE_TIMING, PredictTiming
from trackers.utils.scale_aware_tracklet import ScaleAwareNoiseTracklet
from trackers.utils.state_representations import BaseStateEstimator, XCYCWHStateEstimator


class BoTSORTTracklet(ScaleAwareNoiseTracklet):
    """Tracklet for the BoT-SORT tracker.

    Uses ``XCYCWHStateEstimator`` (center + width/height) by default,
    mirroring the original BoT-SORT Kalman filter model.

    * **Scale-aware noise**: ``Q``, ``R`` and the initial ``P`` are computed
      from the current width / height of the tracked object each frame, so
      that uncertainty scales with object size (see
      :class:`~trackers.utils.scale_aware_tracklet.ScaleAwareNoiseTracklet`).
    * **Width / height clamping** after every predict and update step.
    * ``predict()`` increments ``time_since_update``: unmatched tracks are
      never explicitly fed ``update(None)``.
    * ``number_of_successful_updates`` counts every successful measurement
      update (never reset on a miss).
    * ``apply_cmc(affine_mtx)`` applies a 2x3 affine camera-motion transform to the
      internal Kalman state and covariance.
    """

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

    def update(self, bbox: np.ndarray) -> None:
        """Update tracklet with a new observation.

        In the BoT-SORT flow **only matched tracks** call ``update(bbox)`` with an actual bounding box.  Unmatched
        tracks simply skip ``update`` (their ``time_since_update`` is incremented in ``predict`` instead).
        """
        self._refresh_measurement_noise_from_state()
        self.state_estimator.update(bbox)
        self._clamp_state_bbox()
        self.time_since_update = 0
        self.time_since_update_seconds = 0.0
        self.number_of_successful_updates += 1

    def predict(self, timing: PredictTiming = FIXED_RATE_TIMING) -> np.ndarray:
        """Predict the next bounding-box position.

        Increments ``time_since_update`` to track how many frames have elapsed since the last matched measurement — this
        replaces the ``update(None)`` call used in ByteTrack/SORT.
        """
        self._refresh_process_noise_from_state()
        self.state_estimator.predict(timing.frame_step, timing.frame_rate)
        self._clamp_state_bbox()
        self._advance_miss_clocks(timing)
        return self.state_estimator.state_to_bbox()

    def get_state_bbox(self) -> np.ndarray:
        """Return the current bounding-box estimate in xyxy format."""
        return self.state_estimator.state_to_bbox()

    def apply_cmc(self, affine_mtx: np.ndarray | None) -> None:
        """Apply a 2x3 affine camera-motion transform **in place**.

        Delegates to :meth:`CMC.apply_batch` with ``[self]`` as the
        tracklet list. See that method for full documentation of the
        transform convention, state-representation handling, and covariance
        update rules.

        Args:
            affine_mtx: 2x3 affine transform matrix. If ``None``, this is a no-op.

        Examples:
            >>> import numpy as np
            >>> bbox = np.array([10.0, 20.0, 50.0, 80.0])
            >>> tracklet = BoTSORTTracklet(bbox)
            >>> affine_mtx = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float32)
            >>> tracklet.apply_cmc(affine_mtx)
            >>> tracklet.apply_cmc(None)  # no-op
        """
        CMC.apply_batch(affine_mtx, [self])
