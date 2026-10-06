# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.core.hybridsort.utils import _summed_corner_directions
from trackers.core.ocsort.tracklet import OCSORTTracklet
from trackers.utils.kalman_filter import KalmanFilter
from trackers.utils.motion_models import KalmanMotionModel, init_constant_velocity_filter
from trackers.utils.predict_timing import FIXED_RATE_TIMING, PredictTiming
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCSRStateEstimator,
)

# Confidence filter state layout: [confidence, confidence velocity].
_CONFIDENCE_POS_IDX = np.array([0], dtype=np.int64)
_CONFIDENCE_VEL_IDX = np.array([1], dtype=np.int64)


class HybridSORTTracklet(OCSORTTracklet):
    """Tracklet for Hybrid-SORT: an OC-SORT tracklet plus two weak cues.

    On top of OC-SORT's observation history and ORU (observation-centric
    re-update), each tracklet keeps:

    - **Tracklet Confidence Modeling (TCM).** A constant-velocity Kalman filter
      over detection confidence (state `[c, dc]`), frozen and replayed alongside
      the box filter during ORU, plus the last two observed confidences for a
      linear confidence trend. Confidence tends to drop as an object becomes
      occluded and recover as it reappears, so a detection whose confidence
      matches the tracklet's predicted confidence is more likely to be the same
      object.
    - **Robust OCM (ROCM) corner velocities.** The motion direction of each of
      the four box corners, each the sum of unit vectors from up to `delta_t`
      previous observations to the current one.

    Attributes:
        confidence: Confidence of the most recent matched detection.
        previous_confidence: Confidence of the matched detection before that, or
            `None` when the tracklet has one observation or missed its last frame.
        corner_velocities: Per-corner `[dy, dx]` directions of shape `(4, 2)`, or
            `None` until the tracklet has been matched twice.
    """

    def __init__(
        self,
        initial_bbox: np.ndarray,
        confidence: float,
        state_estimator_class: type[BaseStateEstimator] = XCYCSRStateEstimator,
        delta_t: int = 3,
    ) -> None:
        """Initialize tracklet with its first detection.

        Args:
            initial_bbox: Initial bounding box `[x1, y1, x2, y2]`.
            confidence: Confidence score of the initial detection.
            state_estimator_class: State estimator class used for the box
                Kalman filter. Defaults to `XCYCSRStateEstimator`.
            delta_t: Number of past observations used for velocity estimation.
        """
        super().__init__(initial_bbox, state_estimator_class=state_estimator_class, delta_t=delta_t)
        # As in the reference implementation, ORU only replays a gap between two
        # matched observations: a track lost before its first match has no
        # trajectory to re-update, so it is not frozen until it has been matched.
        self._observed = False
        self.confidence = float(confidence)
        self.previous_confidence: float | None = None
        self.corner_velocities: np.ndarray | None = None

        self._confidence_filter = self._create_confidence_filter(self.confidence)
        self._confidence_motion = KalmanMotionModel.from_filter(
            self._confidence_filter, _CONFIDENCE_POS_IDX, _CONFIDENCE_VEL_IDX
        )
        # Derive the gap-scaled (DWNA) noise from this filter's own one-frame Q; box
        # filters get the same calibration through ``set_kf_covariances``.
        self._confidence_motion.calibrate_from_process_noise(self._confidence_filter.process_noise)
        self._frozen_confidence_state: dict | None = None

    @staticmethod
    def _create_confidence_filter(confidence: float) -> KalmanFilter:
        """Create the confidence Kalman filter with Hybrid-SORT's noise settings.

        Mirrors the confidence rows of the reference 9-dimensional filter, whose
        block-diagonal matrices make it equivalent to a separate filter:
        measurement noise `10`, initial covariance `diag(10, 10000)` and process
        noise `diag(1, 1e-4)`.
        """
        confidence_filter = init_constant_velocity_filter(
            dim_x=2,
            dim_z=1,
            pos_idx=_CONFIDENCE_POS_IDX,
            vel_idx=_CONFIDENCE_VEL_IDX,
            measurement=np.array([confidence]),
        )
        confidence_filter.measurement_noise = np.array([[10.0]])
        confidence_filter.state_covariance = np.diag([10.0, 10000.0])
        confidence_filter.process_noise = np.diag([1.0, 1e-4])
        return confidence_filter

    @property
    def kalman_confidence(self) -> float:
        """Confidence predicted by the confidence Kalman filter (unclipped)."""
        return float(self._confidence_filter.state[0, 0])

    @property
    def linear_confidence(self) -> float:
        """Confidence extrapolated linearly from the last two observed confidences (unclipped)."""
        if self.previous_confidence is None:
            return self.confidence
        return 2.0 * self.confidence - self.previous_confidence

    def _freeze(self) -> None:
        """Save box and confidence filter states before the track is lost (ORU mechanism)."""
        super()._freeze()
        self._frozen_confidence_state = self._confidence_filter.get_state()

    def _replay_confidence(self, confidence: float, timing: PredictTiming) -> None:
        """Restore the frozen confidence filter and re-update it along a virtual trajectory.

        Linearly interpolates from the last observed confidence to the new one
        across the missed frames, exactly like OC-SORT's ORU does for the box.
        The caller applies the final real update.

        Args:
            confidence: Confidence of the detection that re-activates the track.
            timing: Predict timing carrying the actual elapsed frame step.
        """
        if self._frozen_confidence_state is None:
            return
        self._confidence_filter.set_state(self._frozen_confidence_state)
        self._confidence_motion.reset_cache()

        time_gap = self.time_since_update
        step = (confidence - self.confidence) / time_gap
        for i in range(time_gap):
            self._confidence_filter.update(np.array([[self.confidence + (i + 1) * step]]))
            if i < time_gap - 1:
                self._confidence_motion.apply(self._confidence_filter, timing.frame_step)
                self._confidence_filter.predict()
        self._frozen_confidence_state = None

    def _update_corner_velocities(self, bbox: np.ndarray) -> None:
        """Refresh per-corner directions from up to `delta_t` previous observations to `bbox`."""
        if len(self.observations) == 0:
            return
        previous_boxes = [
            self.observations[self.age - lag]
            for lag in range(1, self.delta_t + 1)
            if self.age - lag in self.observations
        ]
        if not previous_boxes:
            previous_boxes = [self.last_observation]
        self.corner_velocities = _summed_corner_directions(np.asarray(previous_boxes, dtype=np.float64), bbox)

    def update(
        self,
        bbox: np.ndarray,
        timing: PredictTiming = FIXED_RATE_TIMING,
        confidence: float = 1.0,
    ) -> None:
        """Update tracklet state with a new detection.

        Args:
            bbox: Bounding box `[x1, y1, x2, y2]`.
            timing: Predict timing for ORU replay sub-step scaling.
            confidence: Confidence score of the matched detection.
        """
        # Both must run before the OC-SORT update below resets the miss state and
        # appends `bbox` to the observation history.
        self._update_corner_velocities(bbox)
        if not self._observed:
            self._replay_confidence(confidence, timing)
        self._confidence_filter.update(np.array([[confidence]]))
        # Deliberate, as in the reference implementation (hybrid_sort.py): the first match after a gap copies the
        # stale pre-gap confidence into `previous_confidence`, so the linear trend straddles the gap instead of
        # restarting from a single observation. `break_confidence_trend` only clears it for the unmatched frames.
        self.previous_confidence = self.confidence
        self.confidence = float(confidence)

        super().update(bbox, timing)

    def break_confidence_trend(self) -> None:
        """Forget the previous confidence after a frame in which the track went unmatched.

        The linear confidence trend is only meaningful between consecutive matches; as in the reference implementation,
        the tracker calls this for every track left unmatched by an update.

        Note:
            This clears `previous_confidence` only for the unmatched frames themselves. The first match after a gap
            deliberately copies the stale pre-gap `confidence` back into `previous_confidence` (reference behaviour),
            so the linear trend of that match spans the gap.
        """
        self.previous_confidence = None

    def predict(self, timing: PredictTiming = FIXED_RATE_TIMING) -> np.ndarray:
        """Predict the next bounding box and confidence.

        Returns:
            Predicted bounding box `[x1, y1, x2, y2]`.
        """
        predicted_bbox = super().predict(timing)
        self._confidence_motion.apply(self._confidence_filter, timing.frame_step, timing.frame_rate)
        self._confidence_filter.predict()
        return predicted_bbox
