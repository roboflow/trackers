# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from trackers.utils.base_tracklet import BaseTracklet
from trackers.utils.cmc import CMC, _warp_kalman_states
from trackers.utils.converters import (
    xyxy_to_xcycsr,
)
from trackers.utils.predict_timing import FIXED_RATE_TIMING, PredictTiming
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCSRStateEstimator,
    XYXYStateEstimator,
)

#: Output norms at or below this are treated as zero when rescaling warped directions.
_DIRECTION_EPS = 1e-12


def _rotate_directions(directions: np.ndarray, rot_mtx: np.ndarray) -> np.ndarray:
    """Map OC-SORT direction vectors through the linear part of a camera-motion transform.

    Directions are stored ``[dy, dx]`` (see ``OCSORTTracklet._compute_velocity``). Each
    one is mapped as an ``(x, y)`` vector by ``rot_mtx`` and rescaled to its original
    norm, so a unit direction stays unit under rotation, zoom or shear, and an identity
    linear part (pure translation) returns it unchanged, bit for bit. Zero vectors stay
    zero.

    Args:
        directions: ``(N, 2)`` direction vectors ``[dy, dx]``.
        rot_mtx: 2x2 linear part of the affine transform.

    Returns:
        ``(N, 2)`` transformed direction vectors ``[dy, dx]``.

    Examples:
        >>> import numpy as np
        >>> rot_90 = np.array([[0.0, -1.0], [1.0, 0.0]])
        >>> _rotate_directions(np.array([[0.0, 1.0]]), rot_90).tolist()  # +x becomes +y
        [[1.0, 0.0]]
    """
    dy = directions[:, 0]
    dx = directions[:, 1]
    # Element-wise rather than a matmul, so each row's result does not depend on how
    # many rows are transformed together.
    new_dx = rot_mtx[0, 0] * dx + rot_mtx[0, 1] * dy
    new_dy = rot_mtx[1, 0] * dx + rot_mtx[1, 1] * dy
    norm_in = np.sqrt(dx * dx + dy * dy)
    norm_out = np.sqrt(new_dx * new_dx + new_dy * new_dy)
    scale = np.divide(norm_in, norm_out, out=np.zeros_like(norm_out), where=norm_out > _DIRECTION_EPS)
    return np.stack([new_dy * scale, new_dx * scale], axis=1)


def _warp_observation_boxes(
    boxes: np.ndarray,
    rot_mtx: np.ndarray,
    translation: np.ndarray,
    *,
    corner_wise: bool,
) -> np.ndarray:
    """Warp stored ``xyxy`` observations with the same model CMC applies to the Kalman state.

    ``CMC.apply_batch`` moves only the centre of centre-based states (``XCYCSR``,
    ``XCYCWH``) and keeps their size terms, while ``XYXY`` states are warped
    corner-wise into the enclosing axis-aligned box. Observations follow the same
    model, so OCR, ORU and the direction term never compare boxes warped two
    different ways (an enclosing box would also grow under every small rotation).

    Args:
        boxes: ``(N, 4)`` boxes ``[x1, y1, x2, y2]``.
        rot_mtx: 2x2 linear part of the affine transform.
        translation: 2-element translation of the affine transform.
        corner_wise: ``True`` for ``XYXY`` states (enclosing box of the warped
            corners); ``False`` to move each box by its centre's displacement and
            keep width and height (box size is not scale-compensated).

    Returns:
        ``(N, 4)`` warped boxes ``[x1, y1, x2, y2]``.

    Examples:
        >>> import numpy as np
        >>> rot_90 = np.array([[0.0, -1.0], [1.0, 0.0]])
        >>> box = np.array([[0.0, 0.0, 2.0, 4.0]])
        >>> _warp_observation_boxes(box, rot_90, np.zeros(2), corner_wise=False).tolist()
        [[-3.0, -1.0, -1.0, 3.0]]
        >>> _warp_observation_boxes(box, rot_90, np.zeros(2), corner_wise=True).tolist()
        [[-4.0, 0.0, 0.0, 2.0]]
    """
    if corner_wise:
        return np.stack(
            CMC.warp_xyxy_corners(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], rot_mtx, translation),
            axis=1,
        )
    centre_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centre_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    # Displacement (R - I) c + t of each centre, element-wise so that an identity linear
    # part shifts by exactly ``translation`` and rows do not depend on batch size.
    shift_x = (rot_mtx[0, 0] - 1.0) * centre_x + rot_mtx[0, 1] * centre_y + translation[0]
    shift_y = rot_mtx[1, 0] * centre_x + (rot_mtx[1, 1] - 1.0) * centre_y + translation[1]
    return boxes + np.stack([shift_x, shift_y, shift_x, shift_y], axis=1)


class OCSORTTracklet(BaseTracklet):
    """Tracklet for OC-SORT tracker with ORU (Observation-centric Re-Update).

    Manages a single tracked object with Kalman filter state estimation.
    Implements OC-SORT specific features: freeze/unfreeze for saving state
    before track is lost, virtual trajectory generation (ORU) for recovering
    lost tracks, and configurable state representation (XCYCSR, XYXY or XCYCWH).

    Attributes:
        age: Age of the tracklet in frames.
        kalman_filter: The Kalman filter wrapping the state representation.
        tracker_id: Unique identifier (-1 until track is mature).
        number_of_successful_consecutive_updates: Consecutive successful updates.
        time_since_update: Frames since last observation.
        last_observation: Last observed bounding box `[x1, y1, x2, y2]`.
        previous_to_last_observation: Second-to-last observation for velocity.
        observations: Dict mapping age to observed bbox for delta_t lookback.
        velocity: Normalized direction vector computed with delta_t lookback.
        delta_t: Number of timesteps back to look for velocity estimation.
    """

    def __init__(
        self,
        initial_bbox: np.ndarray,
        state_estimator_class: type[BaseStateEstimator] = XCYCSRStateEstimator,
        delta_t: int = 3,
    ) -> None:
        """Initialize tracklet with first detection.

        Args:
            initial_bbox: Initial bounding box `[x1, y1, x2, y2]`.
            state_estimator_class: State estimator class to use. Instantiated
                with *initial_bbox*. Defaults to
                `XCYCSRStateEstimator`.
            delta_t: Number of timesteps back to look for velocity estimation.
                Higher values use observations further in the past to estimate
                motion direction, providing more stable velocity estimates.
        """
        # Initialize state estimator (wraps KalmanFilter + state repr)
        super().__init__(initial_bbox, state_estimator_class)
        self._configure_noise()
        # Observation history for ORU and delta_t
        self.delta_t = delta_t
        self.last_observation = initial_bbox
        self.previous_to_last_observation: np.ndarray | None = None
        self.observations: dict[int, np.ndarray] = {}
        self.velocity: np.ndarray | None = None

        # ORU: saved state for freeze/unfreeze
        self._frozen_state: dict | None = None
        self._observed = True

    def _freeze(self) -> None:
        """Save Kalman filter state before track is lost (ORU mechanism)."""
        self._frozen_state = self.state_estimator.get_state()

    def _unfreeze(self, new_bbox: np.ndarray, timing: PredictTiming) -> None:
        """Restore state and apply virtual trajectory (ORU mechanism).

        Generates linear interpolation between last observation and new
        detection, then re-updates the Kalman filter through this virtual
        trajectory. Each sub-step uses ``timing.frame_step / time_gap`` so
        the replayed predictions scale correctly in variable-FPS mode.

        Args:
            new_bbox: New observation bounding box `[x1, y1, x2, y2]`.
            timing: Predict timing carrying the actual elapsed frame step.
        """
        if self._frozen_state is None:
            return

        # Restore to frozen state
        self.state_estimator.set_state(self._frozen_state)

        time_gap = self.time_since_update
        sub_step = timing.frame_step
        # this is oc-sort specific
        if isinstance(self.state_estimator, XCYCSRStateEstimator):
            self._unfreeze_xcycsr(new_bbox, time_gap, sub_step)
        else:
            # Every other estimator: interpolate linearly in xyxy, then encode per estimator.
            self._unfreeze_linear(new_bbox, time_gap, sub_step)

        self._frozen_state = None

    def _unfreeze_xcycsr(self, new_bbox: np.ndarray, time_gap: int, sub_step: float) -> None:
        """ORU interpolation for XCYCSR representation.

        Generates time_gap predict+update cycles with virtual observations interpolated from the last observation to the
        new bbox. The interpolation factors go from 0 to (time_gap-1)/time_gap. The caller is responsible for the final
        real update at factor 1.0.
        """
        # Convert to (x, y, s, r) format
        last_xcycsr = xyxy_to_xcycsr(self.last_observation)
        new_xcycsr = xyxy_to_xcycsr(new_bbox)

        # Convert s, r back to w, h for interpolation
        x1, y1, s1, r1 = last_xcycsr
        w1 = np.sqrt(s1 * r1)
        h1 = np.sqrt(s1 / r1) if r1 != 0 else np.float64(0.0)

        x2, y2, s2, r2 = new_xcycsr
        w2 = np.sqrt(s2 * r2)
        h2 = np.sqrt(s2 / r2) if r2 != 0 else np.float64(0.0)

        # Linear interpolation deltas
        dx = (x2 - x1) / time_gap
        dy = (y2 - y1) / time_gap
        dw = (w2 - w1) / time_gap
        dh = (h2 - h1) / time_gap

        for i in range(time_gap):
            x = x1 + (i + 1) * dx
            y = y1 + (i + 1) * dy
            w = w1 + (i + 1) * dw
            h = h1 + (i + 1) * dh

            # Convert back to (x, y, s, r)
            s = w * h
            r = w / h
            virtual_obs = np.array([x, y, s, r]).reshape((4, 1))

            self.state_estimator.kf.update(virtual_obs)
            if i < time_gap - 1:
                self.state_estimator.predict(sub_step)

    def _unfreeze_linear(self, new_bbox: np.ndarray, time_gap: int, sub_step: float) -> None:
        """ORU interpolation for every estimator except XCYCSR (XYXY, XCYCWH, ...).

        Same pattern as XCYCSR: time_gap predict+update cycles with factors
        0 to (time_gap-1)/time_gap. Caller does the final real update. Boxes are
        interpolated linearly in xyxy, then each one is encoded with
        ``bbox_to_measurement`` so the virtual observation lives in the filter's
        measurement space.
        """
        last_xyxy = self.last_observation
        new_xyxy = new_bbox

        # Linear interpolation deltas for each coordinate
        delta = (new_xyxy - last_xyxy) / time_gap

        for i in range(time_gap):
            virtual_bbox = last_xyxy + (i + 1) * delta
            virtual_obs = self.state_estimator.bbox_to_measurement(virtual_bbox).reshape((4, 1))

            self.state_estimator.kf.update(virtual_obs)
            if i < time_gap - 1:
                self.state_estimator.predict(sub_step)

    def get_k_previous_obs(self) -> np.ndarray | None:
        """Get observation from delta_t steps ago.

        Looks back up to delta_t timesteps in the observation history.
        Falls back to the most recent observation if none found in the window.

        Returns:
            The observation from delta_t steps ago, or most recent if not found,
            or None if no observations exist.
        """
        if len(self.observations) == 0:
            return None
        for i in range(self.delta_t):
            dt = self.delta_t - i
            if self.age - dt in self.observations:
                return self.observations[self.age - dt]
        max_age = max(self.observations.keys())
        return self.observations[max_age]

    @staticmethod
    def _compute_velocity(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
        """Compute normalized direction vector between two bounding box centers.

        Args:
            bbox1: First bounding box `[x1, y1, x2, y2]`.
            bbox2: Second bounding box `[x1, y1, x2, y2]`.

        Returns:
            Normalized direction vector [dy, dx].
        """
        cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
        cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
        speed = np.array([cy2 - cy1, cx2 - cx1])
        norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
        return speed / norm

    def update(self, bbox: np.ndarray, timing: PredictTiming = FIXED_RATE_TIMING) -> None:
        """Update tracklet state with a new bounding-box observation.

        Handles ORU: if the track was lost and is now observed again,
        generates a virtual trajectory to smooth the transition.
        Computes velocity using the observation from delta_t steps ago.

        Args:
            bbox: Bounding box `[x1, y1, x2, y2]`.
            timing: Predict timing for ORU replay sub-step scaling.
        """
        # Compute velocity only after the track has been observed at least once
        # (matches original OC-SORT: velocity is None until 2nd match)
        previous_box = self.get_k_previous_obs()
        if previous_box is not None:
            self.velocity = self._compute_velocity(previous_box, bbox)

        # Check if we need to unfreeze (was lost, now observed)
        if not self._observed and self._frozen_state is not None:
            self._unfreeze(bbox, timing)

        # Update KF with the real observation
        # (after ORU this is the final update at the correct time step;
        #  without ORU this is the normal measurement update)
        self.state_estimator.update(bbox)

        self._observed = True
        self.time_since_update = 0
        self.time_since_update_seconds = 0.0
        self.number_of_successful_consecutive_updates += 1
        self.previous_to_last_observation = self.last_observation
        self.last_observation = bbox
        self.observations[self.age] = bbox
        # Prune entries beyond the delta_t lookback window to bound memory.
        cutoff = self.age - self.delta_t
        for key in [k for k in self.observations if k < cutoff]:
            del self.observations[key]

    def predict(self, timing: PredictTiming = FIXED_RATE_TIMING) -> np.ndarray:
        """Predict next bounding box position.

        Note:
            ORU virtual-trajectory sub-stepping inside ``_unfreeze_*`` still
            uses unit-frame Kalman steps; gap length follows ``time_since_update``
            in frame counts, not wall-clock seconds.

        Returns:
            Predicted bounding box `[x1, y1, x2, y2]`.
        """
        # Freeze KF state on the first miss. At the start of predict,
        # time_since_update reflects last frame: if it is already > 0 and
        # _observed is still True, the track went unmatched last frame and
        # this is the earliest point we can act on that information while
        # capturing the same frozen KF state as miss() would.
        if self._observed and self.time_since_update > 0:
            self._freeze()
            self._observed = False

        self.state_estimator.predict(timing.frame_step, timing.frame_rate)

        if self.time_since_update > 0:
            self.number_of_successful_consecutive_updates = 0

        self._advance_miss_clocks(timing)
        return self.state_estimator.state_to_bbox()

    def apply_camera_motion(self, affine_mtx: np.ndarray) -> None:
        """Warp the stored observations and the ORU frozen state by a camera-motion transform.

        ``CMC.apply_batch`` transforms the live Kalman state; this keeps the geometry
        OC-SORT stores beside it in the same, current-frame coordinates: the
        observations used by OCR, velocity estimation and the direction-consistency
        term, and the frozen filter state that ORU restores when a lost track is
        matched again. Boxes are warped with the same model as the Kalman state (see
        ``_warp_observation_boxes``): for the centre-based estimators (XCYCSR,
        XCYCWH) only the centre moves and width and height are kept, so box size
        is not scale-compensated and camera zoom is absorbed by the next matched
        detection; for XYXY the corners are warped and replaced by their enclosing
        axis-aligned box. The stored unit direction ``velocity`` is mapped through
        the linear part of the transform too, because the direction-consistency term
        reads it before the next match re-estimates it (see ``_rotate_directions``).

        Args:
            affine_mtx: 2x3 affine transform returned by ``CMC.estimate()``.
        """
        rot_mtx = affine_mtx[:2, :2].astype(np.float64)
        translation = affine_mtx[:2, 2].astype(np.float64)

        if self.velocity is not None:
            self.velocity = _rotate_directions(self.velocity[np.newaxis], rot_mtx)[0]

        ages = list(self.observations)
        boxes = [self.last_observation, *(self.observations[age] for age in ages)]
        if self.previous_to_last_observation is not None:
            boxes.append(self.previous_to_last_observation)
        stacked = np.asarray(boxes, dtype=np.float64)
        is_xyxy = isinstance(self.state_estimator, XYXYStateEstimator)
        warped = _warp_observation_boxes(stacked, rot_mtx, translation, corner_wise=is_xyxy)
        self.last_observation = warped[0]
        self.observations = {age: warped[i + 1] for i, age in enumerate(ages)}
        if self.previous_to_last_observation is not None:
            self.previous_to_last_observation = warped[-1]

        if self._frozen_state is not None:
            states, covariances = _warp_kalman_states(
                self._frozen_state["state"].reshape(1, -1).astype(np.float64),
                self._frozen_state["state_covariance"][np.newaxis].astype(np.float64),
                affine_mtx,
                is_xyxy=is_xyxy,
            )
            self._frozen_state["state"] = states[0].reshape(-1, 1)
            self._frozen_state["state_covariance"] = covariances[0]

    def get_state_bbox(self) -> np.ndarray:
        """Get current bounding box estimate from Kalman filter.

        Returns:
            Current bounding box estimate `[x1, y1, x2, y2]`.
        """
        return self.state_estimator.state_to_bbox()

    def _configure_noise(self) -> None:
        """Configure Kalman filter noise matrices (OC-SORT paper tuning)."""
        kf = self.state_estimator.kf
        measurement_noise = kf.measurement_noise
        state_covariance = kf.state_covariance
        process_noise = kf.process_noise
        if isinstance(self.state_estimator, XCYCSRStateEstimator):
            measurement_noise[2:, 2:] *= 10.0
            state_covariance[4:, 4:] *= 1000.0
            state_covariance *= 10.0
            process_noise[-1, -1] *= 0.01
            process_noise[4:, 4:] *= 0.01
        else:
            # XYXY / XCYCWH: same velocity uncertainty scaling
            state_covariance[4:, 4:] *= 1000.0
            state_covariance *= 10.0
            process_noise[4:, 4:] *= 0.01
        self.state_estimator.set_kf_covariances(
            measurement_noise=measurement_noise,
            process_noise=process_noise,
            state_covariance=state_covariance,
        )

    def resolve_tracker_id(
        self,
        minimum_consecutive_frames: int,
        frame_count: int,
        allocate_tracker_id: Callable[[], int],
    ) -> int:
        """Resolve the tracker ID for the current tracklet state.

        Assigns a new unique ID if the tracklet is mature but hasn't been
        assigned one yet. Returns -1 for immature tracklets.

        Args:
            minimum_consecutive_frames: Frames required for track maturity.
            frame_count: Current frame number in tracking process.
            allocate_tracker_id: Zero-argument callable returning a unique int ID;
                called at most once per tracklet, when it first becomes mature.

        Returns:
            Integer tracker ID, or -1 for immature tracks.
        """
        is_mature = self.number_of_successful_consecutive_updates >= minimum_consecutive_frames
        if frame_count <= minimum_consecutive_frames:
            if self.time_since_update == 0:
                if self.tracker_id == -1:
                    self.tracker_id = allocate_tracker_id()
                return self.tracker_id
        else:
            if is_mature:
                if self.tracker_id == -1:
                    self.tracker_id = allocate_tracker_id()
                return self.tracker_id
        return -1
