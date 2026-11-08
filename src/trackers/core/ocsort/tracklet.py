# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable, Sequence

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


def _warp_velocities(tracklets: Sequence[OCSORTTracklet], rot_mtx: np.ndarray) -> None:
    """Rotate the stored direction ``velocity`` of every tracklet that has one, batched per dtype.

    ``_rotate_directions`` computes the input norm in the input dtype, so a ``float32``
    direction (from ``float32`` detection boxes) stacked with ``float64`` ones would be
    promoted and rounded differently than when rotated on its own. Grouping by dtype
    keeps every row bit-identical to a per-tracklet call.

    Args:
        tracklets: Tracklets whose ``velocity`` is rotated in place (``None`` is skipped).
        rot_mtx: 2x2 linear part of the affine transform.
    """
    groups: dict[np.dtype, tuple[list[OCSORTTracklet], list[np.ndarray]]] = {}
    for tracklet in tracklets:
        velocity = tracklet.velocity
        if velocity is not None:
            members, directions = groups.setdefault(velocity.dtype, ([], []))
            members.append(tracklet)
            directions.append(velocity)
    for members, directions in groups.values():
        rotated = _rotate_directions(np.stack(directions), rot_mtx)
        for tracklet, direction in zip(members, rotated, strict=True):
            tracklet.velocity = direction


def _warp_observation_histories(
    tracklets: Sequence[OCSORTTracklet],
    rot_mtx: np.ndarray,
    translation: np.ndarray,
    *,
    corner_wise: bool,
) -> None:
    """Warp every tracklet's stored observations with one ``_warp_observation_boxes`` call.

    Boxes are gathered in a fixed per-tracklet order (``last_observation``, the
    ``observations`` window in insertion order, then ``previous_to_last_observation``
    when set), warped together and scattered back in the same order. Both warp paths
    are row-independent, so each row matches a per-tracklet call bit for bit.

    Args:
        tracklets: Tracklets whose observations are replaced by their warped copies.
        rot_mtx: 2x2 linear part of the affine transform.
        translation: 2-element translation of the affine transform.
        corner_wise: Forwarded to ``_warp_observation_boxes`` (``True`` for ``XYXY``).
    """
    boxes: list[np.ndarray] = []
    for tracklet in tracklets:
        boxes.append(tracklet.last_observation)
        boxes.extend(tracklet.observations.values())
        if tracklet.previous_to_last_observation is not None:
            boxes.append(tracklet.previous_to_last_observation)
    warped = _warp_observation_boxes(np.asarray(boxes, dtype=np.float64), rot_mtx, translation, corner_wise=corner_wise)
    rows = iter(warped)
    for tracklet in tracklets:
        tracklet.last_observation = next(rows)
        tracklet.observations = {age: next(rows) for age in tracklet.observations}
        if tracklet.previous_to_last_observation is not None:
            tracklet.previous_to_last_observation = next(rows)


def _warp_frozen_states(tracklets: Sequence[OCSORTTracklet], affine_mtx: np.ndarray, *, is_xyxy: bool) -> None:
    """Warp the ORU frozen filter state of every lost tracklet with one ``_warp_kalman_states`` call.

    Args:
        tracklets: Tracklets whose ``_frozen_state`` (when set) is warped in place.
        affine_mtx: 2x3 affine transform returned by ``CMC.estimate()``.
        is_xyxy: Whether the states use the ``XYXY`` layout.
    """
    frozen_states = [state for tracklet in tracklets if (state := tracklet._frozen_state) is not None]
    if not frozen_states:
        return
    # Not bit-identical to warping each frozen state on its own: for the centre-based
    # layouts ``_warp_kalman_states`` maps the centre and its velocity with one 2-D
    # ``(K, 2) @ (2, 2)`` product, and the BLAS kernel picked for it depends on K (a
    # single row and a stacked batch round differently in the last ulp; measured
    # relative deviation below 1e-14, so ``np.allclose(rtol=1e-12, atol=0)`` holds).
    # The live states already take this batched path in ``CMC.apply_batch``, so the
    # frozen states now follow the same arithmetic. The covariance product and the
    # ``XYXY`` corner warp are stacked per-slice matmuls and stay bit-identical.
    states, covariances = _warp_kalman_states(
        np.array([state["state"].reshape(-1) for state in frozen_states], dtype=np.float64),
        np.array([state["state_covariance"] for state in frozen_states], dtype=np.float64),
        affine_mtx,
        is_xyxy=is_xyxy,
    )
    for frozen, warped_state, warped_covariance in zip(frozen_states, states, covariances, strict=True):
        frozen["state"] = warped_state.reshape(-1, 1)
        frozen["state_covariance"] = warped_covariance


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
        This is the single-tracklet form of ``apply_camera_motion_batch``.

        Args:
            affine_mtx: 2x3 affine transform returned by ``CMC.estimate()``.
        """
        self.apply_camera_motion_batch([self], affine_mtx)

    @staticmethod
    def apply_camera_motion_batch(tracklets: Sequence[OCSORTTracklet], affine_mtx: np.ndarray) -> None:
        """Warp the stored observations and ORU frozen states of many tracklets at once.

        Same per-tracklet result as ``apply_camera_motion``, with the linear part and
        translation derived once and each kind of stored geometry (observation boxes,
        direction vectors, frozen filter states) gathered into one array, warped by a
        single call and scattered back, instead of one round of NumPy calls per
        tracklet. Observations and directions match the per-tracklet warp bit for bit;
        frozen states can differ from it in the last ulp (see ``_warp_frozen_states``).

        Args:
            tracklets: Tracklets sharing one state estimator type; empty is a no-op.
            affine_mtx: 2x3 affine transform returned by ``CMC.estimate()``.

        Raises:
            TypeError: If the tracklets use different state estimator types, since
                one warp model is chosen for the whole batch.

        Examples:
            >>> import numpy as np
            >>> track = OCSORTTracklet(np.array([10.0, 20.0, 50.0, 80.0]))
            >>> shift = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
            >>> OCSORTTracklet.apply_camera_motion_batch([track], shift)
            >>> track.last_observation.tolist()
            [15.0, 17.0, 55.0, 77.0]
        """
        if len(tracklets) == 0:
            return
        estimator_type = type(tracklets[0].state_estimator)
        mismatch = next((t for t in tracklets if type(t.state_estimator) is not estimator_type), None)
        if mismatch is not None:
            raise TypeError(
                "OCSORTTracklet.apply_camera_motion_batch requires homogeneous state types; "
                f"got {estimator_type.__name__!r} and {type(mismatch.state_estimator).__name__!r}."
            )
        is_xyxy = issubclass(estimator_type, XYXYStateEstimator)
        rot_mtx = affine_mtx[:2, :2].astype(np.float64)
        translation = affine_mtx[:2, 2].astype(np.float64)

        _warp_velocities(tracklets, rot_mtx)
        _warp_observation_histories(tracklets, rot_mtx, translation, corner_wise=is_xyxy)
        _warp_frozen_states(tracklets, affine_mtx, is_xyxy=is_xyxy)

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
