# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Kalman motion, missed-detection clocks and identities for McByte++ tracklets."""

import numpy as np

from trackers.utils.cmc import CMC
from trackers.utils.predict_timing import FIXED_RATE_TIMING, PredictTiming
from trackers.utils.scale_aware_tracklet import ScaleAwareNoiseTracklet
from trackers.utils.state_representations import BaseStateEstimator, XCYCWHStateEstimator


class McBytePlusPlusTracklet(ScaleAwareNoiseTracklet):
    """Maintain one object's motion estimate and lifecycle counters for McByte++.

    Uses ``XCYCWHStateEstimator`` (box center, width and height) by default,
    following the same motion model as ``McByteTracklet``. Both inherit the
    scale-aware noise calculations from ``ScaleAwareNoiseTracklet``:

    - Initial state covariance ``P`` and noise covariances ``Q`` and ``R`` are
      sized from the first detection's width and height.
    - Every ``predict`` rebuilds process noise ``Q`` from the current box size;
      every matched ``update`` rebuilds measurement noise ``R``.
    - Geometric state components are clamped after prediction and correction
      to keep the supported box representations valid.
    - ``predict`` advances missed-detection clocks. Unmatched tracks never
      receive ``update(None)``; a matched ``update(bbox)`` resets those clocks.
    - ``number_of_successful_updates`` starts at one for the initial box and
      counts all matched updates; a missed detection does not reset it.
    - ``apply_cmc`` transforms the motion state and covariance in place.

    ``tracker_id`` and ``public_id`` both start at ``-1``. The tracker assigns
    them when confirming a track. The internal ``tracker_id`` identifies mask
    ownership; ``public_id`` is exported to users. They currently receive the
    same value, but keeping them separate permits later identity mapping
    without moving masks between tracklets. This class neither runs EdgeTAM
    nor stores mask arrays; the tracker and mask backend handle those tasks.

    Args:
        initial_bbox: Initial detection box ``[x1, y1, x2, y2]`` in image pixels.
        state_estimator_class: Motion-state representation and Kalman estimator.
            Defaults to ``XCYCWHStateEstimator``.
    """

    def __init__(
        self,
        initial_bbox: np.ndarray,
        state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator,
    ) -> None:
        super().__init__(initial_bbox, state_estimator_class)
        self._configure_initial_noise(initial_bbox)
        self.number_of_successful_updates = 1
        self.public_id = -1

    def update(self, bbox: np.ndarray) -> None:
        """Correct the motion estimate using one matched detection box.

        Rebuild measurement noise from the current state, apply the Kalman
        correction, and clamp the supported geometric state components. Reset
        both ``time_since_update`` and ``time_since_update_seconds``, then
        increment the cumulative ``number_of_successful_updates`` counter.
        Identity assignment is left to the tracker.

        Only matched tracks call this method. An unmatched track skips it and
        keeps the missed-detection clocks advanced by ``predict``.

        Args:
            bbox: Matched detection box ``[x1, y1, x2, y2]`` in image pixels;
                ``None`` is not accepted.
        """
        self._refresh_measurement_noise_from_state()
        self.state_estimator.update(bbox)
        self._clamp_state_bbox()
        self.time_since_update = 0
        self.time_since_update_seconds = 0.0
        self.number_of_successful_updates += 1

    def predict(self, timing: PredictTiming = FIXED_RATE_TIMING) -> np.ndarray:
        """Predict the next box and advance age and missed-detection clocks.

        Rebuild process noise from the current box size, predict using the
        supplied Kalman time step, and clamp the supported geometric state
        components. Each call increments ``age`` and ``time_since_update`` by
        one. Timestamp mode also adds ``timing.elapsed_seconds`` to
        ``time_since_update_seconds``; fixed-rate mode resets that seconds
        counter to zero. The tracker uses these clocks to expire lost tracks.

        The tracker skips calling this method for duplicate or invalid
        timestamps. This method itself does not perform that timing check.

        Args:
            timing: Kalman step size in frame units, optional elapsed seconds
                and reference frame rate. Defaults to one fixed-rate frame.

        Returns:
            Predicted box ``[x1, y1, x2, y2]`` in image pixels, with shape ``(4,)``.
        """
        self._refresh_process_noise_from_state()
        self.state_estimator.predict(timing.frame_step, timing.frame_rate)
        self._clamp_state_bbox()
        self._advance_miss_clocks(timing)
        return self.state_estimator.state_to_bbox()

    def get_state_bbox(self) -> np.ndarray:
        """Decode the current motion estimate without advancing the filter.

        Returns:
            Current box ``[x1, y1, x2, y2]`` in image pixels, with shape ``(4,)``.
            This may be a prediction or a corrected estimate, depending on the
            most recent operation; it is not necessarily the last detection box.
        """
        return self.state_estimator.state_to_bbox()

    def apply_cmc(self, H: np.ndarray | None) -> None:
        """Apply a 2x3 affine camera-motion transform **in place**.

        Transforms the Kalman state and covariance without changing identities
        or lifecycle counters. Delegates to :meth:`CMC.apply_batch` with ``[self]`` as the
        tracklet list. See that method for full documentation of the
        transform convention, state-representation handling, and covariance
        update rules.

        Args:
            H: 2x3 affine transform matrix. If ``None``, this is a no-op.

        Examples:
            >>> import numpy as np
            >>> bbox = np.array([10.0, 20.0, 50.0, 80.0])
            >>> tracklet = McBytePlusPlusTracklet(bbox)
            >>> H = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float32)
            >>> tracklet.apply_cmc(H)
            >>> tracklet.apply_cmc(None)  # no-op
        """
        CMC.apply_batch(H, [self])
