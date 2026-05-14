# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import ClassVar

import numpy as np
import supervision as sv

from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.utils.iou import BIoU
from trackers.utils.state_representations import BaseStateEstimator, XCYCWHStateEstimator


class CBIoUTracker(BoTSORTTracker):
    """BoT-SORT with CMC disabled and Buffered IoU (BIoU) association.

    CBIoU is identical to :class:`~trackers.core.botsort.tracker.BoTSORTTracker`
    with two fixed differences:

    1. **Camera Motion Compensation is permanently off.** This makes the
       tracker faster and avoids relying on frame pixel data, which is
       convenient when only detection files are available (e.g. standard
       MOT benchmarks).
    2. **BIoU replaces standard IoU** for all association steps. Each
       bounding box is expanded by ``buffer_ratio`` relative to its own
       width/height before IoU is computed, giving the matcher more
       tolerance for small localization gaps between the Kalman prediction
       and the incoming detection.

    Args:
        lost_track_buffer: Time buffer (in frames at 30 FPS) for keeping
            lost tracks alive before deletion. Scaled by ``frame_rate``.
        frame_rate: Video frame rate used to scale the lost track buffer.
        track_activation_threshold: Minimum detection confidence to spawn
            a new track.
        minimum_consecutive_frames: Number of successful updates required
            before assigning a stable track ID.
        minimum_iou_threshold_first_assoc: Minimum fused similarity to
            accept an association during the first association step.
        minimum_iou_threshold_second_assoc: Minimum fused similarity to
            accept an association during the second association step.
        minimum_iou_threshold_unconfirmed_assoc: Minimum fused similarity
            to accept a match between an unconfirmed track and a remaining
            high-confidence detection.
        high_conf_det_threshold: Confidence threshold that splits
            detections into high / low confidence groups.
        instant_first_frame_activation: If ``True`` (default), tracks
            spawned on the very first frame receive a real tracker ID
            immediately.
        state_estimator_class: State estimator class for tracklets.
            Defaults to ``XCYCWHStateEstimator``.
        buffer_ratio: Non-negative relative margin by which each bounding
            box is expanded before IoU is computed. ``0.0`` recovers
            standard IoU exactly; larger values tolerate wider localization
            gaps. Forwarded to :class:`~trackers.utils.iou.BIoU`.

    Notes:
        - CMC parameters (``enable_cmc``, ``cmc_method``, ``cmc_downscale``)
          are intentionally absent from this class's signature — CMC is
          always disabled.
        - Passing a ``frame`` argument to :meth:`update` emits a
          ``UserWarning`` because no CMC processing takes place.
    """

    tracker_id = "cbiou"
    search_space: ClassVar[dict[str, dict]] = {
        "lost_track_buffer": {"type": "randint", "range": [10, 91]},
        "track_activation_threshold": {"type": "uniform", "range": [0.1, 0.9]},
        "minimum_iou_threshold_first_assoc": {"type": "uniform", "range": [0.05, 0.7]},
        "minimum_iou_threshold_second_assoc": {"type": "uniform", "range": [0.05, 0.7]},
        "minimum_iou_threshold_unconfirmed_assoc": {
            "type": "uniform",
            "range": [0.05, 0.7],
        },
        "high_conf_det_threshold": {"type": "uniform", "range": [0.3, 0.8]},
        "minimum_consecutive_frames": {"type": "randint", "range": [1, 4]},
        "buffer_ratio": {"type": "uniform", "range": [0.0, 0.5]},
    }

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold_first_assoc: float = 0.2,
        minimum_iou_threshold_second_assoc: float = 0.5,
        minimum_iou_threshold_unconfirmed_assoc: float = 0.3,
        high_conf_det_threshold: float = 0.6,
        instant_first_frame_activation: bool = True,
        state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator,
        buffer_ratio: float = 0.1,
    ) -> None:
        super().__init__(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold_first_assoc=minimum_iou_threshold_first_assoc,
            minimum_iou_threshold_second_assoc=minimum_iou_threshold_second_assoc,
            minimum_iou_threshold_unconfirmed_assoc=minimum_iou_threshold_unconfirmed_assoc,
            high_conf_det_threshold=high_conf_det_threshold,
            enable_cmc=False,
            instant_first_frame_activation=instant_first_frame_activation,
            state_estimator_class=state_estimator_class,
            iou=BIoU(buffer_ratio=buffer_ratio),
        )
        self.buffer_ratio = buffer_ratio

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
    ) -> sv.Detections:
        """Update the tracker with detections from the current frame.

        Args:
            detections: Supervision detections for the current frame.
            frame: Unused — CBIoU never performs CMC. Passing a non-``None``
                value emits a ``UserWarning``.

        Returns:
            New ``sv.Detections`` with ``tracker_id`` assigned for each
            detection.
        """
        self._warn_if_frame_unused(frame)
        return super().update(detections=detections, frame=None)
