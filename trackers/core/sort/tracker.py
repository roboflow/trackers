# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.sort.kalman import SORTKalmanBoxTracker
from trackers.core.sort.utils import (
    get_alive_trackers,
    get_iou_matrix,
)


class SORTTracker(BaseTracker):
    """In SORT, object tracking begins with high-confidence detections fed into a
    Kalman filter framework assuming uniform motion for state prediction across frames.
    Association occurs via IoU-based costs in the Hungarian algorithm, enforcing a
    threshold to filter weak matches and initialize new identities. Tracks persist only
    with consistent associations, terminating quickly to avoid erroneous propagation.
    This detection-driven approach underscores the importance of upstream detector
    performance in achieving competitive multi-object tracking results. Over time, SORT
    has become a cornerstone for evaluating motion-based improvements in the field.

    SORT's standout strength is its real-time capability, processing hundreds of frames
    per second while maintaining accuracy comparable to more complex offline methods. It
    performs well in controlled environments with reliable detections, minimizing
    computational demands. However, without mechanisms for re-identification, it incurs
    frequent identity switches during object reappearances post-occlusion. The linear
    motion assumption limits effectiveness in non-linear paths, such as those in sports
    or wildlife tracking. Ultimately, SORT's efficiency is offset by its sensitivity to
    environmental complexities, necessitating hybrid extensions for broader
    applicability.

    Args:
        lost_track_buffer: `int` specifying number of frames to buffer when a
            track is lost. Increasing this value enhances occlusion handling but
            may increase ID switching for similar objects.
        frame_rate: `float` specifying video frame rate in frames per second.
            Used to scale the lost track buffer for consistent tracking across
            different frame rates.
        track_activation_threshold: `float` specifying minimum detection
            confidence to create new tracks. Higher values reduce false
            positives but may miss low-confidence objects.
        minimum_consecutive_frames: `int` specifying number of consecutive
            frames before a track is considered valid. Before reaching this
            threshold, tracks are assigned `tracker_id` of `-1`.
        minimum_iou_threshold: `float` specifying IoU threshold for associating
            detections to existing tracks. Higher values require more overlap.
        velocity_decay: `float` in [0, 1] applied to the Kalman velocity
            components each missed frame, preventing runaway linear extrapolation
            during occlusions. `1.0` disables the feature.
        q_miss_alpha: `float` ≥ 0 scaling the per-frame Q inflation rate for
            lost tracks (`Q_eff = Q * (1 + alpha * time_since_update)`). Wider
            uncertainty on re-detection gives higher Kalman gain to fresh
            measurements. `0.0` disables the feature.
        p_reset_threshold: `int` minimum number of missed frames before resetting
            the error covariance P to identity on re-detection. Discards stale
            accumulated uncertainty after long gaps. `0` disables the reset.
        oru_threshold: `int` minimum missed frames before applying observation-
            centric velocity re-estimation on re-detection. Computes a virtual
            trajectory velocity from (current - last_observed) / gap to replace
            the decayed Kalman velocity. Technique from OC-SORT. `0` disables.
        conf_cost_weight: `float` specifying how strongly detection confidence
            breaks IoU ties in the Hungarian assignment. A value of ``0.0``
            disables the feature (pure IoU). Positive values boost
            higher-confidence detections in the solver matrix while keeping
            the IoU gate unchanged, so no invalid matches are accepted.
        iou_age_weight: `float` specifying how much to discount DIoU
            similarity for lost tracks. Each lost track's row is scaled by
            ``1 / (1 + iou_age_weight * lost_frames)`` where
            ``lost_frames = max(0, time_since_update - 1)``. This biases the
            solver to prefer active tracks over stale predictions, reducing
            identity switches from drifted predictions. The threshold gate
            always uses raw DIoU. ``0`` disables the discount.
    """

    tracker_id = "sort"

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
        velocity_decay: float = 0.95,
        q_miss_alpha: float = 0.0,
        p_reset_threshold: int = 0,
        oru_threshold: int = 0,
        conf_cost_weight: float = 0.0,
        iou_age_weight: float = 0.0,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold
        self.velocity_decay = velocity_decay
        self.q_miss_alpha = q_miss_alpha
        self.p_reset_threshold = p_reset_threshold
        self.oru_threshold = oru_threshold
        self.conf_cost_weight = conf_cost_weight
        self.iou_age_weight = iou_age_weight

        # Active trackers
        self.trackers: list[SORTKalmanBoxTracker] = []

    def _get_associated_indices(
        self,
        iou_matrix: np.ndarray,
        detection_boxes: np.ndarray,
        raw_similarity: np.ndarray | None = None,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """Associate detections to trackers based on IOU.

        Args:
            iou_matrix: Similarity matrix used by the solver for ranking.
                May include confidence-boosted values.
            detection_boxes: Detected bounding boxes in the form [x1, y1, x2, y2].
            raw_similarity: Optional unmodified similarity matrix.  When
                provided, the threshold check uses this matrix instead of
                ``iou_matrix`` so that solver-side boosts cannot reject
                otherwise valid matches.

        Returns:
            Matched indices, unmatched trackers, unmatched detections.
        """
        matched_indices = []
        unmatched_trackers = set(range(len(self.trackers)))
        unmatched_detections = set(range(len(detection_boxes)))

        # Use raw similarity for threshold gating when available
        thresh_matrix = raw_similarity if raw_similarity is not None else iou_matrix

        if len(self.trackers) > 0 and len(detection_boxes) > 0:
            # iou_matrix may already be boosted by the caller; use it directly
            # for ranking. The threshold gate uses thresh_matrix (raw) so the
            # IoU semantics are preserved regardless of any solver-side boost.
            solver_iou = iou_matrix

            # Find optimal assignment using scipy.optimize.linear_sum_assignment.
            # Note that it uses a a modified Jonker-Volgenant algorithm with no
            # initialization instead of the Hungarian algorithm as mentioned in the
            # SORT paper.
            row_indices, col_indices = linear_sum_assignment(solver_iou, maximize=True)
            for row, col in zip(row_indices, col_indices):
                if thresh_matrix[row, col] >= self.minimum_iou_threshold:
                    matched_indices.append((row, col))
                    unmatched_trackers.remove(row)
                    unmatched_detections.remove(col)

        return matched_indices, unmatched_trackers, unmatched_detections

    def _spawn_new_trackers(
        self,
        confidences: np.ndarray | None,
        detection_boxes: np.ndarray,
        unmatched_detections: set[int],
    ) -> None:
        for detection_idx in unmatched_detections:
            if (
                confidences is None
                or detection_idx >= len(confidences)
                or confidences[detection_idx] >= self.track_activation_threshold
            ):
                self.trackers.append(
                    SORTKalmanBoxTracker(
                        detection_boxes[detection_idx],
                        velocity_decay=self.velocity_decay,
                        q_miss_alpha=self.q_miss_alpha,
                        p_reset_threshold=self.p_reset_threshold,
                        oru_threshold=self.oru_threshold,
                    )
                )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker state with new detections and return tracked objects.
        Performs Kalman filter prediction, IoU-based association, and initializes
        new tracks for unmatched high-confidence detections.

        Args:
            detections: `sv.Detections` containing bounding boxes with shape
                `(N, 4)` in `(x_min, y_min, x_max, y_max)` format and optional
                confidence scores.

        Returns:
            `sv.Detections` with `tracker_id` assigned for each detection.
                Unmatched or immature tracks have `tracker_id` of `-1`.
        """
        if len(self.trackers) == 0 and len(detections) == 0:
            detections.tracker_id = np.array([], dtype=int)
            return detections

        detection_boxes = (
            detections.xyxy if len(detections) > 0 else np.array([]).reshape(0, 4)
        )

        for tracker in self.trackers:
            tracker.predict()

        raw_iou = get_iou_matrix(self.trackers, detection_boxes)

        solver_iou = raw_iou

        # Age discount: scale down similarity for lost tracks so the solver
        # prefers active tracks over stale predictions.  Reduces identity
        # switches from drifted Kalman predictions "stealing" detections.
        # The threshold gate uses raw DIoU so valid matches are never rejected.
        if self.iou_age_weight > 0 and solver_iou.size > 0:
            lost_frames = np.array(
                [max(0, t.time_since_update - 1) for t in self.trackers],
                dtype=np.float32,
            )
            discount = 1.0 / (1.0 + self.iou_age_weight * lost_frames)
            solver_iou = (solver_iou * discount[:, np.newaxis]).astype(np.float32)

        # Confidence boost: scale up solver similarity for higher-confidence
        # detections so the assignment prefers confident detections over uncertain
        # ones when DIoU values are close.  The threshold gate uses raw DIoU so
        # valid matches are never blocked by the boost.
        if (
            self.conf_cost_weight > 0
            and solver_iou.size > 0
            and detections.confidence is not None
        ):
            conf_boost = 1.0 + self.conf_cost_weight * detections.confidence
            solver_iou = (solver_iou * conf_boost[np.newaxis, :]).astype(np.float32)

        matched_indices, _, unmatched_detections = self._get_associated_indices(
            solver_iou, detection_boxes, raw_similarity=raw_iou
        )

        # Update matched trackers and record the det_idx -> tracker mapping
        matched_tracker_for_det: dict[int, SORTKalmanBoxTracker] = {}
        for row, col in matched_indices:
            self.trackers[row].update(detection_boxes[col])
            matched_tracker_for_det[col] = self.trackers[row]

        self._spawn_new_trackers(
            detections.confidence, detection_boxes, unmatched_detections
        )

        self.trackers = get_alive_trackers(
            self.trackers,
            self.minimum_consecutive_frames,
            self.maximum_frames_without_update,
        )

        # Build tracker_ids from the recorded mapping (no deepcopy, no re-IoU)
        tracker_ids = np.full(len(detection_boxes), -1, dtype=int)
        for det_idx, tracker in matched_tracker_for_det.items():
            if tracker.number_of_successful_updates >= self.minimum_consecutive_frames:
                if tracker.tracker_id == -1:
                    tracker.tracker_id = SORTKalmanBoxTracker.get_next_tracker_id()
                tracker_ids[det_idx] = tracker.tracker_id

        detections.tracker_id = tracker_ids
        return detections

    def reset(self) -> None:
        """Reset tracker state by clearing all tracks and resetting ID counter.
        Call this method when switching to a new video or scene.
        """
        self.trackers = []
        SORTKalmanBoxTracker.count_id = 0
