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
        high_conf_det_threshold: `float` specifying confidence threshold for
            two-stage association. Detections above this threshold are matched
            in stage 1; remaining low-confidence detections are matched in
            stage 2 against unmatched tracks using ``stage2_iou_threshold``.
            ``0.0`` disables two-stage (all detections go to stage 1).
        stage2_iou_threshold: `float` specifying DIoU threshold for the
            second association stage with low-confidence detections. Typically
            lower than ``minimum_iou_threshold`` to be more permissive when
            recovering tracks from occluded or partially-visible objects.
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
        high_conf_det_threshold: float = 0.0,
        stage2_iou_threshold: float = 0.05,
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
        self.high_conf_det_threshold = high_conf_det_threshold
        self.stage2_iou_threshold = stage2_iou_threshold

        # Active trackers
        self.trackers: list[SORTKalmanBoxTracker] = []

    @staticmethod
    def _match(
        similarity: np.ndarray,
        min_thresh: float,
        raw_similarity: np.ndarray | None = None,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """Solve the assignment problem on a similarity matrix.

        Args:
            similarity: Similarity matrix (rows=tracks, cols=detections)
                used by the solver for ranking.
            min_thresh: Minimum similarity for a valid match.
            raw_similarity: Optional unmodified similarity matrix.  When
                provided, the threshold check uses this matrix instead of
                ``similarity`` so that solver-side boosts cannot reject
                otherwise valid matches.

        Returns:
            Matched (row, col) pairs, unmatched row indices, unmatched
            col indices.
        """
        n_rows, n_cols = similarity.shape
        matched: list[tuple[int, int]] = []
        unmatched_rows = set(range(n_rows))
        unmatched_cols = set(range(n_cols))

        thresh_matrix = raw_similarity if raw_similarity is not None else similarity

        if n_rows > 0 and n_cols > 0:
            row_idx, col_idx = linear_sum_assignment(similarity, maximize=True)
            for r, c in zip(row_idx, col_idx):
                if thresh_matrix[r, c] >= min_thresh:
                    matched.append((r, c))
                    unmatched_rows.remove(r)
                    unmatched_cols.remove(c)

        return matched, unmatched_rows, unmatched_cols

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

    def _build_solver_iou(
        self,
        raw_iou: np.ndarray,
        trackers: list[SORTKalmanBoxTracker],
        confidences: np.ndarray | None,
    ) -> np.ndarray:
        """Apply age discount and confidence boost to the raw similarity matrix.

        Args:
            raw_iou: Raw DIoU similarity matrix (tracks x detections).
            trackers: Track list whose rows correspond to ``raw_iou``.
            confidences: Per-detection confidence scores, or None.

        Returns:
            Solver similarity matrix with discounts/boosts applied.
        """
        solver_iou = raw_iou

        # Age discount: scale down similarity for lost tracks so the solver
        # prefers active tracks over stale predictions.
        if self.iou_age_weight > 0 and solver_iou.size > 0:
            lost_frames = np.array(
                [max(0, t.time_since_update - 1) for t in trackers],
                dtype=np.float32,
            )
            discount = 1.0 / (1.0 + self.iou_age_weight * lost_frames)
            solver_iou = (solver_iou * discount[:, np.newaxis]).astype(np.float32)

        # Confidence boost: scale up solver similarity for higher-confidence
        # detections so the assignment prefers confident detections over uncertain
        # ones when DIoU values are close.
        if (
            self.conf_cost_weight > 0
            and solver_iou.size > 0
            and confidences is not None
        ):
            conf_boost = 1.0 + self.conf_cost_weight * confidences
            solver_iou = (solver_iou * conf_boost[np.newaxis, :]).astype(np.float32)

        return solver_iou

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker state with new detections and return tracked objects.

        Performs Kalman filter prediction, optionally two-stage DIoU-based
        association (high-confidence then low-confidence), and initializes
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
        confidences = detections.confidence

        for tracker in self.trackers:
            tracker.predict()

        # Two-stage split: separate high and low confidence detections.
        # When high_conf_det_threshold == 0, all detections are "high" and
        # stage 2 is empty — equivalent to single-stage SORT.
        use_two_stage = (
            self.high_conf_det_threshold > 0
            and confidences is not None
            and len(detection_boxes) > 0
        )

        if use_two_stage:
            high_mask = confidences >= self.high_conf_det_threshold
            high_indices = np.where(high_mask)[0]
            low_indices = np.where(~high_mask)[0]
            high_boxes = detection_boxes[high_indices]
            low_boxes = detection_boxes[low_indices]
            high_confs = confidences[high_indices]
        else:
            high_indices = np.arange(len(detection_boxes))
            high_boxes = detection_boxes
            high_confs = confidences
            low_indices = np.array([], dtype=int)
            low_boxes = np.array([]).reshape(0, 4)

        # --- Stage 1: match high-confidence detections to all tracks ---
        raw_iou = get_iou_matrix(self.trackers, high_boxes)
        solver_iou = self._build_solver_iou(raw_iou, self.trackers, high_confs)

        matched_s1, unmatched_tracks_s1, unmatched_high = self._match(
            solver_iou, self.minimum_iou_threshold, raw_similarity=raw_iou
        )

        # Record matched track<->detection mapping (global det indices)
        matched_tracker_for_det: dict[int, SORTKalmanBoxTracker] = {}
        for row, col in matched_s1:
            global_idx = int(high_indices[col])
            self.trackers[row].update(high_boxes[col])
            matched_tracker_for_det[global_idx] = self.trackers[row]

        # --- Stage 2: match low-confidence detections to unmatched tracks ---
        if use_two_stage and len(low_boxes) > 0 and len(unmatched_tracks_s1) > 0:
            remaining = [self.trackers[i] for i in unmatched_tracks_s1]
            raw_iou_s2 = get_iou_matrix(remaining, low_boxes)
            # No conf boost or age discount for stage 2 — keep it simple
            matched_s2, _, _ = self._match(raw_iou_s2, self.stage2_iou_threshold)
            for row, col in matched_s2:
                global_idx = int(low_indices[col])
                remaining[row].update(low_boxes[col])
                matched_tracker_for_det[global_idx] = remaining[row]

        # Spawn new tracks from unmatched high-confidence detections only
        unmatched_global = {int(high_indices[c]) for c in unmatched_high}
        self._spawn_new_trackers(confidences, detection_boxes, unmatched_global)

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
