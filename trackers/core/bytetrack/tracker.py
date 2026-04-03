# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.bytetrack.kalman import ByteTrackKalmanBoxTracker
from trackers.core.sort.utils import (
    get_alive_trackers,
    get_iou_matrix,
)


def _giou_matrix(
    track_bboxes: np.ndarray,
    det_bboxes: np.ndarray,
) -> np.ndarray:
    """Compute Generalized IoU between all track-detection pairs.

    GIoU extends standard IoU by adding a penalty term for the gap between
    the two boxes and their smallest enclosing box, providing a non-zero
    gradient even for non-overlapping boxes.

    Args:
        track_bboxes: Track predicted bboxes, shape ``(N, 4)`` in
            ``[x1, y1, x2, y2]`` format.
        det_bboxes: Detection bboxes, shape ``(M, 4)`` in
            ``[x1, y1, x2, y2]`` format.

    Returns:
        GIoU matrix of shape ``(N, M)`` with values in ``[-1, 1]``.
    """
    # Intersection corners
    x1_i = np.maximum(track_bboxes[:, None, 0], det_bboxes[None, :, 0])
    y1_i = np.maximum(track_bboxes[:, None, 1], det_bboxes[None, :, 1])
    x2_i = np.minimum(track_bboxes[:, None, 2], det_bboxes[None, :, 2])
    y2_i = np.minimum(track_bboxes[:, None, 3], det_bboxes[None, :, 3])
    inter = np.maximum(0.0, x2_i - x1_i) * np.maximum(0.0, y2_i - y1_i)

    # Individual box areas and union
    area_t = (track_bboxes[:, 2] - track_bboxes[:, 0]) * (
        track_bboxes[:, 3] - track_bboxes[:, 1]
    )
    area_d = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (
        det_bboxes[:, 3] - det_bboxes[:, 1]
    )
    union = area_t[:, None] + area_d[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)

    # Smallest enclosing box area
    x1_c = np.minimum(track_bboxes[:, None, 0], det_bboxes[None, :, 0])
    y1_c = np.minimum(track_bboxes[:, None, 1], det_bboxes[None, :, 1])
    x2_c = np.maximum(track_bboxes[:, None, 2], det_bboxes[None, :, 2])
    y2_c = np.maximum(track_bboxes[:, None, 3], det_bboxes[None, :, 3])
    enc_area = (x2_c - x1_c) * (y2_c - y1_c)

    giou = iou - np.where(enc_area > 0, (enc_area - union) / enc_area, 0.0)
    return giou.astype(np.float32)


class ByteTrackTracker(BaseTracker):
    """ByteTrack operates online by processing all detector outputs, categorizing them
    by confidence thresholds to enable a two-stage association process. High-score boxes
    are initially linked to tracklets via Kalman filter predictions and IoU-based
    Hungarian matching, optionally enhanced with appearance features. Low-score boxes
    follow in a secondary matching phase using pure motion similarity to revive occluded
    tracks. Tracks without matches are kept briefly for potential re-association,
    preventing premature termination. This inclusive approach addresses common pitfalls
    in detection filtering, establishing ByteTrack as a flexible enhancer for existing
    tracking frameworks.

    ByteTrack excels in dense environments, where its low-score recovery mechanism
    minimizes missed detections and enhances overall trajectory completeness. It
    consistently improves performance across diverse datasets, demonstrating robustness
    and generalization. The tracker's speed remains competitive, facilitating
    integration into production pipelines. On the downside, it is highly dependent on
    detector quality, with performance drops in noisy or low-resolution inputs.
    Additionally, the motion-only secondary association may lead to erroneous matches
    in scenes with similar moving objects.

    Args:
        lost_track_buffer: `int` specifying number of frames to buffer when a
            track is lost. Increasing this value enhances occlusion handling but
            may increase ID switching for disappearing objects.
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
        stage2_iou_threshold: `float` specifying IoU threshold for stage-2
            low-confidence association. Lower values are more permissive when
            reviving tracks from low-confidence detections.
        iou_age_weight: `float` specifying how much to discount IoU
            similarity for lost tracks in stage-1 assignment ranking.
            Each lost track's IoU row is scaled by
            ``1 / (1 + iou_age_weight * lost_frames)`` where
            ``lost_frames = max(0, time_since_update - 1)``. The discount
            biases the solver to prefer active tracks over stale
            predictions (reducing identity switches) but only affects
            ranking — the minimum-IoU threshold is checked against the
            *raw* IoU so that valid matches are never rejected by the
            discount alone. ``0`` disables the discount. Only applied in
            stage 1; stage 2 is unaffected so that lost tracks can still
            recover via low-confidence detections.
        high_conf_det_threshold: `float` specifying threshold for separating
            high and low confidence detections in the two-stage association.
    """

    tracker_id = "bytetrack"

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.1,
        stage2_iou_threshold: float = 0.05,
        iou_age_weight: float = 0.03,
        high_conf_det_threshold: float = 0.6,
        conf_cost_weight: float = 0.0,
        stage2_min_updates: int = 0,
        giou_blend: float = 0.0,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.stage2_iou_threshold = stage2_iou_threshold
        self.iou_age_weight = iou_age_weight
        self.track_activation_threshold = track_activation_threshold
        self.high_conf_det_threshold = high_conf_det_threshold
        self.conf_cost_weight = conf_cost_weight
        self.stage2_min_updates = stage2_min_updates
        self.giou_blend = giou_blend
        self.tracks: list[ByteTrackKalmanBoxTracker] = []

    def update(
        self,
        detections: sv.Detections,
    ) -> sv.Detections:
        """Update tracker state with new detections and return tracked objects.
        Performs Kalman filter prediction, two-stage association (high then low
        confidence), and initializes new tracks for unmatched detections.

        Args:
            detections: `sv.Detections` containing bounding boxes with shape
                `(N, 4)` in `(x_min, y_min, x_max, y_max)` format and optional
                confidence scores.

        Returns:
            `sv.Detections` with `tracker_id` assigned for each detection.
                Unmatched detections have `tracker_id` of `-1`. Detection order
                may differ from input.
        """
        if len(self.tracks) == 0 and len(detections) == 0:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        out_det_indices: list[int] = []
        out_tracker_ids: list[int] = []

        for tracker in self.tracks:
            tracker.predict()

        detection_boxes = detections.xyxy
        confidences = (
            detections.confidence
            if detections.confidence is not None
            else np.zeros(len(detections))
        )

        # Split indices by confidence threshold (no sv.Detections slicing)
        high_mask = confidences >= self.high_conf_det_threshold
        high_indices = np.where(high_mask)[0]
        low_indices = np.where(~high_mask)[0]
        high_boxes = detection_boxes[high_indices]
        low_boxes = detection_boxes[low_indices]

        # Step 1: associate high-confidence detections to all tracks
        raw_iou = get_iou_matrix(self.tracks, high_boxes)

        # GIoU blend: mix standard IoU with normalized GIoU (∈ [-1,1] mapped
        # to [0,1]) to provide proximity credit for near-miss cases where the
        # Kalman prediction drifted slightly below the IoU threshold.  The
        # threshold gate always uses *raw* IoU so the blend only affects
        # ranking, never filtering.
        if self.giou_blend > 0 and raw_iou.size > 0:
            track_bboxes = np.array(
                [t.get_state_bbox() for t in self.tracks], dtype=np.float32
            )
            giou = _giou_matrix(track_bboxes, high_boxes)
            iou_matrix = (
                (1.0 - self.giou_blend) * raw_iou + self.giou_blend * (giou + 1.0) / 2.0
            ).astype(np.float32)
        else:
            iou_matrix = raw_iou

        # Age discount: scale down IoU for lost tracks so the assignment
        # algorithm prefers active tracks over stale predictions.  This
        # reduces identity switches from a drifted prediction "stealing"
        # a detection that should go to the correct active track.
        #
        # The discount is applied only to the cost matrix used by the
        # solver for ranking; the threshold check uses the *raw* IoU so
        # that valid matches are never rejected by the discount alone.
        # Only tracks that have been lost for at least 1 frame are
        # discounted — freshly-seen tracks (time_since_update == 1 after
        # predict, i.e. lost_frames == 0) are never penalised.
        if self.iou_age_weight > 0 and iou_matrix.size > 0:
            lost_frames = np.array(
                [max(0, t.time_since_update - 1) for t in self.tracks],
                dtype=np.float32,
            )
            discount = 1.0 / (1.0 + self.iou_age_weight * lost_frames)
            solver_iou = iou_matrix * discount[:, np.newaxis]
        else:
            solver_iou = iou_matrix

        # Confidence boost: scale up solver IoU for higher-confidence detections
        # so the Hungarian assignment prefers confident detections over uncertain
        # ones when IoU values are close.  The boost only affects ranking; the
        # threshold gate still uses raw IoU so valid matches are never blocked.
        if self.conf_cost_weight > 0 and solver_iou.size > 0 and len(high_indices) > 0:
            conf_boost = 1.0 + self.conf_cost_weight * confidences[high_indices]
            solver_iou = solver_iou * conf_boost[np.newaxis, :]

        matched, unmatched_tracks, unmatched_high = self._get_associated_indices(
            solver_iou, self.minimum_iou_threshold, raw_similarity=raw_iou
        )

        for row, col in matched:
            track = self.tracks[row]
            track.update(high_boxes[col])
            if (
                track.number_of_successful_updates >= self.minimum_consecutive_frames
                and track.tracker_id == -1
            ):
                track.tracker_id = ByteTrackKalmanBoxTracker.get_next_tracker_id()
            out_det_indices.append(int(high_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        # Stage-2 maturity gate: only established tracks participate in low-conf
        # recovery.  Young tracks (number_of_successful_updates < stage2_min_updates)
        # are excluded to prevent tentative ghost tracks from being incorrectly
        # recovered via low-confidence detections.  When stage2_min_updates == 0
        # (default) all unmatched tracks participate, preserving the original behaviour.
        if self.stage2_min_updates > 0:
            remaining_tracks = [
                self.tracks[i]
                for i in unmatched_tracks
                if self.tracks[i].number_of_successful_updates
                >= self.stage2_min_updates
            ]
        else:
            remaining_tracks = [self.tracks[i] for i in unmatched_tracks]

        # Step 2: associate low-confidence detections to remaining tracks
        iou_matrix = get_iou_matrix(remaining_tracks, low_boxes)
        matched, _, unmatched_low = self._get_associated_indices(
            iou_matrix, self.stage2_iou_threshold
        )

        for row, col in matched:
            track = remaining_tracks[row]
            track.update(low_boxes[col])
            if (
                track.number_of_successful_updates >= self.minimum_consecutive_frames
                and track.tracker_id == -1
            ):
                track.tracker_id = ByteTrackKalmanBoxTracker.get_next_tracker_id()
            out_det_indices.append(int(low_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        # Unmatched low-confidence detections
        for det_local_idx in unmatched_low:
            out_det_indices.append(int(low_indices[det_local_idx]))
            out_tracker_ids.append(-1)

        # Spawn new tracks from unmatched high-confidence detections
        self._spawn_new_trackers(
            detection_boxes,
            confidences,
            unmatched_high,
            high_indices,
            out_det_indices,
            out_tracker_ids,
        )

        self.tracks = get_alive_trackers(
            trackers=self.tracks,
            maximum_frames_without_update=self.maximum_frames_without_update,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )

        # Build final sv.Detections from original by indexing
        if not out_det_indices:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        idx = np.array(out_det_indices)
        result = detections[idx]
        result.tracker_id = np.array(out_tracker_ids, dtype=int)
        return result

    def _get_associated_indices(
        self,
        similarity_matrix: np.ndarray,
        min_similarity_thresh: float,
        raw_similarity: np.ndarray | None = None,
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """Associate detections to tracks based on similarity (IoU).

        Uses the Jonker-Volgenant algorithm (via ``linear_sum_assignment``)
        to solve the assignment optimally.

        Args:
            similarity_matrix: Similarity matrix between tracks (rows) and
                detections (columns).  Used by the solver for ranking.
            min_similarity_thresh: Minimum similarity threshold for a valid
                match.
            raw_similarity: Optional unmodified similarity matrix.  When
                provided, the threshold check uses this matrix instead of
                ``similarity_matrix`` so that solver-side discounts (e.g.
                the age discount) cannot reject otherwise valid matches.

        Returns:
            Matched indices (list of (tracker_idx, detection_idx)), indices
                of unmatched tracks, indices of unmatched detections.
        """
        matched_indices = []
        n_tracks, n_detections = similarity_matrix.shape
        unmatched_tracks = set(range(n_tracks))
        unmatched_detections = set(range(n_detections))

        # Use raw similarity for threshold gating when available
        thresh_matrix = (
            raw_similarity if raw_similarity is not None else similarity_matrix
        )

        if n_tracks > 0 and n_detections > 0:
            row_indices, col_indices = linear_sum_assignment(
                similarity_matrix, maximize=True
            )
            for row, col in zip(row_indices, col_indices):
                if thresh_matrix[row, col] >= min_similarity_thresh:
                    matched_indices.append((row, col))
                    unmatched_tracks.remove(row)
                    unmatched_detections.remove(col)

        return matched_indices, unmatched_tracks, unmatched_detections

    def _spawn_new_trackers(
        self,
        detection_boxes: np.ndarray,
        confidences: np.ndarray,
        unmatched_high_local: set[int],
        high_indices: np.ndarray,
        out_det_indices: list[int],
        out_tracker_ids: list[int],
    ) -> None:
        for det_local_idx in unmatched_high_local:
            global_idx = int(high_indices[det_local_idx])
            conf = float(confidences[global_idx])
            if conf >= self.track_activation_threshold:
                self.tracks.append(
                    ByteTrackKalmanBoxTracker(bbox=detection_boxes[global_idx])
                )
                out_det_indices.append(global_idx)
                out_tracker_ids.append(-1)

    def reset(self) -> None:
        """Reset tracker state by clearing all tracks and resetting ID counter.
        Call this method when switching to a new video or scene.
        """
        self.tracks = []
        ByteTrackKalmanBoxTracker.count_id = 0
