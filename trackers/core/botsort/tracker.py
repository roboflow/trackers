# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.botsort.cmc import CMC, CMCConfig
from trackers.core.botsort.tracklet import BoTSORTTracklet
from trackers.core.botsort.utils import get_alive_trackers
from trackers.core.sort.utils import _get_iou_matrix
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCWHStateEstimator,
)


class BoTSORTTracker(BaseTracker):
    """
    BoT-SORT-style multi-object tracker (IoU association + optional CMC).

    The tracker maintains a list of active tracks (Kalman-filter-based) and, for each
    frame, performs:
      1) Predict existing track states (Kalman predict)
      2) Split detections into high/low confidence groups
      3) Apply camera motion compensation to predicted tracks
      4) Associate high-confidence detections to tracks (IoU + assignment)
      5) Associate low-confidence detections to remaining tracks
      6) Spawn new tracks from unmatched high-confidence detections
      7) Remove tracks that have been lost for too long

    Parameters in __init__ control thresholds and lifecycle logic similarly to
    ByteTrack.

    Attributes:
        tracks: List of active ``BoTSORTTracklet`` objects.
        maximum_frames_without_update: Max number of consecutive frames a track can go
            unmatched before being removed.
        minimum_consecutive_frames: Track maturity threshold before assigning a
            permanent ID.
        minimum_iou_threshold_first_assoc: Minimum IoU required for a valid match
            in the first association step
        minimum_iou_threshold_second_assoc: Minimum IoU required for a valid match
            in the second association step
        track_activation_threshold: Confidence threshold for spawning a new track.
        high_conf_det_threshold: Confidence threshold splitting detections into
            high/low groups.
        enable_cmc: Whether to run camera motion compensation each frame
            (if `cmc` is set).
        cmc: Camera motion compensation instance (or None if disabled).
    """

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold_first_assoc: float = 0.2,
        minimum_iou_threshold_second_assoc: float = 0.5,
        high_conf_det_threshold: float = 0.6,
        enable_cmc: bool = True,
        cmc_method: str = "sparseOptFlow",
        cmc_downscale: int = 2,
        state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator,
    ) -> None:
        """
        Initialize the tracker.

        Args:
            lost_track_buffer: Time buffer (in frames at 30 FPS) for keeping lost tracks
                alive before deletion. This is scaled by `frame_rate`.
            frame_rate: Video frame rate used to scale the lost track buffer to
                time-like behavior.
            track_activation_threshold: Minimum detection confidence to spawn a new
                track.
            minimum_consecutive_frames: Number of successful updates required before
                assigning a stable track ID (different than initial -1).
            minimum_iou_threshold_first_assoc: Minimum IoU to accept a detection-track
                association during the first association step.
            minimum_iou_threshold_second_assoc: Minimum IoU to accept a detection-track
                association during the second association step.
            high_conf_det_threshold: Confidence threshold used to split detections into:
                - high confidence: confidence >= threshold
                - low confidence:  confidence < threshold
            enable_cmc: Whether to enable camera motion compensation (CMC).
            cmc_method: CMC method string passed into `CMCConfig(method=...)`.
                Supported values depend on `CMC` (e.g. "orb", "sift", "sparseOptFlow",
                "ecc"). See CMCConfig.
            cmc_downscale: Downscale factor used inside CMC for speed/robustness.
            state_estimator_class: State estimator class for tracklets. Defaults
                to ``XCYCWHStateEstimator``.

        Notes:
            - `maximum_frames_without_update` is computed as:
                int(frame_rate / 30.0 * lost_track_buffer)
              to maintain consistent “seconds” worth of buffer across different FPS.
        """
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold_first_assoc = minimum_iou_threshold_first_assoc
        self.minimum_iou_threshold_second_assoc = minimum_iou_threshold_second_assoc
        self.track_activation_threshold = track_activation_threshold
        self.high_conf_det_threshold = high_conf_det_threshold
        self.tracks: list[BoTSORTTracklet] = []
        self.state_estimator_class = state_estimator_class

        self.enable_cmc = enable_cmc
        self.cmc = (
            CMC(CMCConfig(method=cmc_method, downscale=cmc_downscale))
            if enable_cmc
            else None
        )

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
    ) -> sv.Detections:
        """
        Update the tracker with detections from the current frame.

        This is the main per-frame entry point.

        Args:
            detections: Supervision detections for the current frame. Must include `
                .xyxy`. Confidence (`detections.confidence`) is optional but
                recommended. The method writes/overwrites `detections.tracker_id`.
            frame: Current video frame in BGR format (H, W, 3), required if CMC is
                enabled.

        Returns:
            ``sv.Detections`` with ``tracker_id`` assigned (>= 0 confirmed,
            -1 unconfirmed).

        Notes:
            - If CMC is enabled, the tracker estimates a global affine transform (2x3)
              from the frame and uses it to warp predicted track states before
              association.
        """
        if len(self.tracks) == 0 and len(detections) == 0:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        out_det_indices: list[int] = []
        out_tracker_ids: list[int] = []

        # Predict new locations for existing tracks
        for tracker in self.tracks:
            tracker.predict()

        detection_boxes = detections.xyxy
        confidences = (
            detections.confidence
            if detections.confidence is not None
            else np.zeros(len(detections))
        )

        # Split indices into high / low / discarded by confidence
        high_mask = confidences >= self.high_conf_det_threshold
        low_mask = (confidences > 0.1) & (~high_mask)

        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]

        high_boxes = detection_boxes[high_indices]
        low_boxes = detection_boxes[low_indices]

        # CMC: apply to all predicted tracks before association
        if self.enable_cmc and self.cmc is not None and frame is not None:
            mask_boxes = high_boxes if len(high_boxes) > 0 else None
            H = self.cmc.estimate(frame, mask_boxes)
            if H is not None:
                for trk in self.tracks:
                    trk.apply_cmc(H)

        # Step 1: associate high-confidence detections to all tracks
        iou_matrix = _get_iou_matrix(self.tracks, high_boxes)
        matched, unmatched_tracks, unmatched_high = self._get_associated_indices(
            iou_matrix, self.minimum_iou_threshold_first_assoc
        )

        for row, col in matched:
            track = self.tracks[row]
            track.update(high_boxes[col])
            if (
                track.number_of_successful_updates
                >= self.minimum_consecutive_frames
                and track.tracker_id == -1
            ):
                track.tracker_id = BoTSORTTracklet.get_next_tracker_id()
            out_det_indices.append(int(high_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]

        # Step 2: associate low-confidence detections to remaining tracks
        iou_matrix = _get_iou_matrix(remaining_tracks, low_boxes)
        matched, _, unmatched_low = self._get_associated_indices(
            iou_matrix, self.minimum_iou_threshold_second_assoc
        )

        for row, col in matched:
            track = remaining_tracks[row]
            track.update(low_boxes[col])
            if (
                track.number_of_successful_updates
                >= self.minimum_consecutive_frames
                and track.tracker_id == -1
            ):
                track.tracker_id = BoTSORTTracklet.get_next_tracker_id()
            out_det_indices.append(int(low_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        # Unmatched low-confidence detections
        for det_local_idx in unmatched_low:
            out_det_indices.append(int(low_indices[det_local_idx]))
            out_tracker_ids.append(-1)

        # Spawn new tracks from unmatched high-confidence detections
        self._spawn_new_tracks(
            detection_boxes,
            confidences,
            unmatched_high,
            high_indices,
            out_det_indices,
            out_tracker_ids,
        )

        # Kill lost tracks
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
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """
        Associate detections to tracks based on Similarity (IoU) using the
        Jonker-Volgenant algorithm approach with no initialization instead of the
        Hungarian algorithm as mentioned in the SORT paper, but it solves the
        assignment problem in an optimal way.

        Args:
            similarity_matrix: Similarity matrix between tracks (rows) and detections
            (columns). min_similarity_thresh: Minimum similarity threshold for a valid
            match.

        Returns:
            Matched indices (list of (tracker_idx, detection_idx)), indices of
                unmatched tracks, indices of unmatched detections.
        """
        matched_indices = []
        n_tracks, n_detections = similarity_matrix.shape
        unmatched_tracks = set(range(n_tracks))
        unmatched_detections = set(range(n_detections))

        if n_tracks > 0 and n_detections > 0:
            row_indices, col_indices = linear_sum_assignment(
                similarity_matrix, maximize=True
            )
            for row, col in zip(row_indices, col_indices):
                if similarity_matrix[row, col] >= min_similarity_thresh:
                    matched_indices.append((row, col))
                    unmatched_tracks.remove(row)
                    unmatched_detections.remove(col)

        return matched_indices, unmatched_tracks, unmatched_detections

    def _spawn_new_tracks(
        self,
        detection_boxes: np.ndarray,
        confidences: np.ndarray,
        unmatched_high_local: set[int],
        high_indices: np.ndarray,
        out_det_indices: list[int],
        out_tracker_ids: list[int],
    ) -> None:
        """Create new tracklets from unmatched high-confidence detections."""
        for det_local_idx in unmatched_high_local:
            global_idx = int(high_indices[det_local_idx])
            conf = float(confidences[global_idx])
            if conf >= self.track_activation_threshold:
                self.tracks.append(
                    BoTSORTTracklet(
                        initial_bbox=detection_boxes[global_idx],
                        state_estimator_class=self.state_estimator_class,
                    )
                )
                out_det_indices.append(global_idx)
                out_tracker_ids.append(-1)

    def reset(self) -> None:
        """Reset tracker state by clearing all tracks and resetting ID counter.
        Call this method when switching to a new video or scene.
        """
        self.tracks = []
        BoTSORTTracklet.count_id = 0
        if self.cmc is not None:
            self.cmc.reset()
