# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import ClassVar, cast

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.botsort.cmc import CMC, CMCConfig, CMCTMethod
from trackers.core.botsort.tracklet import BoTSORTTracklet
from trackers.core.botsort.utils import _fuse_score, get_alive_tracklets
from trackers.core.sort.utils import _get_iou_matrix
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCWHStateEstimator,
    XYXYStateEstimator,
)


class BoTSORTTracker(BaseTracker):
    """
    BoT-SORT-style multi-object tracker (IoU association + optional CMC).

    The tracker maintains a list of active tracks (Kalman-filter-based) and, for each
    frame, performs:
      1) Predict existing track states (Kalman predict)
      2) Split detections into high/low confidence groups
      3) Split tracks into confirmed, unconfirmed, and lost
      4) Apply camera motion compensation to predicted tracks
      5) Associate high-confidence detections to confirmed + lost tracks
         (IoU fused with detection scores + assignment)
      6) Associate low-confidence detections to remaining tracks
         (excluding lost tracks)
      7) Match remaining unmatched high-confidence detections to unconfirmed tracks
         and remove unmatched unconfirmed tracks
      8) Spawn new tracks from still unmatched high-confidence detections
         (instantly activated on the very first frame)
      9) Remove tracks that have been lost for too long

    Args:
        lost_track_buffer: Time buffer (in frames at 30 FPS) for keeping lost tracks
            alive before deletion. This is scaled by `frame_rate`.
        frame_rate: Video frame rate used to scale the lost track buffer to
            time-like behavior.
        track_activation_threshold: Minimum detection confidence to spawn a new
            track.
        minimum_consecutive_frames: Number of successful updates required before
            assigning a stable track ID (different than initial -1).
        minimum_iou_threshold_first_assoc: Minimum fused similarity (IoU x
            detection confidence) to accept a detection-track association during
            the first association step.
        minimum_iou_threshold_second_assoc: Minimum fused similarity (IoU x
            detection confidence) to accept a detection-track association during
            the second association step.
        minimum_iou_threshold_unconfirmed_assoc: Minimum fused similarity (IoU x
            score) to accept a match between an unconfirmed track and a remaining
            high-confidence detection.  Corresponds to the original ByteTrack's
            hardcoded cost threshold of 0.7 (= similarity 0.3).
        high_conf_det_threshold: Confidence threshold used to split detections into:
            - high confidence: confidence >= threshold
            - low confidence:  confidence < threshold
        enable_cmc: Whether to enable camera motion compensation (CMC).
        cmc_method: CMC method string passed into `CMCConfig(method=...)`.
            Supported values depend on `CMC` (e.g. "orb", "sift", "sparseOptFlow",
            "ecc"). See CMCConfig.
        cmc_downscale: Downscale factor used inside CMC for speed/robustness.
        instant_first_frame_activation: If ``True`` (default), tracks spawned on
            the very first frame receive a real tracker ID immediately. If ``False``,
            they start as unconfirmed (-1) and must survive
            ``minimum_consecutive_frames`` before getting an ID, matching the
            behaviour on every other frame.
        state_estimator_class: State estimator class for tracklets. Defaults
            to ``XCYCWHStateEstimator``.

    Notes:
        - `maximum_frames_without_update` is computed as:
            int(frame_rate / 30.0 * lost_track_buffer)
            to maintain consistent “seconds” worth of buffer across different FPS.
        - When CMC is enabled, pass the current video frame via the ``frame``
          argument of :meth:`update`.
    """

    tracker_id = "botsort"
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
        "cmc_downscale": {"type": "randint", "range": [1, 4]},
        "enable_cmc": {
            "type": "choice",
            "options": [False],
        },  # CMC disabled for tuner class until frame reading is added to tuner #
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
        enable_cmc: bool = True,
        cmc_method: CMCTMethod = "sparseOptFlow",
        cmc_downscale: int = 2,
        instant_first_frame_activation: bool = True,
        state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold_first_assoc = minimum_iou_threshold_first_assoc
        self.minimum_iou_threshold_second_assoc = minimum_iou_threshold_second_assoc
        self.minimum_iou_threshold_unconfirmed_assoc = (
            minimum_iou_threshold_unconfirmed_assoc
        )
        self.track_activation_threshold = track_activation_threshold
        self.high_conf_det_threshold = high_conf_det_threshold
        self.instant_first_frame_activation = instant_first_frame_activation
        self.tracks: list[BoTSORTTracklet] = []
        self.state_estimator_class = state_estimator_class
        self.frame_id: int = 0

        self.enable_cmc = enable_cmc
        self.cmc = (
            CMC(CMCConfig(method=cmc_method, downscale=cmc_downscale))
            if enable_cmc
            else None
        )

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
    ) -> sv.Detections:
        """
        Update the tracker with detections from the current frame.

        This is the main per-frame entry point.

        Args:
            detections: Supervision detections for the current frame. Must include
                ``.xyxy``. Confidence (`detections.confidence`) is optional but
                recommended. This method does not mutate the input detections;
                it returns a new ``sv.Detections`` with ``tracker_id`` assigned.

        Returns:
            New sv.Detections with tracker_id assigned for each detection.
            Confirmed tracks have tracker_id >= 0; unconfirmed tracks have
            tracker_id of -1.

        Notes:
            - If CMC is enabled, pass the current video frame via ``frame`` so the
              tracker can estimate a global affine transform and warp predicted
              track states before association.
        """
        self.frame_id += 1

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
            else np.ones(len(detections))
        )

        # Split indices into high / low / discarded by confidence
        high_mask = confidences >= self.high_conf_det_threshold
        low_mask = (confidences > 0.1) & (~high_mask)

        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]

        high_boxes = detection_boxes[high_indices]
        low_boxes = detection_boxes[low_indices]
        high_scores = confidences[high_indices]

        # Split tracks into confirmed, unconfirmed, and lost.
        # After predict(), time_since_update == 1 means the track was matched in
        # the previous frame ("tracked"), while time_since_update > 1 means the
        # track has been unmatched for multiple frames ("lost").
        confirmed_tracks: list[BoTSORTTracklet] = []
        unconfirmed_tracks: list[BoTSORTTracklet] = []
        lost_tracks: list[BoTSORTTracklet] = []
        for track in self.tracks:
            if track.time_since_update > 1:
                lost_tracks.append(track)
            elif track.number_of_successful_updates >= self.minimum_consecutive_frames:
                confirmed_tracks.append(track)
            else:
                unconfirmed_tracks.append(track)

        # CMC: apply to all predicted tracks before association
        if self.enable_cmc and self.cmc is not None and frame is not None:
            mask_boxes = high_boxes if len(high_boxes) > 0 else None
            H = self.cmc.estimate(frame, mask_boxes)
            if H is not None:
                self.apply_cmc_batch(H)
        # Step 1: associate high-confidence detections to confirmed + lost tracks.
        # Lost tracks are included here (following the original ByteTrack), and
        # IoU is fused with detection scores.
        strack_pool = confirmed_tracks + lost_tracks
        iou_matrix = _get_iou_matrix(strack_pool, high_boxes)
        iou_matrix = _fuse_score(iou_matrix, high_scores)
        matched, unmatched_pool, unmatched_high = self._get_associated_indices(
            iou_matrix, self.minimum_iou_threshold_first_assoc
        )

        for row, col in matched:
            track = strack_pool[row]
            track.update(high_boxes[col])
            if (
                track.number_of_successful_updates >= self.minimum_consecutive_frames
                and track.tracker_id == -1
            ):
                track.tracker_id = BoTSORTTracklet.get_next_tracker_id()
            out_det_indices.append(int(high_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        # Step 2: associate low-confidence detections to remaining *tracked* tracks
        # only (excluding lost tracks, following the original ByteTrack).
        # No score fusing in second association.
        remaining_tracked = [
            strack_pool[i]
            for i in unmatched_pool
            if strack_pool[i].time_since_update == 1
        ]
        iou_matrix = _get_iou_matrix(remaining_tracked, low_boxes)
        matched, _, unmatched_low = self._get_associated_indices(
            iou_matrix, self.minimum_iou_threshold_second_assoc
        )

        for row, col in matched:
            track = remaining_tracked[row]
            track.update(low_boxes[col])
            if (
                track.number_of_successful_updates >= self.minimum_consecutive_frames
                and track.tracker_id == -1
            ):
                track.tracker_id = BoTSORTTracklet.get_next_tracker_id()
            out_det_indices.append(int(low_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        # Unmatched low-confidence detections
        for det_local_idx in sorted(unmatched_low):
            out_det_indices.append(int(low_indices[det_local_idx]))
            out_tracker_ids.append(-1)

        # Step 3: match unconfirmed tracks with remaining unmatched high-confidence
        # detections (with score fusing, following the original ByteTrack).
        # Unmatched unconfirmed tracks are removed (not kept as lost).
        unmatched_high_list = sorted(unmatched_high)
        unmatched_uc_indices: list[int] = list(range(len(unconfirmed_tracks)))

        if len(unconfirmed_tracks) > 0 and len(unmatched_high_list) > 0:
            uh_boxes = high_boxes[unmatched_high_list]
            uh_scores = high_scores[unmatched_high_list]

            iou_matrix = _get_iou_matrix(unconfirmed_tracks, uh_boxes)
            iou_matrix = _fuse_score(iou_matrix, uh_scores)
            matched_uc, unmatched_uc_indices, remaining_uh = (
                self._get_associated_indices(
                    iou_matrix, self.minimum_iou_threshold_unconfirmed_assoc
                )
            )

            for row, col in matched_uc:
                track = unconfirmed_tracks[row]
                orig_high_idx = unmatched_high_list[col]
                track.update(high_boxes[orig_high_idx])
                if (
                    track.number_of_successful_updates
                    >= self.minimum_consecutive_frames
                    and track.tracker_id == -1
                ):
                    track.tracker_id = BoTSORTTracklet.get_next_tracker_id()
                out_det_indices.append(int(high_indices[orig_high_idx]))
                out_tracker_ids.append(track.tracker_id)

            # Only remaining unmatched high-conf dets proceed to spawning
            unmatched_high = [unmatched_high_list[i] for i in remaining_uh]

        # Remove unmatched unconfirmed tracks (following original ByteTrack,
        # which marks them as removed rather than keeping them as lost).
        if len(unmatched_uc_indices) > 0:
            remove_ids = {id(unconfirmed_tracks[i]) for i in unmatched_uc_indices}
            self.tracks = [t for t in self.tracks if id(t) not in remove_ids]

        # Spawn new tracks from unmatched high-confidence detections
        self._spawn_new_tracks(
            detection_boxes,
            confidences,
            unmatched_high,
            high_indices,
            out_det_indices,
            out_tracker_ids,
            is_first_frame=(self.frame_id == 1),
        )

        # Kill lost tracks
        self.tracks = get_alive_tracklets(
            tracklets=self.tracks,
            maximum_frames_without_update=self.maximum_frames_without_update,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )

        # Build final detections
        if not out_det_indices:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        idx = np.array(out_det_indices)
        result = cast(sv.Detections, detections[idx])
        result.tracker_id = np.array(out_tracker_ids, dtype=int)
        return result

    def apply_cmc_batch(self, H: np.ndarray | None) -> None:
        """Apply a 2x3 affine camera-motion transform to all tracklets at once."""
        if H is None or len(self.tracks) == 0:
            return

        R = H[:2, :2].astype(np.float64)
        t = H[:2, 2].astype(np.float64)

        first_estimator = self.tracks[0].state_estimator
        dim = first_estimator.kf.x.shape[0]
        is_xyxy = isinstance(first_estimator, XYXYStateEstimator)

        # Stack states (N, dim) and covariances (N, dim, dim)
        states = np.array([trk.state_estimator.kf.x.reshape(-1) for trk in self.tracks])
        Ps = np.array([trk.state_estimator.kf.P for trk in self.tracks])

        if is_xyxy:
            states[:, 0:2] = states[:, 0:2] @ R.T + t
            states[:, 2:4] = states[:, 2:4] @ R.T + t
            states[:, 4:6] = states[:, 4:6] @ R.T
            states[:, 6:8] = states[:, 6:8] @ R.T
        else:
            # Batch-transform centre positions: x' = x @ R.T + t
            states[:, 0:2] = states[:, 0:2] @ R.T + t
            # Batch-transform centre velocities: v' = v @ R.T
            states[:, 4:6] = states[:, 4:6] @ R.T

        A = np.eye(dim, dtype=np.float64)
        if is_xyxy:
            A[0:2, 0:2] = R
            A[2:4, 2:4] = R
            A[4:6, 4:6] = R
            A[6:8, 6:8] = R
        else:
            A[0:2, 0:2] = R
            A[4:6, 4:6] = R

        Ps = A @ Ps @ A.T

        for i, trk in enumerate(self.tracks):
            trk.state_estimator.kf.x = states[i].reshape(-1, 1)
            trk.state_estimator.kf.P = Ps[i]

    def _get_associated_indices(
        self,
        similarity_matrix: np.ndarray,
        min_similarity_thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
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
            matched: List of ``(tracker_idx, detection_idx)`` tuples for
                associations that meet the similarity threshold.
            unmatched_tracks: Sorted list of track indices not matched to any
                detection.
            unmatched_detections: Sorted list of detection indices not matched
                to any track.
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

        # Return sorted lists for deterministic order across Python runtimes.
        return matched_indices, sorted(unmatched_tracks), sorted(unmatched_detections)

    def _spawn_new_tracks(
        self,
        detection_boxes: np.ndarray,
        confidences: np.ndarray,
        unmatched_high_local: list[int],
        high_indices: np.ndarray,
        out_det_indices: list[int],
        out_tracker_ids: list[int],
        is_first_frame: bool = False,
    ) -> None:
        """Create new tracklets from unmatched high-confidence detections.

        On the very first frame, new tracklets are immediately activated with a
        real tracker ID, following the original ByteTrack convention where
        ``activate()`` sets ``is_activated = True`` only when
        ``frame_id == 1``.
        """
        for det_local_idx in unmatched_high_local:
            global_idx = int(high_indices[det_local_idx])
            conf = float(confidences[global_idx])
            if conf >= self.track_activation_threshold:
                tracklet = BoTSORTTracklet(
                    initial_bbox=detection_boxes[global_idx],
                    state_estimator_class=self.state_estimator_class,
                )
                if is_first_frame and self.instant_first_frame_activation:
                    tracklet.tracker_id = BoTSORTTracklet.get_next_tracker_id()
                self.tracks.append(tracklet)
                out_det_indices.append(global_idx)
                out_tracker_ids.append(tracklet.tracker_id)

    def reset(self) -> None:
        """Reset tracker state by clearing all tracks and resetting ID counter.
        Call this method when switching to a new video or scene.
        """
        self.tracks = []
        self.frame_id = 0
        BoTSORTTracklet.count_id = 0
        if self.cmc is not None:
            self.cmc.reset()
