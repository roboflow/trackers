# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import cast

import numpy as np
import supervision as sv
from deprecate import deprecated
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.mcbyte.tracklet import McByteTracklet
from trackers.core.mcbyte.utils import _fuse_score, get_alive_tracklets
from trackers.utils.cmc import CMC, CMCConfig, CMCMethod
from trackers.utils.detections import default_confidences
from trackers.utils.iou import BaseIoU, IoU
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCWHStateEstimator,
)
from trackers.core.mcbyte.mask_manager import MaskManager
from trackers.core.mcbyte.masks.base import MaskOutput, TrackletSnapshot
from trackers.core.mcbyte.masks.dummy import (
    DummyBoxMaskGenerator,
    DummyIdentityMaskPropagator,
)


class McByteTracker(BaseTracker):
    tracker_id = "mcbyte"

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
        cmc_method: CMCMethod = "sparseOptFlow",
        cmc_downscale: int = 2,
        instant_first_frame_activation: bool = True,
        state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator,
        iou: BaseIoU | None = None,
        enable_mask_manager: bool = False,
        mask_manager: MaskManager | None = None,
    ) -> None:
        # Calculate maximum frames without update based on lost_track_buffer and
        # frame_rate. This scales the buffer based on the frame rate to ensure
        # consistent time-based tracking across different frame rates.
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold_first_assoc = minimum_iou_threshold_first_assoc
        self.minimum_iou_threshold_second_assoc = minimum_iou_threshold_second_assoc
        self.minimum_iou_threshold_unconfirmed_assoc = minimum_iou_threshold_unconfirmed_assoc
        self.track_activation_threshold = track_activation_threshold
        self.high_conf_det_threshold = high_conf_det_threshold
        self.instant_first_frame_activation = instant_first_frame_activation
        self.tracks: list[McByteTracklet] = []
        self.state_estimator_class = state_estimator_class
        self.iou = iou if iou is not None else IoU()
        self.frame_id: int = 0

        self.enable_cmc = enable_cmc
        self.cmc = CMC(CMCConfig(method=cmc_method, downscale=cmc_downscale)) if enable_cmc else None

        self.mask_manager = mask_manager
        if self.mask_manager is None and enable_mask_manager:
            self.mask_manager = MaskManager(
                mask_generator=DummyBoxMaskGenerator(),
                mask_propagator=DummyIdentityMaskPropagator(),
            )

        self._previous_frame: np.ndarray | None = None
        self._previous_tracklets: list[TrackletSnapshot] = []
        self._last_mask_output: MaskOutput | None = None

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

        # For the convenience and better understanding. McByte processes uses previous 
        # frame and current frame. It is better to keep the method argument as "frame",
        # as in case of the other trackers.
        current_frame = frame

        if self.mask_manager is not None and current_frame is not None:
            self._last_mask_output = self.mask_manager.get_updated_masks(
                frame=current_frame,
                previous_frame=self._previous_frame,
                previous_tracklets=self._previous_tracklets,
            )
        else:
            self._last_mask_output = None
       
        if len(self.tracks) == 0 and len(detections) == 0:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            self._store_previous_mask_inputs(current_frame, result)
            return result

        out_det_indices: list[int] = []
        out_tracker_ids: list[int] = []

        # Predict new locations for existing tracks
        for tracker in self.tracks:
            tracker.predict()

        detection_boxes = detections.xyxy
        confidences = default_confidences(detections)

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
        confirmed_tracks: list[McByteTracklet] = []
        unconfirmed_tracks: list[McByteTracklet] = []
        lost_tracks: list[McByteTracklet] = []
        for track in self.tracks:
            if track.time_since_update > 1:
                lost_tracks.append(track)
            elif track.number_of_successful_updates >= self.minimum_consecutive_frames:
                confirmed_tracks.append(track)
            else:
                unconfirmed_tracks.append(track)

        # CMC: apply to all predicted tracks before association
        if self.enable_cmc and self.cmc is not None and current_frame is not None:
            mask_boxes = high_boxes if len(high_boxes) > 0 else None
            H = self.cmc.estimate(current_frame, mask_boxes)
            CMC.apply_batch(H, self.tracks)
        # Step 1: associate high-confidence detections to confirmed + lost tracks.
        # Lost tracks are included here (following the original ByteTrack), and
        # IoU is fused with detection scores.
        strack_pool = confirmed_tracks + lost_tracks
        iou_matrix = self._get_iou_matrix(strack_pool, high_boxes)
        iou_matrix = _fuse_score(self.iou.normalize_for_fusion(iou_matrix), high_scores)
        matched, unmatched_pool, unmatched_high = self._get_associated_indices(
            iou_matrix, self.minimum_iou_threshold_first_assoc
        )

        for row, col in matched:
            track = strack_pool[row]
            track.update(high_boxes[col])
            if track.number_of_successful_updates >= self.minimum_consecutive_frames and track.tracker_id == -1:
                track.tracker_id = McByteTracklet.get_next_tracker_id()
            out_det_indices.append(int(high_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        # Step 2: associate low-confidence detections to remaining *tracked* tracks
        # only (excluding lost tracks, following the original ByteTrack).
        # No score fusing in second association.
        remaining_tracked = [strack_pool[i] for i in unmatched_pool if strack_pool[i].time_since_update == 1]
        iou_matrix = self._get_iou_matrix(remaining_tracked, low_boxes)
        matched, _, unmatched_low = self._get_associated_indices(iou_matrix, self.minimum_iou_threshold_second_assoc)

        for row, col in matched:
            track = remaining_tracked[row]
            track.update(low_boxes[col])
            if track.number_of_successful_updates >= self.minimum_consecutive_frames and track.tracker_id == -1:
                track.tracker_id = McByteTracklet.get_next_tracker_id()
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

            iou_matrix = self._get_iou_matrix(unconfirmed_tracks, uh_boxes)
            iou_matrix = _fuse_score(self.iou.normalize_for_fusion(iou_matrix), uh_scores)
            matched_uc, unmatched_uc_indices, remaining_uh = self._get_associated_indices(
                iou_matrix, self.minimum_iou_threshold_unconfirmed_assoc
            )

            for row, col in matched_uc:
                track = unconfirmed_tracks[row]
                orig_high_idx = unmatched_high_list[col]
                track.update(high_boxes[orig_high_idx])
                if track.number_of_successful_updates >= self.minimum_consecutive_frames and track.tracker_id == -1:
                    track.tracker_id = McByteTracklet.get_next_tracker_id()
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
            self._store_previous_mask_inputs(current_frame, result)
            return result

        idx = np.array(out_det_indices)
        result = cast(sv.Detections, detections[idx])
        result.tracker_id = np.array(out_tracker_ids, dtype=int)
        self._store_previous_mask_inputs(current_frame, result)
        return result
    
    def _store_previous_mask_inputs(
        self,
        frame: np.ndarray | None,
        detections: sv.Detections,
    ) -> None:
        """Store current tracker output for mask preparation on the next frame."""
        self._previous_frame = None if frame is None else frame.copy()
        self._previous_tracklets = []

        if detections.tracker_id is None:
            return

        for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
            if tracker_id < 0:
                continue
            self._previous_tracklets.append(
                TrackletSnapshot(
                    tracker_id=int(tracker_id),
                    xyxy=xyxy.copy(),
                )
            )

    def _get_iou_matrix(self, tracklets: list[McByteTracklet], detections: np.ndarray) -> np.ndarray:
        if len(tracklets) == 0:
            tracklet_boxes = np.empty((0, 4))
        else:
            tracklet_boxes = np.array([tracklet.get_state_bbox() for tracklet in tracklets])
        return self.iou.compute(tracklet_boxes, detections)

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
            row_indices, col_indices = linear_sum_assignment(similarity_matrix, maximize=True)
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
                tracklet = McByteTracklet(
                    initial_bbox=detection_boxes[global_idx],
                    state_estimator_class=self.state_estimator_class,
                )
                if is_first_frame and self.instant_first_frame_activation:
                    tracklet.tracker_id = McByteTracklet.get_next_tracker_id()
                self.tracks.append(tracklet)
                out_det_indices.append(global_idx)
                out_tracker_ids.append(tracklet.tracker_id)

    def reset(self) -> None:
        """Reset tracker state by clearing all tracks and resetting ID counter.
        Call this method when switching to a new video or scene.
        """
        self.tracks = []
        self.frame_id = 0
        McByteTracklet.count_id = 0
        if self.cmc is not None:
            self.cmc.reset()

    @deprecated(target=None, deprecated_in="2.5", remove_in="3.0")
    def apply_cmc_batch(self, H: np.ndarray | None) -> None:
        """Apply CMC to all active tracks.

        .. deprecated:: 2.5
            Use CMC.apply_batch(H, self.tracks) directly.

        Args:
            H: 2x3 affine transform matrix returned by CMC.estimate().
                If None, this method is a no-op.

        Examples:
            >>> tracker = McByteTracker()
            >>> tracker.apply_cmc_batch(None)  # no-op
        """
        CMC.apply_batch(H, self.tracks)
