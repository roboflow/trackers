# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import ClassVar, cast

import numpy as np
import supervision as sv

from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.botsort.tracklet import BoTSORTTracklet
from trackers.core.botsort.utils import _fuse_score, get_alive_tracklets
from trackers.utils.base_tracklet import BaseTracklet
from trackers.utils.detections import default_confidences
from trackers.utils.iou import BIoU
from trackers.utils.state_representations import BaseStateEstimator, XCYCWHStateEstimator


class CBIoUTracker(BoTSORTTracker):
    """Cascaded-Buffered IoU (C-BIoU) tracker.

    Implements the matching strategy from Yang et al., *Hard To Track Objects with
    Irregular Motions and Similar Appearances? Make It Easier by Buffering the
    Matching Space*, WACV 2023
    ([paper](https://openaccess.thecvf.com/content/WACV2023/papers/Yang_Hard_To_Track_Objects_With_Irregular_Motions_and_Similar_Appearances_WACV_2023_paper.pdf)).

    The paper proposes **Buffered IoU (BIoU)** — expanding boxes by a proportional
    margin before computing overlap — and **cascaded matching** with a small buffer
    scale ``b1`` followed by a larger scale ``b2`` (typically ``b1 < b2``).

    Each association step uses its own ``buffer_ratio``:

    * ``buffer_ratio_first`` — first pass (high-confidence detections vs tracks;
      paper: small ``b1``).
    * ``buffer_ratio_second`` — second pass (remaining *confirmed* tracks vs
      low-confidence detections; paper: large ``b2``).

    The ByteTrack-style unconfirmed-track step (leftover high-confidence
    detections vs tentative tracks) reuses **b1** (``iou_first``).

    Camera motion compensation is not used (detection-only / MOT-file workflows).

    Args:
        lost_track_buffer: Time buffer (in frames at 30 FPS) for keeping lost
            tracks alive before deletion. Scaled by ``frame_rate``.
        frame_rate: Video frame rate used to scale the lost track buffer.
        track_activation_threshold: Minimum detection confidence to spawn a
            new track.
        minimum_consecutive_frames: Number of successful updates required
            before assigning a stable track ID.
        minimum_iou_threshold_first_assoc: Minimum fused similarity for the
            first association step.
        minimum_iou_threshold_second_assoc: Minimum fused similarity for the
            second association step.
        minimum_iou_threshold_unconfirmed_assoc: Minimum fused similarity for
            the unconfirmed association step.
        high_conf_det_threshold: Confidence threshold splitting high / low
            detections.
        instant_first_frame_activation: If ``True``, first-frame tracks receive
            a real ID immediately.
        state_estimator_class: Kalman state representation for tracklets.
        buffer_ratio_first: Buffer scale ``b1`` for the first BIoU pass. It is suggested to
            be **less than** ``buffer_ratio_second`` (``b1 < b2``) per the paper.
        buffer_ratio_second: Buffer scale ``b2`` for the second BIoU pass. It is suggested to
            be **greater than** ``buffer_ratio_first``.

    Raises:
        ValueError: If ``lost_track_buffer`` is negative or ``frame_rate`` is not a
            finite positive value (inherited from ``BoTSORTTracker``).
        ValueError: If ``buffer_ratio_first`` or ``buffer_ratio_second`` is negative.

    Note:
        Unmatched low-confidence detections (confidence in ``(0.1, high_conf_det_threshold)``)
        that are not associated in Step 2 appear in the output with ``tracker_id == -1``,
        consistent with ``BoTSORTTracker`` behaviour. Callers filtering by
        ``tracker_id >= 0`` will silently drop these rows.

    Example:
        Run C-BIoU on a batch of detections::

            import numpy as np
            import supervision as sv
            from trackers import CBIoUTracker

            tracker = CBIoUTracker()
            detections = sv.Detections(
                xyxy=np.array([[0.0, 0.0, 100.0, 100.0]]),
                confidence=np.array([0.9]),
            )
            result = tracker.update(detections)
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
        "minimum_consecutive_frames": {"type": "randint", "range": [1, 3]},
        "buffer_ratio_first": {"type": "uniform", "range": [0.0, 0.7]},
        "buffer_ratio_second": {"type": "uniform", "range": [0.0, 0.7]},
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
        buffer_ratio_first: float = 0.3,
        buffer_ratio_second: float = 0.5,
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
        )
        self.iou_first = BIoU(buffer_ratio=buffer_ratio_first)
        self.iou_second = BIoU(buffer_ratio=buffer_ratio_second)
        self.buffer_ratio_first = buffer_ratio_first
        self.buffer_ratio_second = buffer_ratio_second

    def _biou_matrix(
        self,
        tracklets: list[BoTSORTTracklet],
        boxes: np.ndarray,
        iou: BIoU,
    ) -> np.ndarray:
        if len(tracklets) == 0:
            track_boxes = np.empty((0, 4))
        else:
            track_boxes = np.array([t.get_state_bbox() for t in tracklets])
        return iou.compute(track_boxes, boxes)

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        """Update the C-BIoU tracker with detections from the current frame.

        Runs the association pipeline with a distinct BIoU instance per step
        (cascaded buffers per Yang et al., WACV 2023). Does not use frames or CMC.

        Args:
            detections: Supervision detections for the current frame.
            frame: Unused. Emits a ``UserWarning`` if provided.
            timestamp: Absolute time of the current frame in seconds, or ``None``
                for fixed-rate mode (``frame_step = 1.0`` per call).

        Returns:
            Detections with ``tracker_id`` assigned. Unmatched
            low-confidence detections are included with ``tracker_id == -1``;
            callers filtering by ``tracker_id >= 0`` will silently drop these rows.

        Warns:
            UserWarning: If ``frame`` is passed but C-BIoU does not perform
                camera motion compensation (CMC), the frame is ignored.
        """
        timing = self._predict_timing(timestamp)
        if timing.skip_update:
            return self._detections_for_skipped_update(detections)
        self._warn_if_frame_unused(frame)
        self.frame_id += 1

        if len(self.tracks) == 0 and len(detections) == 0:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        out_det_indices: list[int] = []
        out_tracker_ids: list[int] = []

        # Predict new locations for existing tracks
        self._predict_tracklets(self.tracks, timing)

        # Ghost-ID prevention: budget-only filter before association (variable-FPS only).
        # At fixed frame rate the frame-count budget is enforced post-association, so
        # tracks keep their last-frame re-association opportunity.
        _budget = self._lost_track_time_budget(timing, self.maximum_time_without_update)
        if _budget is not None:
            self.tracks = [
                t
                for t in self.tracks
                if BaseTracklet.within_lost_track_budget(
                    t,
                    maximum_frames_without_update=self.maximum_frames_without_update,
                    maximum_time_without_update=_budget,
                )
            ]

        detection_boxes = detections.xyxy
        confidences = default_confidences(detections)

        # Split detections into high / low / discarded by confidence
        high_mask = confidences >= self.high_conf_det_threshold
        low_mask = (confidences > 0.1) & (~high_mask)
        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]
        high_boxes = detection_boxes[high_indices]
        low_boxes = detection_boxes[low_indices]
        high_scores = confidences[high_indices]

        # Split tracks into confirmed, unconfirmed, and lost.
        # After predict(), time_since_update == 1 means "tracked"; > 1 means "lost".
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

        # Step 1: associate high-confidence detections to confirmed + lost tracks.
        # Paper b1 (small buffer); BIoU fused with detection scores.
        strack_pool = confirmed_tracks + lost_tracks
        iou_matrix = self._biou_matrix(strack_pool, high_boxes, self.iou_first)
        iou_matrix = _fuse_score(self.iou_first.normalize_for_fusion(iou_matrix), high_scores)
        matched, unmatched_pool, unmatched_high = self._get_associated_indices(
            iou_matrix, self.minimum_iou_threshold_first_assoc
        )

        for row, col in matched:
            track = strack_pool[row]
            track.update(high_boxes[col])
            if track.number_of_successful_updates >= self.minimum_consecutive_frames and track.tracker_id == -1:
                track.tracker_id = self._allocate_tracker_id()
            out_det_indices.append(int(high_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        # Step 2: associate low-confidence detections to remaining *tracked* tracks
        # only (excluding lost tracks). Paper b2 (large buffer); no score fusion.
        remaining_tracked = [strack_pool[i] for i in unmatched_pool if strack_pool[i].time_since_update == 1]
        iou_matrix = self._biou_matrix(remaining_tracked, low_boxes, self.iou_second)
        matched, _, unmatched_low = self._get_associated_indices(iou_matrix, self.minimum_iou_threshold_second_assoc)

        for row, col in matched:
            track = remaining_tracked[row]
            track.update(low_boxes[col])
            if track.number_of_successful_updates >= self.minimum_consecutive_frames and track.tracker_id == -1:
                track.tracker_id = self._allocate_tracker_id()
            out_det_indices.append(int(low_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        # Unmatched low-confidence detections (output with tracker_id=-1)
        for det_local_idx in sorted(unmatched_low):
            out_det_indices.append(int(low_indices[det_local_idx]))
            out_tracker_ids.append(-1)

        # Step 3: match unconfirmed tracks with remaining unmatched high-confidence
        # detections (ByteTrack lifecycle; reuses b1 / iou_first).
        # Unmatched unconfirmed tracks are removed (not kept as lost).
        unmatched_high_list = sorted(unmatched_high)
        unmatched_uc_indices: list[int] = list(range(len(unconfirmed_tracks)))

        if len(unconfirmed_tracks) > 0 and len(unmatched_high_list) > 0:
            uh_boxes = high_boxes[unmatched_high_list]
            uh_scores = high_scores[unmatched_high_list]
            iou_matrix = self._biou_matrix(unconfirmed_tracks, uh_boxes, self.iou_first)
            iou_matrix = _fuse_score(self.iou_first.normalize_for_fusion(iou_matrix), uh_scores)
            matched_uc, unmatched_uc_indices, remaining_uh = self._get_associated_indices(
                iou_matrix, self.minimum_iou_threshold_unconfirmed_assoc
            )

            for row, col in matched_uc:
                track = unconfirmed_tracks[row]
                orig_high_idx = unmatched_high_list[col]
                track.update(high_boxes[orig_high_idx])
                if track.number_of_successful_updates >= self.minimum_consecutive_frames and track.tracker_id == -1:
                    track.tracker_id = self._allocate_tracker_id()
                out_det_indices.append(int(high_indices[orig_high_idx]))
                out_tracker_ids.append(track.tracker_id)

            # Only remaining unmatched high-conf dets proceed to spawning
            unmatched_high = [unmatched_high_list[i] for i in remaining_uh]

        # Remove unmatched unconfirmed tracks (following original ByteTrack)
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

        # Full lifecycle prune: removes immature+unmatched and any remaining expired
        self.tracks = get_alive_tracklets(
            tracklets=self.tracks,
            maximum_frames_without_update=self.maximum_frames_without_update,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            maximum_time_without_update=_budget,
        )

        if not out_det_indices:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        # Build final detections
        idx = np.array(out_det_indices)
        result = cast(sv.Detections, detections[idx])
        result.tracker_id = np.array(out_tracker_ids, dtype=int)
        return result
