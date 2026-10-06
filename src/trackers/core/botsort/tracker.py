# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import ClassVar, cast

import numpy as np
import supervision as sv
from deprecate import TargetMode, deprecated
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.botsort.tracklet import BoTSORTTracklet
from trackers.core.botsort.utils import _fuse_score, get_alive_tracklets
from trackers.utils.cmc import CMC, CMCConfig, CMCMethod
from trackers.utils.detections import default_confidences
from trackers.utils.iou import BaseIoU, IoU
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCWHStateEstimator,
)


class BoTSORTTracker(BaseTracker):
    """BoT-SORT-style multi-object tracker (IoU association + optional CMC).

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
        lost_track_buffer: Non-negative time buffer (in frames at 30 FPS) for
            keeping lost tracks alive before deletion. `0` deletes a confirmed
            track on the first missed frame. This is scaled by `frame_rate`.
        frame_rate: Video frame rate used to scale the lost track buffer to
            time-like behavior. Must be positive.
        track_activation_threshold: Minimum detection confidence to spawn a new
            track.
        minimum_consecutive_frames: Number of successful updates required before
            assigning a stable track ID (different than initial -1).
        minimum_iou_threshold_first_assoc: Minimum fused similarity (IoU x
            detection confidence) to accept a detection-track association during
            the first association step.
        minimum_iou_threshold_second_assoc: Minimum IoU to accept a
            detection-track association during the second association step.
            No score fusion is applied in this pass, so this is plain IoU.
        minimum_iou_threshold_unconfirmed_assoc: Minimum fused similarity (IoU x
            score) to accept a match between an unconfirmed track and a remaining
            high-confidence detection.  Corresponds to the original ByteTrack's
            hardcoded cost threshold of 0.7 (= similarity 0.3).
        high_conf_det_threshold: Confidence threshold used to split detections into:
            - high confidence: confidence >= threshold
            - low confidence:  confidence < threshold
        enable_cmc: Whether to build the internal frame-based camera motion
            compensation (CMC) estimator, used when ``update()`` receives a
            ``frame``. A precomputed ``cmc_transform`` passed to ``update()`` is applied
            regardless of this flag.
        cmc_method: CMC method string passed into `CMCConfig(method=...)`.
            Supported values: "orb", "sift", "sparseOptFlow", "ecc". See CMCConfig.
        cmc_downscale: Downscale factor used inside CMC for speed/robustness.
        instant_first_frame_activation: If ``True`` (default), tracks spawned on
            the very first frame receive a real tracker ID immediately. If ``False``,
            they start as unconfirmed (-1) and must survive
            ``minimum_consecutive_frames`` before getting an ID, matching the
            behaviour on every other frame.
        state_estimator_class: State estimator class for tracklets. Defaults
            to ``XCYCWHStateEstimator``.
        iou: IoU similarity metric instance to use for data association.
            Defaults to standard `IoU`. Can be replaced with any `BaseIoU`
            subclass (e.g. GIoU, DIoU, CIoU) to change how bounding-box
            similarity is computed during association.
            Passing ``None`` (the default) is equivalent to ``IoU()`` and is
            provided for backward compatibility with existing code that did not
            supply an ``iou`` argument.

    Notes:
        - Positive `maximum_frames_without_update` values are scaled by
          ``frame_rate`` and rounded up to at least one missed frame. Explicit
          zero-buffer configurations remain zero.
        - Camera motion compensation in :meth:`update` takes its transform from
          one of two alternatives: the current video frame via ``frame``
          (estimated internally, requires ``enable_cmc=True``) or a precomputed
          2x3 affine via the keyword-only ``cmc_transform`` (applied regardless of
          ``enable_cmc``). Passing both raises ``ValueError``; passing neither
          skips CMC for that step.
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
        cmc_method: CMCMethod = "sparseOptFlow",
        cmc_downscale: int = 2,
        instant_first_frame_activation: bool = True,
        state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator,
        iou: BaseIoU | None = None,
    ) -> None:
        self.maximum_frames_without_update = self._compute_maximum_frames_without_update(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
        )
        self.maximum_time_without_update: float = lost_track_buffer / 30.0
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold_first_assoc = minimum_iou_threshold_first_assoc
        self.minimum_iou_threshold_second_assoc = minimum_iou_threshold_second_assoc
        self.minimum_iou_threshold_unconfirmed_assoc = minimum_iou_threshold_unconfirmed_assoc
        self.track_activation_threshold = track_activation_threshold
        self.high_conf_det_threshold = high_conf_det_threshold
        self.instant_first_frame_activation = instant_first_frame_activation
        self.tracks: list[BoTSORTTracklet] = []
        self.state_estimator_class = state_estimator_class
        self.iou = iou if iou is not None else IoU()
        self.frame_id: int = 0
        self._reset_id_allocator()

        self.enable_cmc = enable_cmc
        self.cmc = CMC(CMCConfig(method=cmc_method, downscale=cmc_downscale)) if enable_cmc else None

        self._init_timestamp_state(frame_rate)

    @staticmethod
    def _validate_cmc_matrix(cmc_transform: np.ndarray) -> np.ndarray:
        """Check a precomputed camera-motion transform before any tracker state changes.

        ``update()`` calls this before predict, so a rejected ``cmc_transform`` leaves tracks,
        IDs and the timestamp anchor untouched (same contract as
        ``_validate_detections``).

        Args:
            cmc_transform: Value passed to ``update(cmc_transform=...)``. Must be a real-valued
                ``(2, 3)`` affine transform mapping previous-frame to current-frame
                pixel coordinates, the same convention as ``CMC.estimate()``.

        Returns:
            The transform as a new ``float64`` array of shape ``(2, 3)``.

        Raises:
            ValueError: If ``cmc_transform`` is not a real-valued numeric array, does not
                have shape ``(2, 3)`` (3x3 homographies are rejected), or contains
                NaN or inf.

        Example:
            >>> import numpy as np
            >>> shift = np.array([[1, 0, 5], [0, 1, -2]])
            >>> BoTSORTTracker._validate_cmc_matrix(shift).tolist()
            [[1.0, 0.0, 5.0], [0.0, 1.0, -2.0]]
        """
        try:
            matrix = np.asarray(cmc_transform)
        except ValueError as exc:
            raise ValueError(
                f"cmc_transform must be a numeric array of shape (2, 3); conversion failed: {exc}"
            ) from exc
        if matrix.dtype.kind not in "iuf":
            raise ValueError(f"cmc_transform must contain real numbers, got dtype {matrix.dtype}")
        if matrix.shape != (2, 3):
            hint = (
                "; 3x3 homographies are not accepted, pass the per-frame previous-to-current 2x3 affine"
                if matrix.shape == (3, 3)
                else ""
            )
            raise ValueError(f"cmc_transform must have shape (2, 3), got {matrix.shape}{hint}")
        if not np.isfinite(matrix).all():
            raise ValueError("cmc_transform contains non-finite values (NaN or inf)")
        return matrix.astype(np.float64)

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
        *,
        cmc_transform: np.ndarray | None = None,
    ) -> sv.Detections:
        """
        Update the tracker with detections from the current frame.

        This is the main per-frame entry point.

        Args:
            detections: Supervision detections for the current frame. Must include
                ``.xyxy``. Confidence (`detections.confidence`) is optional but
                recommended. This method does not mutate the input detections;
                it returns a new ``sv.Detections`` with ``tracker_id`` assigned.
            frame: Current video frame in BGR format (H, W, 3), or ``None``.
                When given and ``enable_cmc=True``, the internal CMC estimator
                computes the camera motion from it. Mutually exclusive with
                ``cmc_transform``.
            timestamp: Absolute time of the current frame in seconds, or ``None``
                for fixed-rate mode (``frame_step = 1.0`` per call).
            cmc_transform: Keyword-only precomputed camera-motion transform for this
                step, or ``None``. A ``(2, 3)`` affine matrix mapping
                previous-frame to current-frame pixel coordinates (original image
                scale), the same convention as ``CMC.estimate()``. Applied
                regardless of ``enable_cmc``. Mutually exclusive with ``frame``.

        Returns:
            New sv.Detections with tracker_id assigned for each detection.
            Confirmed tracks have tracker_id >= 0; unconfirmed tracks have
            tracker_id of -1.

        Raises:
            ValueError: If ``detections.xyxy`` contains NaN or inf, if both
                ``frame`` and ``cmc_transform`` are given, or if ``cmc_transform`` is not a
                finite real-valued ``(2, 3)`` matrix. All checks run before any
                tracker state is advanced, so a failed call leaves the tracker
                unchanged.

        Warns:
            UserWarning: If ``timestamp`` is earlier than the previous call
                (backwards order); the whole update is skipped and all output
                IDs are ``-1``. If ``timestamp`` equals the previous call
                (duplicate); predict is skipped but association still runs on
                the last state.

        Notes:
            - Camera motion compensation (CMC) warps predicted track states before
              association. The transform comes from one of two alternatives: pass
              the current video frame via ``frame`` so the internal estimator
              (built when ``enable_cmc=True``) computes it, or pass a precomputed
              transform via ``cmc_transform``, which is applied regardless of
              ``enable_cmc``.
            - Passing both ``frame`` and ``cmc_transform`` raises ``ValueError``. When both
              are ``None``, CMC is silently skipped for that step; ``frame`` alone
              is also ignored when ``enable_cmc=False``.
            - ``cmc_transform`` is a per-frame ``(2, 3)`` affine mapping previous-frame to
              current-frame pixel coordinates, the convention of
              ``CMC.estimate()``. A 3x3 homography, such as the cumulative one
              emitted by ``MotionEstimator``, is rejected.
            - Applying ``cmc_transform`` resets the internal estimator, so switching back
              to ``frame`` re-initializes it (one uncompensated step) instead of
              compensating the externally handled motion twice.
        """
        self._validate_detections(detections)
        if frame is not None and cmc_transform is not None:
            raise ValueError(
                "update() received both frame and cmc_transform; pass frame for the internal CMC estimate "
                "or cmc_transform for a precomputed transform, not both"
            )
        if cmc_transform is not None:
            cmc_transform = self._validate_cmc_matrix(cmc_transform)
        timing = self._predict_timing(timestamp)
        if timing.skip_update:
            return self._detections_for_skipped_update(detections)
        self.frame_id += 1

        if len(self.tracks) == 0 and len(detections) == 0:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        out_det_indices: list[int] = []
        out_tracker_ids: list[int] = []

        # Predict new locations for existing tracks
        self._predict_tracklets(self.tracks, timing)

        # Ghost-ID prevention: budget-only filter before association.
        # Keeps immature tracks alive for matching; full lifecycle prune runs after.
        _budget = self._lost_track_time_budget(timing, self.maximum_time_without_update)
        self._prune_lost_tracks(timing)

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
        confirmed_tracks: list[BoTSORTTracklet] = []
        unconfirmed_tracks: list[BoTSORTTracklet] = []
        lost_tracks: list[BoTSORTTracklet] = []
        for track in self.tracks:
            if track.time_since_update > 1:
                lost_tracks.append(track)
            elif track.tracker_id != -1 or track.number_of_successful_updates >= self.minimum_consecutive_frames:
                # Maturity is sticky: a track that already holds a real
                # tracker_id (e.g. an instant-activated first-frame track) stays
                # confirmed even before it reaches minimum_consecutive_frames.
                # On a miss it is kept as a confirmed (then eventually lost)
                # track rather than discarded as an unconfirmed one.
                confirmed_tracks.append(track)
            else:
                unconfirmed_tracks.append(track)

        # CMC: apply to all predicted tracks before association
        # A precomputed cmc_transform is applied regardless of enable_cmc, which only
        # controls the internal frame-based estimator (self.cmc).
        H: np.ndarray | None = None
        if cmc_transform is not None:
            H = cmc_transform
            if self.cmc is not None:
                # Motion for this step is compensated externally; drop the internal
                # estimator's previous-frame state so a later frame-based call
                # re-initializes instead of re-applying motion already compensated.
                self.cmc.reset()
        elif self.cmc is not None and frame is not None:
            mask_boxes = high_boxes if len(high_boxes) > 0 else None
            H = self.cmc.estimate(frame, mask_boxes)
        CMC.apply_batch(H, self.tracks)  # None (no CMC this step) is a no-op

        # Cache each tracklet's predicted state bbox once per update. Track state
        # is unchanged across the three association stages: a track matched in an
        # earlier stage never re-enters a later one, and the CMC adjustment above
        # is already applied. Recomputing get_state_bbox() per stage would be
        # redundant, so all stages read boxes from this map keyed by ``id()``.
        predicted_state_boxes = {id(track): track.get_state_bbox() for track in self.tracks}

        # Step 1: associate high-confidence detections to confirmed + lost tracks.
        # Lost tracks are included here (following the original ByteTrack), and
        # IoU is fused with detection scores.
        strack_pool = confirmed_tracks + lost_tracks
        iou_matrix = self._get_iou_matrix(strack_pool, high_boxes, predicted_state_boxes)
        iou_matrix = _fuse_score(self.iou.normalize_for_fusion(iou_matrix), high_scores)
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
        # only (excluding lost tracks, following the original ByteTrack).
        # No score fusing in second association.
        remaining_tracked = [strack_pool[i] for i in unmatched_pool if strack_pool[i].time_since_update == 1]
        iou_matrix = self._get_iou_matrix(remaining_tracked, low_boxes, predicted_state_boxes)
        matched, _, unmatched_low = self._get_associated_indices(iou_matrix, self.minimum_iou_threshold_second_assoc)

        for row, col in matched:
            track = remaining_tracked[row]
            track.update(low_boxes[col])
            if track.number_of_successful_updates >= self.minimum_consecutive_frames and track.tracker_id == -1:
                track.tracker_id = self._allocate_tracker_id()
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

            iou_matrix = self._get_iou_matrix(unconfirmed_tracks, uh_boxes, predicted_state_boxes)
            iou_matrix = _fuse_score(self.iou.normalize_for_fusion(iou_matrix), uh_scores)
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

        # Full lifecycle prune: removes immature+unmatched and any remaining expired
        self.tracks = get_alive_tracklets(
            tracklets=self.tracks,
            maximum_frames_without_update=self.maximum_frames_without_update,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            maximum_time_without_update=_budget,
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

    def _get_iou_matrix(
        self,
        tracklets: list[BoTSORTTracklet],
        detections: np.ndarray,
        tracklet_boxes_by_id: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Compute IoU similarity between tracklet states and detection boxes.

        Args:
            tracklets: Tracklets forming the rows of the returned matrix.
            detections: Detection boxes in ``xyxy`` format forming the columns.
            tracklet_boxes_by_id: Mapping from ``id(track)`` to the track's
                predicted state bbox, computed once per ``update()`` and reused
                across association stages to avoid recomputing ``get_state_bbox``.

        Raises:
            KeyError: If a tracklet passed in is absent from ``tracklet_boxes_by_id``
                — an internal-invariant violation, since the map is built from
                ``self.tracks`` and every tracklet here is drawn from it.
        """
        if len(tracklets) == 0:
            tracklet_boxes = np.empty((0, 4))
        else:
            try:
                tracklet_boxes = np.array([tracklet_boxes_by_id[id(tracklet)] for tracklet in tracklets])
            except KeyError as exc:
                raise KeyError(
                    f"tracklet id {exc.args[0]} missing from the per-frame decode-once box cache; "
                    "tracklet_boxes_by_id must contain every tracklet passed to this helper "
                    "(it is built from self.tracks once per update())"
                ) from exc
        return self.iou.compute(tracklet_boxes, detections)

    def _get_associated_indices(
        self,
        similarity_matrix: np.ndarray,
        min_similarity_thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate detections to tracks based on Similarity (IoU) using the Jonker-Volgenant algorithm approach with
        no initialization instead of the Hungarian algorithm as mentioned in the SORT paper, but it solves the
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
            out_det_indices.append(global_idx)
            if conf >= self.track_activation_threshold:
                tracklet = BoTSORTTracklet(
                    initial_bbox=detection_boxes[global_idx],
                    state_estimator_class=self.state_estimator_class,
                )
                if is_first_frame and self.instant_first_frame_activation:
                    tracklet.tracker_id = self._allocate_tracker_id()
                self.tracks.append(tracklet)
                out_tracker_ids.append(tracklet.tracker_id)
            else:
                out_tracker_ids.append(-1)

    def reset(self) -> None:
        """Reset tracker state by clearing all tracks and resetting ID counter.

        Call this method when switching to a new video or scene.
        """
        self.tracks = []
        self.frame_id = 0
        self._last_timestamp = None
        self._reset_id_allocator()
        if self.cmc is not None:
            self.cmc.reset()

    @deprecated(target=TargetMode.NOTIFY, deprecated_in="2.5", remove_in="3.0")
    def apply_cmc_batch(self, H: np.ndarray | None) -> None:
        """Apply CMC to all active tracks.

        .. deprecated:: 2.5
            Use CMC.apply_batch(H, self.tracks) directly.

        Args:
            H: 2x3 affine transform matrix returned by CMC.estimate().
                If None, this method is a no-op.

        Examples:
            >>> tracker = BoTSORTTracker()
            >>> tracker.apply_cmc_batch(None)  # no-op
        """
        CMC.apply_batch(H, self.tracks)
