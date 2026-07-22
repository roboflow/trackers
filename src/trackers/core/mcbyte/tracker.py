# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import supervision as sv
from deprecate import deprecated  # type: ignore[import-untyped]
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.mcbyte.mask_association import (
    MINIMUM_MASK_AVERAGE_CONFIDENCE,
    MINIMUM_MASK_COVERAGE,
    MINIMUM_MASK_FILL_RATIO,
    condition_similarity_with_masks,
)
from trackers.core.mcbyte.mask_manager import (
    MASK_CREATION_BBOX_OVERLAP_THRESHOLD,
    MaskManager,
)
from trackers.core.mcbyte.masks.base import MaskOutput, TrackletSnapshot
from trackers.core.mcbyte.tracklet import McByteTracklet
from trackers.core.mcbyte.utils import _fuse_score, get_alive_tracklets
from trackers.utils.cmc import CMC, CMCConfig, CMCMethod
from trackers.utils.detections import default_confidences
from trackers.utils.iou import BaseIoU, IoU
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCWHStateEstimator,
)


@dataclass(frozen=True)
class McByteMaskConfig:
    """Configuration for McByte's SAM and Cutie mask pipeline.

    The configuration is used only when ``McByteTracker`` automatically creates
    its default real ``MaskManager``. It is ignored when a custom manager is
    supplied directly.

    Args:
        device: Device shared by SAM and Cutie, for example ``"cuda"``,
            ``"cuda:0"``, or ``"cpu"``.
        sam_checkpoint_path: Optional SAM checkpoint path. When omitted, the
            default checkpoint for ``sam_model_type`` is used and downloaded
            automatically when necessary.
        sam_model_type: SAM model variant used for box-prompted mask generation.
        cutie_weights_path: Optional Cutie checkpoint path. When omitted, the
            default checkpoint for ``cutie_model_type`` is used and downloaded
            automatically when necessary.
        cutie_model_type: Cutie model variant used for temporal propagation.
        cutie_config_path: Optional Cutie Hydra configuration directory. When
            omitted, it is inferred from the installed Cutie package.
        cutie_config_name: Hydra configuration name loaded by Cutie.
        cutie_use_amp: Whether Cutie may use automatic mixed precision. AMP is
            activated only when Cutie runs on a CUDA device.
        mask_creation_bbox_overlap_threshold: Bounding-box overlap fraction at
            or above which mask creation is delayed by ``MaskManager``.
    """

    device: str = "cpu"

    sam_checkpoint_path: str | Path | None = None
    sam_model_type: str = "vit_b"

    cutie_weights_path: str | Path | None = None
    cutie_model_type: str = "base-mega"
    cutie_config_path: str | Path | None = None
    cutie_config_name: str = "eval_config"
    cutie_use_amp: bool = True

    mask_creation_bbox_overlap_threshold: float = MASK_CREATION_BBOX_OVERLAP_THRESHOLD


def _build_default_mask_manager(
    config: McByteMaskConfig,
) -> MaskManager:
    """Create McByte's standard SAM + Cutie mask-management pipeline."""

    from trackers.core.mcbyte.masks.cutie import CutieMaskPropagator
    from trackers.core.mcbyte.masks.sam import SAMBoxMaskGenerator

    mask_generator = SAMBoxMaskGenerator(
        checkpoint_path=config.sam_checkpoint_path,
        model_type=config.sam_model_type,
        device=config.device,
    )

    mask_propagator = CutieMaskPropagator(
        weights_path=config.cutie_weights_path,
        model_type=config.cutie_model_type,
        config_path=config.cutie_config_path,
        config_name=config.cutie_config_name,
        device=config.device,
        use_amp=config.cutie_use_amp,
    )

    return MaskManager(
        mask_generator=mask_generator,
        mask_propagator=mask_propagator,
        mask_creation_bbox_overlap_threshold=(config.mask_creation_bbox_overlap_threshold),
    )


class McByteTracker(BaseTracker):
    """McByte multi-object tracker with optional mask-conditioned association.

    McByte extends a ByteTrack-style multi-stage tracking pipeline with
    clear-match locking, reduced assignment, optional camera motion
    compensation, and optional propagated-mask evidence.

    The tracker can operate in two configurations:

    - without a ``MaskManager``, association uses the McByte clear-match locking
      and reduced-assignment procedure with IoU-based similarities;
    - with a ``MaskManager`` (full McByte), masks are additionally used to condition
      ambiguous associations and, when enabled, isolated positive-IoU associations
      below the normal stage threshold.

    When ``enable_mask_manager=True``, the default mask pipeline initializes
    masks from detection boxes using SAM and propagates them temporally using
    Cutie. A custom ``MaskManager`` may instead be supplied directly, for
    example to inject alternative mask components or lightweight test doubles.

    Mask processing follows the original McByte timing. At frame ``t``, masks
    are updated before association using the frame, visible tracklets, newly
    created tracklets, and removed-tracklet events stored after processing frame
    ``t - 1``. Temporarily lost but still active tracklets retain their masks.
    Masks are removed only after the corresponding tracklets are terminated
    during tracker pruning.

    Input frames are expected in RGB channel order. A frame is required when
    mask management is enabled and is also needed for camera motion
    compensation. When no frame is supplied, those frame-dependent operations
    are skipped.

    Args:
        lost_track_buffer: Time buffer, expressed as a number of frames at
            30 FPS, for retaining unmatched tracks before deletion. The value
            is scaled according to ``frame_rate``.
        frame_rate: Sequence frame rate used to scale ``lost_track_buffer``.
        track_activation_threshold: Minimum detection confidence required to
            create a new tracklet.
        minimum_consecutive_frames: Number of successful tracklet updates
            required before assigning a confirmed non-negative tracker ID.
        minimum_iou_threshold_first_assoc: Minimum association similarity for
            matching high-confidence detections to confirmed and lost tracks.
            The default of ``0.1`` is intentionally lower than in BoT-SORT and
            other trackers, allowing mask-conditioned association to evaluate
            a broader set of plausible candidates before resolving ambiguities
            and optional isolations.
        minimum_iou_threshold_second_assoc: Minimum association similarity for
            matching low-confidence detections to remaining tracked tracks.
        minimum_iou_threshold_unconfirmed_assoc: Minimum association similarity
            for matching unconfirmed tracks to remaining high-confidence
            detections.
        high_conf_det_threshold: Confidence threshold separating high- and
            low-confidence detections. Detections with confidence at or below
            0.1 are discarded.
        enable_cmc: Whether to apply camera motion compensation before
            association.
        cmc_method: Camera motion compensation method.
        cmc_downscale: Image downscale factor used during camera motion
            estimation.
        instant_first_frame_activation: Whether tracklets created on the first
            frame receive confirmed tracker IDs immediately.
        state_estimator_class: State estimator class used by newly created
            ``McByteTracklet`` instances.
        iou: IoU implementation used to compute association similarities. When
            omitted, the default ``IoU`` implementation is used.
        enable_mask_manager: Whether to construct McByte's default SAM and
            Cutie mask pipeline. It is disabled by default to avoid loading
            optional heavyweight models when mask-conditioned tracking is not
            requested.
        mask_manager: Optional custom ``MaskManager``. When supplied, it is used
            directly regardless of ``enable_mask_manager``, and automatic
            SAM/Cutie construction is skipped.
        mask_config: Configuration for automatic construction of the default
            SAM/Cutie pipeline. It requires ``enable_mask_manager=True`` and
            cannot be combined with a custom ``mask_manager``.
        minimum_mask_average_confidence: Minimum average confidence of a
            propagated mask before it may influence association.
        minimum_mask_coverage: Minimum fraction of the visible tracklet mask
            that must lie inside a candidate detection box.
        minimum_mask_fill_ratio: Minimum fraction of a candidate detection-box
            area that must be occupied by the tracklet mask.
        enable_isolated_mask_matching: Whether mask evidence may rescue an
            isolated candidate with positive IoU whose association similarity
            is below the normal stage threshold.
    """

    tracker_id = "mcbyte"

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold_first_assoc: float = 0.1,
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
        mask_config: McByteMaskConfig | None = None,
        minimum_mask_average_confidence: float = MINIMUM_MASK_AVERAGE_CONFIDENCE,
        minimum_mask_coverage: float = MINIMUM_MASK_COVERAGE,
        minimum_mask_fill_ratio: float = MINIMUM_MASK_FILL_RATIO,
        enable_isolated_mask_matching: bool = False,
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
        self.minimum_mask_average_confidence = minimum_mask_average_confidence
        self.minimum_mask_coverage = minimum_mask_coverage
        self.minimum_mask_fill_ratio = minimum_mask_fill_ratio
        self.enable_isolated_mask_matching = enable_isolated_mask_matching
        self.tracks: list[McByteTracklet] = []
        self.state_estimator_class = state_estimator_class
        self.iou = iou if iou is not None else IoU()
        self.frame_id: int = 0
        self._reset_id_allocator()

        self.enable_cmc = enable_cmc
        self.cmc = CMC(CMCConfig(method=cmc_method, downscale=cmc_downscale)) if enable_cmc else None

        self.mask_manager: MaskManager | None

        if mask_manager is not None and mask_config is not None:
            raise ValueError("mask_config cannot be used together with a custom mask_manager.")
        if mask_config is not None and not enable_mask_manager:
            raise ValueError("mask_config requires enable_mask_manager=True when no custom mask_manager is supplied.")
        if mask_manager is not None:
            self.mask_manager = mask_manager
        elif enable_mask_manager:
            self.mask_manager = _build_default_mask_manager(
                mask_config if mask_config is not None else McByteMaskConfig()
            )
        else:
            self.mask_manager = None

        self._previous_frame: np.ndarray | None = None
        self._previous_tracklets: list[TrackletSnapshot] = []
        self._last_mask_output: MaskOutput | None = None
        self._previous_new_tracklets: list[TrackletSnapshot] = []
        self._previous_removed_tracklet_ids: list[int] = []
        self._mask_tracklet_ids: set[int] = set()

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        """Update the tracker with detections from the current frame.

        This is the main per-frame entry point. If a mask manager is configured and a
        frame is provided, masks are updated before association using tracker lifecycle
        events stored from the previous call. After association, the method stores the
        current frame's visible tracklets, newly created tracklets, and explicitly
        terminated tracklet IDs for the next frame's mask update.

        Args:
            detections: Supervision detections for the current frame. Must include
                ``.xyxy``. Confidence (`detections.confidence`) is optional but
                recommended. This method does not mutate the input detections; it
                returns a new ``sv.Detections`` with ``tracker_id`` assigned.
            frame: Current RGB frame. Required for camera motion compensation and for
                mask-manager propagation.

        Returns:
            New sv.Detections with tracker_id assigned for each output detection.
            Confirmed tracks have tracker_id >= 0; unmatched/unconfirmed detections have
            tracker_id of -1.
        """
        # Accepted for compatibility with the current BaseTracker interface.
        # McByte currently does not use timestamps.
        _ = timestamp

        self.frame_id += 1

        # For the convenience and better understanding. McByte processes uses previous
        # frame and current frame. It is better to keep the method argument as "frame",
        # as in case of the other trackers.
        current_frame = frame
        terminated_tracklet_ids: list[int] = []

        if self.mask_manager is not None and current_frame is not None:
            self._last_mask_output = self.mask_manager.get_updated_masks(
                frame=current_frame,
                previous_frame=self._previous_frame,
                previous_tracklets=self._previous_tracklets,
                new_tracklets=self._previous_new_tracklets,
                removed_tracklet_ids=self._previous_removed_tracklet_ids,
            )
        else:
            self._last_mask_output = None

        if len(self.tracks) == 0 and len(detections) == 0:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            self._store_previous_mask_inputs(
                frame=current_frame,
                detections=result,
                removed_tracklet_ids=terminated_tracklet_ids,
            )
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
        raw_iou_similarity = self._get_iou_matrix(
            strack_pool,
            high_boxes,
        )
        association_similarity = _fuse_score(
            self.iou.normalize_for_fusion(raw_iou_similarity.copy()),
            high_scores,
        )

        matched, unmatched_pool, unmatched_high = self._get_mask_conditioned_associated_indices(
            similarity_matrix=association_similarity,
            raw_iou_similarity=raw_iou_similarity,
            tracklets=strack_pool,
            detection_boxes=high_boxes,
            min_similarity_thresh=self.minimum_iou_threshold_first_assoc,
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
        raw_iou_similarity = self._get_iou_matrix(
            remaining_tracked,
            low_boxes,
        )

        # There is no score fusion in stage 2, so the assignment matrix
        # and raw-IoU matrix are the same.
        matched, _, unmatched_low = self._get_mask_conditioned_associated_indices(
            similarity_matrix=raw_iou_similarity,
            raw_iou_similarity=raw_iou_similarity,
            tracklets=remaining_tracked,
            detection_boxes=low_boxes,
            min_similarity_thresh=self.minimum_iou_threshold_second_assoc,
        )

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

            raw_iou_similarity = self._get_iou_matrix(
                unconfirmed_tracks,
                uh_boxes,
            )
            association_similarity = _fuse_score(
                self.iou.normalize_for_fusion(raw_iou_similarity.copy()),
                uh_scores,
            )

            matched_uc, unmatched_uc_indices, remaining_uh = self._get_mask_conditioned_associated_indices(
                similarity_matrix=association_similarity,
                raw_iou_similarity=raw_iou_similarity,
                tracklets=unconfirmed_tracks,
                detection_boxes=uh_boxes,
                min_similarity_thresh=self.minimum_iou_threshold_unconfirmed_assoc,
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

        # Kill terminated tracks. Temporarily lost tracks remain alive and keep masks.
        tracklet_ids_before_pruning = {int(track.tracker_id) for track in self.tracks if track.tracker_id >= 0}
        self.tracks = get_alive_tracklets(
            tracklets=self.tracks,
            maximum_frames_without_update=self.maximum_frames_without_update,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )
        tracklet_ids_after_pruning = {int(track.tracker_id) for track in self.tracks if track.tracker_id >= 0}
        terminated_tracklet_ids = sorted(tracklet_ids_before_pruning - tracklet_ids_after_pruning)

        # Build final detections
        if not out_det_indices:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            self._store_previous_mask_inputs(
                frame=current_frame,
                detections=result,
                removed_tracklet_ids=terminated_tracklet_ids,
            )
            return result

        idx = np.array(out_det_indices)
        result = cast(sv.Detections, detections[idx])
        result.tracker_id = np.array(out_tracker_ids, dtype=int)
        self._store_previous_mask_inputs(
            frame=current_frame,
            detections=result,
            removed_tracklet_ids=terminated_tracklet_ids,
        )
        return result

    def _detections_to_tracklet_snapshots(
        self,
        detections: sv.Detections,
    ) -> list[TrackletSnapshot]:
        """Convert tracker output detections into mask-manager tracklet snapshots.

        Only detections with valid non-negative tracker IDs are converted. The returned
        snapshots contain the tracker ID and ``xyxy`` box needed by mask generators.
        """
        if detections.tracker_id is None:
            return []

        return [
            TrackletSnapshot(
                tracker_id=int(tracker_id),
                xyxy=xyxy.astype(np.float32),
            )
            for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id)
            if tracker_id >= 0
        ]

    def _store_previous_mask_inputs(
        self,
        frame: np.ndarray | None,
        detections: sv.Detections,
        removed_tracklet_ids: list[int],
    ) -> None:
        """Store tracker outputs and mask lifecycle events for the next frame.

        The mask manager consumes these values at the beginning of the next ``update()``
        call. New tracklets are detected among current visible outputs that do not yet
        have masks. Removed tracklets are provided explicitly from tracker pruning, so
        temporarily lost but still alive tracklets keep their masks.
        """
        if self.mask_manager is None or frame is None:
            self._previous_frame = None
            self._previous_tracklets = []
            self._previous_new_tracklets = []
            self._previous_removed_tracklet_ids = []
            self._mask_tracklet_ids = set()
            return

        # Convert current output detections into TrackletSnapshots.
        # Only valid tracker IDs are kept.
        current_tracklets = self._detections_to_tracklet_snapshots(detections)

        # Remove from the “tracks that already have masks” set
        # only the IDs that were truly terminated/pruned.
        removed_tracklet_id_set = set(removed_tracklet_ids)
        self._mask_tracklet_ids -= removed_tracklet_id_set

        # Find current visible tracklets that do not yet have masks.
        # These will be passed to SAM/Cutie on the next frame.
        new_tracklets = [
            tracklet for tracklet in current_tracklets if tracklet.tracker_id not in self._mask_tracklet_ids
        ]

        # Mark those new tracklets as now mask-managed, so if they disappear temporarily
        # and later reappear, they are not treated as new again.
        self._mask_tracklet_ids.update(tracklet.tracker_id for tracklet in new_tracklets)

        # Store lifecycle events from this frame. At the next update(),
        # MaskManager receives these and calls add_masks() / remove_masks().
        self._previous_new_tracklets = new_tracklets
        self._previous_removed_tracklet_ids = removed_tracklet_ids

        # Stores the current frame and current visible tracklets
        # as “previous” inputs for the next frame.
        self._previous_frame = frame
        self._previous_tracklets = current_tracklets

    def _get_iou_matrix(self, tracklets: list[McByteTracklet], detections: np.ndarray) -> np.ndarray:
        """Compute IoU similarity between tracklet states and detection boxes.

        Returns an ``(N, M)`` matrix where ``N`` is the number of tracklets and ``M`` is
        the number of detections. Empty inputs are handled by returning an empty matrix
        with the expected shape.
        """
        if len(tracklets) == 0:
            tracklet_boxes = np.empty((0, 4))
        else:
            tracklet_boxes = np.array([tracklet.get_state_bbox() for tracklet in tracklets])
        return self.iou.compute(tracklet_boxes, detections)

    def _get_mask_conditioned_associated_indices(
        self,
        similarity_matrix: np.ndarray,
        raw_iou_similarity: np.ndarray,
        tracklets: list[McByteTracklet],
        detection_boxes: np.ndarray,
        min_similarity_thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate tracklets and detections using McByte mask conditioning.

        Clear threshold-valid pairs are locked before assignment when they are the
        only eligible candidate in both their row and column. The remaining
        association problem is conditioned with propagated-mask evidence for
        ambiguous pairs and, optionally, isolated positive-IoU pairs below the
        normal threshold.

        Hungarian assignment is applied only to the remaining reduced matrix.
        Reduced row and column indices are then mapped back to the original
        ``tracklets`` and ``detection_boxes`` index spaces and combined with the
        locked matches.

        When no propagated mask output is available, mask-based score updates are
        skipped. Clear-match locking and assignment of the remaining problem still
        follow the McByte association pipeline.

        Args:
            similarity_matrix: Stage-specific association similarity matrix with
                shape ``(num_tracklets, num_detections)``. This is score-fused IoU
                for the first and unconfirmed association stages, and raw IoU for
                the second association stage.
            raw_iou_similarity: Unfused IoU similarity matrix with the same shape.
                It is used to determine optional isolated geometric candidates.
            tracklets: Tracklets corresponding, in order, to the rows of both
                similarity matrices.
            detection_boxes: Detection boxes in ``xyxy`` format corresponding, in
                order, to the columns of both similarity matrices.
            min_similarity_thresh: Minimum stage-specific similarity required for
                a valid association.

        Returns:
            A tuple containing:

            - matched original ``(tracklet_index, detection_index)`` pairs;
            - sorted original indices of unmatched tracklets;
            - sorted original indices of unmatched detections.
        """
        conditioned_association = condition_similarity_with_masks(
            similarity=similarity_matrix,
            raw_iou_similarity=raw_iou_similarity,
            tracklet_ids=[int(tracklet.tracker_id) for tracklet in tracklets],
            detection_boxes=detection_boxes,
            mask_output=self._last_mask_output,
            minimum_similarity=min_similarity_thresh,
            minimum_mask_average_confidence=self.minimum_mask_average_confidence,
            minimum_mask_coverage=self.minimum_mask_coverage,
            minimum_mask_fill_ratio=self.minimum_mask_fill_ratio,
            enable_isolated_mask_matching=self.enable_isolated_mask_matching,
        )

        (
            reduced_matches,
            reduced_unmatched_track_indices,
            reduced_unmatched_detection_indices,
        ) = self._get_associated_indices(
            similarity_matrix=conditioned_association.conditioned_similarity,
            min_similarity_thresh=min_similarity_thresh,
        )

        remapped_matches = [
            (
                conditioned_association.remaining_track_indices[reduced_track_index],
                conditioned_association.remaining_detection_indices[reduced_detection_index],
            )
            for reduced_track_index, reduced_detection_index in reduced_matches
        ]

        matched = sorted(conditioned_association.locked_matches + remapped_matches)

        unmatched_tracks = sorted(
            conditioned_association.remaining_track_indices[reduced_track_index]
            for reduced_track_index in reduced_unmatched_track_indices
        )

        unmatched_detections = sorted(
            conditioned_association.remaining_detection_indices[reduced_detection_index]
            for reduced_detection_index in reduced_unmatched_detection_indices
        )

        return matched, unmatched_tracks, unmatched_detections

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
                    tracklet.tracker_id = self._allocate_tracker_id()
                self.tracks.append(tracklet)
                out_det_indices.append(global_idx)
                out_tracker_ids.append(tracklet.tracker_id)

    def reset(self) -> None:
        """Reset tracker, camera-motion, and mask-manager state.

        This clears active tracklets, resets the global McByte track ID counter, clears
        stored mask lifecycle inputs, and resets optional camera motion compensation and
        mask-manager components. Call this when switching to a new video or scene.
        """
        self.tracks = []
        self.frame_id = 0
        self._reset_id_allocator()
        self._previous_frame = None
        self._previous_tracklets = []
        self._last_mask_output = None
        if self.mask_manager is not None:
            self.mask_manager.reset()
        if self.cmc is not None:
            self.cmc.reset()
        self._previous_new_tracklets = []
        self._previous_removed_tracklet_ids = []
        self._mask_tracklet_ids = set()

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
