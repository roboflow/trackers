# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""McByte++ tracking with Kalman motion, camera compensation and optional EdgeTAM masks.

The tracker owns detection association and mask creation, refresh and exclusion policies. The shared mask manager
coordinates generation and propagation; its default EdgeTAM backend owns model inference and bounded video-session
history.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment

from trackers.core.base import BaseTracker
from trackers.core.botsort.utils import _fuse_score
from trackers.core.masks.base import MaskOutput, TrackletSnapshot
from trackers.core.masks.edgetam import DEFAULT_EDGETAM_MODEL
from trackers.core.masks.manager import MaskManager
from trackers.core.mcbyteplusplus.mask_association import (
    MINIMUM_MASK_COVERAGE,
    MINIMUM_MASK_FILL_RATIO,
    _get_mask_metrics,
    condition_similarity_with_masks,
)
from trackers.core.mcbyteplusplus.tracklet import McBytePlusPlusTracklet
from trackers.core.mcbyteplusplus.utils import _get_alive_tracklets
from trackers.utils.cmc import CMC, CMCConfig, CMCMethod
from trackers.utils.detections import default_confidences
from trackers.utils.iou import BaseIoU, IoU
from trackers.utils.state_representations import BaseStateEstimator, XCYCWHStateEstimator

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_MASK_FAILURES = 3
_MINIMUM_DETECTION_CONFIDENCE = 0.1


def _is_out_of_memory(exc: BaseException) -> bool:
    """Return whether an exception represents a torch out-of-memory error.

    Detected without importing ``torch`` (an optional EdgeTAM dependency): a
    ``torch.cuda.OutOfMemoryError`` is a ``RuntimeError`` subclass, so it is
    recognised by class name, and plain out-of-memory ``RuntimeError``
    instances are recognised by their canonical message. This also covers the
    MPS backend, whose OOM is a plain ``RuntimeError`` with an
    ``"MPS backend out of memory"`` message.

    Args:
        exc: Exception raised by the mask pipeline.

    Returns:
        ``True`` when ``exc`` is a torch out-of-memory error (CUDA or MPS),
        otherwise ``False``.
    """
    if any(klass.__name__ == "OutOfMemoryError" for klass in type(exc).__mro__):
        return True
    return "out of memory" in str(exc).lower()


@dataclass(frozen=True)
class McBytePlusPlusMaskConfig:
    """Configure automatic EdgeTAM mask-pipeline construction for McByte++.

    Pass this configuration as ``edgetam_config`` with
    ``enable_mask_manager=True`` and no custom ``mask_manager``. This dataclass
    only stores settings; validation and model loading happen when the tracker
    builds the default pipeline. That pipeline loads one EdgeTAM model and its
    image/prompt processor. The same backend
    creates masks from detection boxes and propagates those masks to later
    images. This configuration also controls the box-overlap heuristic used to
    defer mask creation or refreshing when the target may be occluded.

    An EdgeTAM video session contains processed frames, image features and
    object-propagation memory. When its frame limit is reached, the backend
    discards that session and initializes another from the last processed image
    and all remaining objects' masks before processing the next image. This is
    session rollover: model weights remain loaded, object IDs are retained, and
    older propagation history is discarded. It does not create new box masks.

    Args:
        model_id: Hugging Face repository ID or local directory containing a
            Transformers-compatible EdgeTAM model and processor. Defaults to
            ``yonigozlan/EdgeTAM-hf``. Model files are loaded through Hugging
            Face; they are not bundled in the Trackers package.
        revision: Hugging Face branch, tag or commit ID passed to both the model
            and processor loaders. With ``revision=None`` and the default
            ``model_id="yonigozlan/EdgeTAM-hf"``, Trackers explicitly selects
            commit ``c266ce53b3fc00f0f495b583f6a116c4e57f53bb`` (the
            ``DEFAULT_EDGETAM_REVISION`` constant in ``core/masks/edgetam.py``).
            With a different Hub repository and ``revision=None``, no commit is
            pinned: Hugging Face loads its default ``main`` revision. An
            explicit revision overrides either choice; use a commit ID to fix
            the model version for reproducible runs. A local ``model_id`` loads
            the files in that directory rather than selecting a Hub revision.
        device: PyTorch inference device. ``"auto"`` selects CUDA when available,
            otherwise CPU; explicit devices such as ``"cpu"`` or ``"mps"`` are
            also accepted. The default backend runs in float32.
        max_session_frames: Maximum images processed in one EdgeTAM video
            session, including its initial image. Defaults to 200; must be an
            integer of at least two, or ``None`` to disable session rollover.
            The transferred boundary image counts as the first image of the
            new session. The limit applies to files and live streams alike;
            it bounds history length, not total RAM or device memory. Disabling
            the limit can exhaust available memory on long sequences; see
            ``McBytePlusPlusTracker.update`` for the tracker's out-of-memory handling.
        mask_threshold: Finite logit threshold applied independently to every
            object's mask. Pixels with logits strictly above this value become
            foreground. Object masks may overlap; this is not a mask-confidence
            threshold. Defaults to 0.0.
        mask_creation_bbox_overlap_threshold: Occlusion threshold in ``[0, 1]``.
            For each target box, take the maximum of intersection area divided
            by target-box area over other boxes whose bottom ``y2`` is greater
            (lower in the image). Creation and refresh are deferred when this
            ratio is at least the threshold. Defaults to 0.1. No such box gives
            a ratio of zero, so a threshold of zero defers every candidate.
    """

    model_id: str = DEFAULT_EDGETAM_MODEL
    revision: str | None = None
    device: str = "auto"
    max_session_frames: int | None = 200
    mask_threshold: float = 0.0
    mask_creation_bbox_overlap_threshold: float = 0.1


def _build_default_mask_manager(config: McBytePlusPlusMaskConfig) -> MaskManager:
    """Build the shared mask manager with one EdgeTAM backend in both roles.

    ``MaskManager`` accepts separate ``mask_generator`` and ``mask_propagator``
    objects, as used by ``McByteTracker`` with SAM and Cutie. Here a single
    ``EdgeTAMMaskBackend`` instance is passed twice: as the generator to create
    masks from boxes, and as the propagator to maintain masks across images.
    Its model and processor are loaded once. Box generation uses temporary
    sessions separate from the persistent propagation session.

    Args:
        config: Model, device, session-limit, mask-threshold and occlusion settings.

    Returns:
        A manager whose generator and propagator reference the same backend.

    Raises:
        ValueError: If the overlap threshold or backend settings are invalid.
        ImportError: If the optional EdgeTAM dependencies are unavailable.
    """
    from trackers.core.masks.edgetam import EdgeTAMMaskBackend

    overlap = config.mask_creation_bbox_overlap_threshold
    if not np.isfinite(overlap) or not 0 <= overlap <= 1:
        raise ValueError("mask_creation_bbox_overlap_threshold must be finite and in [0, 1].")
    backend = EdgeTAMMaskBackend(
        model_id=config.model_id,
        revision=config.revision,
        device=config.device,
        max_session_frames=config.max_session_frames,
        mask_threshold=config.mask_threshold,
    )
    return MaskManager(backend, backend, mask_creation_bbox_overlap_threshold=overlap)


class McBytePlusPlusTracker(BaseTracker):
    """Track detections with Kalman motion, camera compensation and optional EdgeTAM masks.

    This independent tracker associates detections in three stages: high-confidence
    detections with confirmed and lost tracks, low-confidence detections with the
    remaining recently tracked objects, and remaining high-confidence detections
    with tentative tracks. Clear matches are locked first; eligible mask evidence
    adjusts ambiguous matches before assignment. Unmatched high-confidence
    detections can create tracks, and tracks expire after their lost-track budget.

    Each ``update`` predicts motion, prepares masks for the supplied image, applies
    camera motion compensation (CMC), performs association, and stores image/box
    pairs for the next mask update. BGR images go to CMC; their RGB conversion goes
    to the mask pipeline. The default backend uses Hugging Face Transformers'
    ``EdgeTamVideoModel`` with ``Sam2VideoProcessor``, whose streaming API accepts
    new images one at a time. This choice of EdgeTAM implementation means future
    images and the final sequence length are not required by the mask pipeline.
    Automatic mask-pipeline construction is disabled by default; enable it with
    ``enable_mask_manager=True`` or supply a custom ``mask_manager``.

    The default pipeline uses one EdgeTAM backend for both box-to-mask generation
    and temporal mask propagation. Creation and refresh use confirmed,
    detector-matched boxes on the stored image that supplied those boxes, then
    propagate the resulting masks to the next supplied image. A box-overlap
    heuristic defers prompts for potentially occluded objects. Temporarily lost
    tracks keep their mask state until they expire. EdgeTAM supplies independent,
    potentially overlapping boolean masks without confidence scores.

    When mask-trust filtering is enabled, an empty propagated mask is excluded
    before association. A displaced mask is identified after association by the
    fraction of its foreground inside its matched detection box, and is excluded
    from subsequent associations. These checks do not use predicted boxes.
    Exclusion retains the backend's object memory; association use resumes only
    after a box refresh and its next propagated mask pass the recovery checks.

    EdgeTAM stores image features and object history in a video session. To limit
    memory growth from retaining this history over long sequences, the backend
    replaces the session before processing another image once its limit is reached:
    by default, 200 images including the initial image. This session rollover
    transfers the last image and remaining masks, retains object IDs and loaded
    model weights, and discards older history. It does not generate new box masks
    or by itself restore trust in an excluded mask. The bound is set through
    ``edgetam_config``. It limits retained history rather than guaranteeing a fixed
    RAM/VRAM budget; image resolution and object count also affect memory use.

    Internal tracklet IDs own masks; public IDs are returned to users. This
    separation prepares for planned long-term tracking with re-identification
    (re-ID): multiple tracklets could later share one public identity while each
    keeps its own mask ownership. Both IDs currently have the same assigned value;
    long-term re-ID and selective CMC are not implemented yet. See ``update`` for
    timing, return-value and failure rules.

    Args:
        lost_track_buffer: Missed-frame budget in 30 FPS units, scaled by frame_rate.
        frame_rate: Positive reference FPS used to scale the lost-frame budget
            and convert timestamp gaps from seconds to Kalman frame units.
        track_activation_threshold: Minimum confidence to create a new tracklet.
        minimum_consecutive_frames: Successful observations required for confirmation,
            counting the initial detection. Despite the parameter name, this counter
            is cumulative and is not reset on a missed detection.
        minimum_iou_threshold_first_assoc: Similarity threshold for high-confidence matches.
        minimum_iou_threshold_second_assoc: IoU threshold for low-confidence matches.
        minimum_iou_threshold_unconfirmed_assoc: Similarity threshold for tentative tracks.
        high_conf_det_threshold: Inclusive high-confidence cutoff, strictly above 0.1.
            Scores at or below 0.1 are discarded; intermediate scores enter stage two.
        enable_cmc: Enable shared camera motion compensation on BGR images.
        cmc_method: Camera motion estimator, defaulting to the research ORB preset.
        cmc_downscale: Downscale factor for camera motion estimation.
        instant_first_frame_activation: Immediately confirm tracks created on the first update.
        state_estimator_class: Motion-state representation used by new tracklets.
        iou: Similarity implementation; defaults to standard IoU.
        enable_mask_manager: Build the default EdgeTAM pipeline when no custom manager is given.
            Disabled by default, so constructing the tracker without a custom manager
            does not load EdgeTAM unless this flag is true.
        mask_manager: Optional custom manager, used regardless of enable_mask_manager.
            Supplying one skips automatic EdgeTAM construction. Its propagator must
            support refresh_masks when enable_mask_refresh is true.
        edgetam_config: Settings for automatic construction of the EdgeTAM pipeline.
            Requires enable_mask_manager=True and cannot be combined with a custom mask_manager.
        minimum_mask_coverage: Minimum mask fraction inside a candidate detection for association.
        minimum_mask_fill_ratio: Minimum foreground fraction of a candidate detection for association.
        enable_isolated_mask_matching: Allow mask evidence for isolated positive-IoU candidates.
        minimum_mask_creation_frames: Consecutive accepted images with a confirmed,
            detector-matched tracklet before its box becomes eligible for mask
            creation. Creation can still be deferred by the occlusion heuristic.
        enable_mask_refresh: Re-prompt empty, displaced or excluded masks using visible matched boxes.
            Enabled by default; has no effect without a mask manager and does not construct one.
        mask_refresh_coverage_threshold: Minimum fraction of a mask's foreground
            inside its matched detection box. A strictly smaller fraction marks
            the mask as displaced for refresh and, when enabled, trust filtering.
            Empty masks remain eligible even when this threshold is zero.
        mask_refresh_interval: Minimum processed images between refreshes of the same tracklet.
        enable_mask_trust_filter: Exclude suspect masks until a validated refresh.
            Enabled by default; has no effect without a mask manager and does not construct one.
            Independent of enable_mask_refresh; with refresh disabled, excluded masks
            stay excluded until removal or reset. When false, association uses
            available propagated masks without persistent exclusion.

    Example:
        >>> tracker = McBytePlusPlusTracker(enable_cmc=False)
        >>> detections = sv.Detections(xyxy=np.array([[0., 0., 10., 20.]]))
        >>> tracker.update(detections).tracker_id.tolist()
        [0]
        >>> tracker.reset()
    """

    tracker_id = "mcbyteplusplus"

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
        cmc_method: CMCMethod = "orb",
        cmc_downscale: int = 4,
        instant_first_frame_activation: bool = True,
        state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator,
        iou: BaseIoU | None = None,
        enable_mask_manager: bool = False,
        mask_manager: MaskManager | None = None,
        edgetam_config: McBytePlusPlusMaskConfig | None = None,
        minimum_mask_coverage: float = MINIMUM_MASK_COVERAGE,
        minimum_mask_fill_ratio: float = MINIMUM_MASK_FILL_RATIO,
        enable_isolated_mask_matching: bool = False,
        minimum_mask_creation_frames: int = 1,
        enable_mask_refresh: bool = True,
        mask_refresh_coverage_threshold: float = 0.5,
        mask_refresh_interval: int = 1,
        enable_mask_trust_filter: bool = True,
    ) -> None:
        self.maximum_frames_without_update = self._compute_maximum_frames_without_update(lost_track_buffer, frame_rate)
        self.maximum_time_without_update = lost_track_buffer / 30.0
        for name, value in {
            "track_activation_threshold": track_activation_threshold,
            "high_conf_det_threshold": high_conf_det_threshold,
            "minimum_iou_threshold_first_assoc": minimum_iou_threshold_first_assoc,
            "minimum_iou_threshold_second_assoc": minimum_iou_threshold_second_assoc,
            "minimum_iou_threshold_unconfirmed_assoc": minimum_iou_threshold_unconfirmed_assoc,
            "minimum_mask_coverage": minimum_mask_coverage,
            "minimum_mask_fill_ratio": minimum_mask_fill_ratio,
            "mask_refresh_coverage_threshold": mask_refresh_coverage_threshold,
        }.items():
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if high_conf_det_threshold <= 0.1:
            raise ValueError("high_conf_det_threshold must be greater than 0.1.")
        for name, value in {
            "minimum_consecutive_frames": minimum_consecutive_frames,
            "minimum_mask_creation_frames": minimum_mask_creation_frames,
            "mask_refresh_interval": mask_refresh_interval,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if edgetam_config is not None and (mask_manager is not None or not enable_mask_manager):
            raise ValueError("edgetam_config requires enable_mask_manager=True without a custom mask_manager.")

        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.track_activation_threshold = track_activation_threshold
        self.high_conf_det_threshold = high_conf_det_threshold
        self.minimum_iou_threshold_first_assoc = minimum_iou_threshold_first_assoc
        self.minimum_iou_threshold_second_assoc = minimum_iou_threshold_second_assoc
        self.minimum_iou_threshold_unconfirmed_assoc = minimum_iou_threshold_unconfirmed_assoc
        self.instant_first_frame_activation = instant_first_frame_activation
        self.state_estimator_class = state_estimator_class
        self.iou = iou if iou is not None else IoU()
        self.minimum_mask_coverage = minimum_mask_coverage
        self.minimum_mask_fill_ratio = minimum_mask_fill_ratio
        self.enable_isolated_mask_matching = enable_isolated_mask_matching
        self.enable_cmc = enable_cmc
        self.cmc = CMC(CMCConfig(method=cmc_method, downscale=cmc_downscale)) if enable_cmc else None
        self.mask_manager = mask_manager
        if mask_manager is None and enable_mask_manager:
            self.mask_manager = _build_default_mask_manager(edgetam_config or McBytePlusPlusMaskConfig())
        self.minimum_mask_creation_frames = minimum_mask_creation_frames
        self.enable_mask_refresh = enable_mask_refresh
        self.mask_refresh_coverage_threshold = mask_refresh_coverage_threshold
        self.mask_refresh_interval = mask_refresh_interval
        self.enable_mask_trust_filter = enable_mask_trust_filter
        # EdgeTAM provides that extra method. A different propagator supplied
        # through a custom MaskManager might not provide it.
        refresh = getattr(self.mask_manager.mask_propagator, "refresh_masks", None) if self.mask_manager else None
        if self.mask_manager is not None and enable_mask_refresh and not callable(refresh):
            raise ValueError("Mask refresh requires a propagator with refresh_masks; set enable_mask_refresh=False.")
        self._refresh_masks = cast(Callable[[np.ndarray, MaskOutput], None], refresh)
        self._mask_manager_original = self.mask_manager
        self._consecutive_mask_failures = 0
        self._previous_frame: np.ndarray | None = None
        self._previous_tracklets: list[TrackletSnapshot] = []
        self._previous_new_tracklets: list[TrackletSnapshot] = []
        self._previous_removed_tracklet_ids: set[int] = set()
        self._previous_reference_tracklets: list[TrackletSnapshot] = []
        self._previous_refresh_tracklets: list[TrackletSnapshot] = []
        self._last_mask_output: MaskOutput | None = None
        self._mask_tracklet_ids: set[int] = set()
        self._untrusted_mask_ids: set[int] = set()
        self._refreshed_mask_ids: set[int] = set()
        self._mask_pending_ages: dict[int, int] = {}
        self._mask_last_refresh: dict[int, int] = {}
        self._mask_image_count = 0
        self.tracks: list[McBytePlusPlusTracklet] = []
        self.frame_id = 0
        self._reset_id_allocator()
        self._init_timestamp_state(frame_rate)

    @property
    def last_mask_output(self) -> MaskOutput | None:
        """Return raw current-image masks, including those excluded from association."""
        return self._last_mask_output

    @property
    def untrusted_mask_ids(self) -> frozenset[int]:
        """Return internal tracklet IDs whose masks are excluded from association."""
        return frozenset(self._untrusted_mask_ids)

    @property
    def tracked_objects(self) -> sv.Detections:
        """Return alive confirmed motion estimates with their public identities."""
        result = super().tracked_objects
        result.tracker_id = np.array([t.public_id for t in self.tracks if t.tracker_id != -1], dtype=int)
        return result

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        """Associate one set of detections and advance McByte++ tracking state.

        For an accepted update, the processing order is:

        1. Validate inputs and determine elapsed time. Predict tracklet motion
           and advance missed-detection clocks. In timestamp mode, remove tracks
           whose time budget has expired before they can be associated again.
        2. If a mask manager and a new image are available, convert BGR to RGB.
           Apply queued mask refreshes, creations and removals using boxes from
           the previously accepted image, then propagate masks to the current
           image. The EdgeTAM backend performs session rollover here if needed.
           Empty masks are excluded before association when trust filtering is on.
        3. Split detections by confidence and apply CMC to predicted track states
           using the current BGR image. Match high-confidence detections to
           confirmed/lost tracks, low-confidence detections to remaining recently
           tracked objects, then remaining high-confidence detections to tentative
           tracks. Stages one and three fuse detection scores with IoU; stage two
           uses IoU directly. Each stage locks clear matches and can use eligible
           masks to condition the remaining assignment problem. Correct matched
           motion states and confirm eligible tracks after each association stage.
        4. Remove unmatched tentative tracks and create tracks from qualifying unmatched
           high-confidence detections. Prune tracks beyond their lost-track budget.
        5. Assess masks against matched detection boxes after association. Store a
           copy of the accepted RGB image with its boxes and queue eligible mask
           creation, refresh and removal work for the next image. Finally map
           internal tracklet IDs to public IDs in the returned detections.

        A displaced mask first detected in step five has already been available
        to this call's association; exclusion affects later calls. An excluded
        mask becomes usable on a later call only after both its replacement mask
        and the immediately following propagated mask pass coverage/fill checks
        against suitable matched boxes. The backend retains excluded masks.

        Timestamps control Kalman elapsed time and track expiry, not the number
        of EdgeTAM inference steps. Sampling every Nth source frame is supported:
        pass those images with their capture timestamps, and masks advance once
        per supplied, accepted image once initialized, without requiring intermediate frames.
        An explicit ``frame=None`` call still associates detections but skips CMC
        and masks. It discards queued box prompts and consecutive creation counts,
        while retaining existing propagation memory and queued removals. New
        prompts must be collected from later detector-matched images.

        A duplicate timestamp skips prediction, CMC and masks but still permits
        detection association and correction. If an image is supplied, the last
        accepted image/box pairs, queued prompts and creation counts are retained;
        their masks are not reused as evidence on the duplicate call. A non-finite
        or backward timestamp skips all tracking work after input validation.

        A mask-pipeline out-of-memory error resets mask state, logs the failure
        and continues association without masks. Three such failures without a
        successful mask result disable the manager until ``reset()``. Calls that
        cannot produce masks do not clear that failure count. Other mask-pipeline
        errors reset mask state and propagate to the caller; the whole tracker
        update is not rolled back.

        Args:
            detections: Current-image detections with finite, positive-area
                ``xyxy`` boxes in original-image pixel coordinates. Confidences
                must be finite when supplied; absent confidence means 1.0.
                Input tracker IDs do not drive association. Classes are preserved
                as metadata but do not restrict matching.
            frame: Nonempty BGR ``uint8`` image with shape ``(H, W, 3)``, or
                ``None`` for detection-only processing. Keep a constant image
                resolution during an active EdgeTAM session; reset the tracker
                before starting a sequence with a different resolution.
            timestamp: Capture time in seconds, or ``None`` for one nominal
                Kalman step per call. Increasing times use
                ``(timestamp - previous_timestamp) * frame_rate`` frame units;
                the first timestamped call uses one nominal step. Track expiry
                uses elapsed seconds in timestamp mode. Use one timing mode per
                sequence: passing ``None`` clears the timestamp anchor.

        Returns:
            A new ``sv.Detections`` containing input detection boxes and metadata,
            with public ``tracker_id`` values for confirmed tracks and ``-1`` for
            unmatched or unconfirmed detections. Scores at or below 0.1 are omitted.
            Order follows processing: stage-one matches, stage-two matches,
            unmatched low-confidence detections, stage-three matches, then remaining
            high-confidence detections considered for new tracks. Lost-track
            predictions are not added. If a timestamp causes the whole update to
            be skipped, all input detections are returned with IDs set to ``-1``.

        Raises:
            ValueError: If boxes, confidences or the supplied image are invalid,
                or the mask backend rejects an image or mask contract. Initial
                input validation occurs before tracking state changes.
            RuntimeError: If mask inference or its returned object IDs violate
                the backend contract, except for the handled out-of-memory case.
                Other backend exceptions also propagate after mask-state cleanup.

        Warns:
            UserWarning: If a timestamp is non-finite, backward or duplicated.
        """
        self._validate_inputs(detections, frame)
        timing = self._predict_timing(timestamp)
        if timing.skip_update:
            return self._detections_for_skipped_update(detections)

        self.frame_id += 1
        current_frame = frame
        preserve_mask_observation = timing.skip_predict and current_frame is not None and self.mask_manager is not None
        out_det_indices: list[int] = []
        out_tracker_ids: list[int] = []

        # Expire tracks before propagation so EdgeTAM never returns evidence for
        # an ID whose timestamp budget has already elapsed.
        self._predict_tracklets(self.tracks, timing)
        _budget = self._lost_track_time_budget(timing, self.maximum_time_without_update)
        tracklet_ids_before_early_prune = self._alive_tracklet_ids()
        self._prune_lost_tracks(timing)
        early_pruned_tracklet_ids = tracklet_ids_before_early_prune - self._alive_tracklet_ids()
        self._merge_pending_removals(sorted(early_pruned_tracklet_ids))

        mask_frame = None
        if self.mask_manager is not None and current_frame is not None and not timing.skip_predict:
            mask_frame = np.ascontiguousarray(current_frame[..., ::-1])
            self._last_mask_output = self._run_mask_manager(self.mask_manager, mask_frame)
        elif preserve_mask_observation:
            # Keep the accepted image/box pairs, but do not reuse their masks
            # as evidence for detections supplied on the duplicate call.
            self._last_mask_output = None
        else:
            self._invalidate_mask_observation()

        detection_boxes = detections.xyxy
        confidences = default_confidences(detections)

        # Split indices into high / low / discarded by confidence
        high_mask = confidences >= self.high_conf_det_threshold
        low_mask = (confidences > _MINIMUM_DETECTION_CONFIDENCE) & (~high_mask)

        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]

        high_boxes = detection_boxes[high_indices]
        low_boxes = detection_boxes[low_indices]
        high_scores = confidences[high_indices]

        # Split tracks into confirmed, unconfirmed, and lost.
        # After predict(), time_since_update == 1 means the track was matched in
        # the previous frame ("tracked"), while time_since_update > 1 means the
        # track has been unmatched for multiple frames ("lost").
        confirmed_tracks: list[McBytePlusPlusTracklet] = []
        unconfirmed_tracks: list[McBytePlusPlusTracklet] = []
        lost_tracks: list[McBytePlusPlusTracklet] = []
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
        if self.enable_cmc and self.cmc is not None and current_frame is not None and not timing.skip_predict:
            mask_boxes = high_boxes if len(high_boxes) > 0 else None
            H = self.cmc.estimate(current_frame, mask_boxes)
            CMC.apply_batch(H, self.tracks)

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
        raw_iou_similarity = self._get_iou_matrix(
            strack_pool,
            high_boxes,
            predicted_state_boxes,
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
                track.public_id = track.tracker_id
            out_det_indices.append(int(high_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        # Step 2: associate low-confidence detections to remaining *tracked* tracks
        # only (excluding lost tracks, following the original ByteTrack).
        # No score fusing in second association.
        remaining_tracked = [strack_pool[i] for i in unmatched_pool if strack_pool[i].time_since_update == 1]
        raw_iou_similarity = self._get_iou_matrix(
            remaining_tracked,
            low_boxes,
            predicted_state_boxes,
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
                track.public_id = track.tracker_id
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
                predicted_state_boxes,
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
                    track.public_id = track.tracker_id
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
        tracklet_ids_before_pruning = self._alive_tracklet_ids()
        self.tracks = _get_alive_tracklets(
            tracklets=self.tracks,
            maximum_frames_without_update=self.maximum_frames_without_update,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            maximum_time_without_update=_budget,
        )
        tracklet_ids_after_pruning = self._alive_tracklet_ids()
        # Union with whatever the early ghost-ID prune above already removed, so those
        # tracks are reported too instead of silently vanishing from mask cleanup.
        terminated_tracklet_ids = sorted(
            early_pruned_tracklet_ids | (tracklet_ids_before_pruning - tracklet_ids_after_pruning)
        )

        # Mask snapshots consume internal IDs before public re-ID mapping.
        result = cast(sv.Detections, detections[np.array(out_det_indices, dtype=int)])
        result.tracker_id = np.array(out_tracker_ids, dtype=int)
        if preserve_mask_observation:
            # Association still runs on duplicates, so honour any terminations
            # without replacing or ageing the accepted mask observation.
            self._merge_pending_removals(terminated_tracklet_ids)
        else:
            self._store_previous_mask_inputs(mask_frame, result, terminated_tracklet_ids)
        public_ids = {track.tracker_id: track.public_id for track in self.tracks if track.tracker_id >= 0}
        result.tracker_id = np.array([public_ids.get(key, -1) for key in out_tracker_ids], dtype=int)
        return result

    def _validate_inputs(self, detections: sv.Detections, frame: np.ndarray | None) -> None:
        """Reject invalid observations before IDs, clocks, or model memory change."""
        self._validate_detections(detections)
        if len(detections) and np.any(detections.xyxy[:, 2:] <= detections.xyxy[:, :2]):
            raise ValueError("Detection boxes must have positive area.")
        if not np.isfinite(default_confidences(detections)).all():
            raise ValueError("Detection confidences must be finite.")
        if frame is not None and (
            frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1 or frame.dtype != np.uint8
        ):
            raise ValueError("frame must be a nonempty BGR uint8 image with shape (H, W, 3).")

    def _run_mask_manager(
        self,
        mask_manager: MaskManager,
        current_frame: np.ndarray,
    ) -> MaskOutput | None:
        """Refresh stored RGB prompts, apply lifecycle events, then propagate.

        Actual backend ownership is adopted only after validating the complete ID set. Any failure resets potentially
        partial EdgeTAM memory. OOM falls back to geometry, as in McByte, and disables masks after three failed
        inference attempts. Deferred/no-image calls do not interrupt that failure streak.
        """
        self._mask_image_count += 1
        # A refresh may be validated only on the image propagated immediately after it.
        self._refreshed_mask_ids.clear()
        removed = self._previous_removed_tracklet_ids
        snapshots = [t for t in self._previous_tracklets if t.tracker_id not in removed]
        new_tracklets = [t for t in self._previous_new_tracklets if t.tracker_id not in removed]
        expected_ids = (self._mask_tracklet_ids - removed) | {
            t.tracker_id
            for t in snapshots
            if not mask_manager._is_tracklet_occluded(t, self._previous_reference_tracklets)
        }
        try:
            self._refresh_previous_masks(mask_manager)
            mask_output = mask_manager.get_updated_masks(
                frame=current_frame,
                previous_frame=self._previous_frame,
                previous_tracklets=snapshots,
                new_tracklets=new_tracklets,
                removed_tracklet_ids=sorted(removed),
                reference_tracklets=self._previous_reference_tracklets,
            )
            if mask_output is None:
                if self._mask_tracklet_ids:
                    mask_manager.reset()
                    self._clear_mask_state()
                return None
            self._validate_mask_output(mask_output, current_frame, expected_ids)
        except Exception as exc:
            # A failed inference may have already mutated memory. Recreate from
            # future matched boxes rather than continuing a partial session.
            mask_manager.reset()
            self._clear_mask_state()
            if not isinstance(exc, RuntimeError) or not _is_out_of_memory(exc):
                raise
            self._consecutive_mask_failures += 1
            logger.warning(
                "McByte++ mask pipeline ran out of memory on frame %d; falling back to IoU-only "
                "association (failure %d/%d): %s",
                self.frame_id,
                self._consecutive_mask_failures,
                _MAX_CONSECUTIVE_MASK_FAILURES,
                exc,
            )
            if self._consecutive_mask_failures >= _MAX_CONSECUTIVE_MASK_FAILURES:
                logger.error(
                    "McByte++ disabling mask-conditioned association after %d out-of-memory failures.",
                    self._consecutive_mask_failures,
                )
                self.mask_manager = None
            return None

        self._consecutive_mask_failures = 0
        self._mask_tracklet_ids = set(mask_output.tracklet_mask_dict)
        self._untrusted_mask_ids.intersection_update(self._mask_tracklet_ids)
        if self.enable_mask_trust_filter and mask_output.masks is not None:
            # Empty masks can reappear on another object later. Remember them even
            # without a detector match; retaining backend memory does not imply trust.
            self._untrusted_mask_ids.update(
                key for key, row in mask_output.tracklet_mask_dict.items() if not np.any(mask_output.masks[row])
            )
        self._previous_removed_tracklet_ids.clear()
        self._previous_new_tracklets = []
        self._previous_refresh_tracklets = []
        return mask_output

    def _refresh_previous_masks(self, mask_manager: MaskManager) -> None:
        """Re-prompt only eligible survivors on the image that supplied their boxes."""
        candidates = [
            t for t in self._previous_refresh_tracklets if t.tracker_id not in self._previous_removed_tracklet_ids
        ]
        if not candidates or self._previous_frame is None:
            return
        replacements = mask_manager.mask_generator.generate(self._previous_frame, candidates)
        self._validate_mask_output(replacements, self._previous_frame, {t.tracker_id for t in candidates})
        self._refresh_masks(self._previous_frame, replacements)
        self._mask_last_refresh.update({t.tracker_id: self._mask_image_count - 1 for t in candidates})
        if self.enable_mask_trust_filter and replacements.masks is not None:
            self._refreshed_mask_ids = {
                t.tracker_id
                for t in candidates
                if self._mask_passes_recovery_checks(
                    replacements.masks[replacements.tracklet_mask_dict[t.tracker_id]], t
                )
            }

    def _validate_mask_output(self, output: MaskOutput, frame: np.ndarray, expected_ids: set[int]) -> None:
        """Require the completed lifecycle ID set and integer channel mapping."""
        for key, row in output.tracklet_mask_dict.items():
            if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (key, row)):
                raise ValueError("Mask tracklet IDs and channel indices must be integers, not booleans.")
        ids = set(output.tracklet_mask_dict)
        if expected_ids - ids:
            raise RuntimeError("Mask propagation omitted a requested tracklet ID.")
        if ids - expected_ids:
            raise RuntimeError("Mask propagation returned an unexpected tracklet ID.")
        masks = output.masks
        if masks is None:
            if ids:
                raise ValueError("Mask output has tracklet IDs but no masks.")
            return
        if masks.ndim != 3 or masks.shape[1:] != frame.shape[:2]:
            raise ValueError("Mask output must have shape (N, H, W) on the current image grid.")
        if sorted(output.tracklet_mask_dict.values()) != list(range(len(masks))):
            raise ValueError("Mask output must map each channel to exactly one tracklet ID.")

    def _alive_tracklet_ids(self) -> set[int]:
        """Return non-negative tracker IDs held by active tracklets."""
        return {int(track.tracker_id) for track in self.tracks if track.tracker_id >= 0}

    def _detections_to_tracklet_snapshots(
        self,
        detections: sv.Detections,
    ) -> list[TrackletSnapshot]:
        """Convert detections carrying internal IDs into mask-manager snapshots.

        Only detections with valid non-negative tracker IDs are converted. The returned snapshots contain the tracker ID
        and ``xyxy`` box needed by mask generators.
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
        """Store same-image boxes with internal IDs for the next mask update.

        Eligible and pending tracklets are separate from actual mask ownership. Unconfirmed/unmatched boxes also
        participate in the occlusion veto, using unique negative reference IDs; they never receive masks.
        """
        self._merge_pending_removals(removed_tracklet_ids)
        if self.mask_manager is None:
            self._clear_mask_state()
            return
        if frame is None:
            self._invalidate_mask_observation()
            return

        current_tracklets = self._detections_to_tracklet_snapshots(detections)
        self._previous_new_tracklets = self._collect_new_masked_tracklets(current_tracklets)
        eligible_ids = self._mask_tracklet_ids | {t.tracker_id for t in self._previous_new_tracklets}
        # Keep this copy so later mask creation and refresh prompts use the image paired with these boxes.
        # The RGB conversion can still share caller memory for a channel-reversed BGR view;
        # reusing or drawing into that buffer must not alter this snapshot. Only one is retained here.
        self._previous_frame = frame.copy()
        self._previous_tracklets = [t for t in current_tracklets if t.tracker_id in eligible_ids]
        self._previous_reference_tracklets = (
            [
                TrackletSnapshot(int(key) if key >= 0 else -index - 1, box.copy())
                for index, (key, box) in enumerate(zip(detections.tracker_id, detections.xyxy))
            ]
            if detections.tracker_id is not None
            else []
        )
        self._update_mask_trust(current_tracklets)
        self._previous_refresh_tracklets = self._collect_mask_refresh_tracklets(current_tracklets)

    def _merge_pending_removals(self, ids: list[int]) -> None:
        """Queue terminal IDs until the backend consumes them on a real image."""
        self._previous_removed_tracklet_ids.update(ids)
        self._untrusted_mask_ids.difference_update(ids)
        self._refreshed_mask_ids.difference_update(ids)
        for key in ids:
            self._mask_last_refresh.pop(key, None)
            self._mask_pending_ages.pop(key, None)

    def _collect_new_masked_tracklets(
        self,
        current_tracklets: list[TrackletSnapshot],
    ) -> list[TrackletSnapshot]:
        """Select visible unowned tracklets after enough consecutive image matches.

        Keep eligibility while an occlusion delays creation. Ownership changes only when validated backend output
        includes the new ID.
        """
        self._mask_pending_ages = {
            t.tracker_id: self._mask_pending_ages.get(t.tracker_id, 0) + 1
            for t in current_tracklets
            if t.tracker_id not in self._mask_tracklet_ids
        }
        return [
            t
            for t in current_tracklets
            if self._mask_pending_ages.get(t.tracker_id, 0) >= self.minimum_mask_creation_frames
        ]

    def _collect_mask_refresh_tracklets(
        self,
        current_tracklets: list[TrackletSnapshot],
    ) -> list[TrackletSnapshot]:
        """Select suspect masks with visible matches, no occluder and an elapsed cooldown."""
        output = self._last_mask_output
        if not self.enable_mask_refresh or self.mask_manager is None or output is None or output.masks is None:
            return []
        candidates = []
        for tracklet in current_tracklets:
            key = tracklet.tracker_id
            row = output.tracklet_mask_dict.get(key)
            if (
                row is None
                or self._mask_image_count - self._mask_last_refresh.get(key, -self.mask_refresh_interval)
                < self.mask_refresh_interval
            ):
                continue
            if self.mask_manager._is_tracklet_occluded(tracklet, self._previous_reference_tracklets):
                continue
            needs_refresh = (
                key in self._untrusted_mask_ids
                if self.enable_mask_trust_filter
                else self._mask_needs_refresh(output.masks[row], tracklet)
            )
            if needs_refresh:
                candidates.append(tracklet)
        return candidates

    def _mask_needs_refresh(self, mask: np.ndarray, tracklet: TrackletSnapshot) -> bool:
        """Check emptiness or strict displacement against a matched detector box."""
        metrics = _get_mask_metrics(mask, tracklet.xyxy)
        if metrics is None:
            # Empty masks qualify even at threshold zero. A nonempty mask outside
            # a clipped, zero-area box has zero coverage.
            return not np.any(mask) or self.mask_refresh_coverage_threshold > 0
        return metrics[0] < self.mask_refresh_coverage_threshold

    def _mask_passes_recovery_checks(self, mask: np.ndarray, tracklet: TrackletSnapshot) -> bool:
        """Require visible mask evidence usable for association and no displacement."""
        metrics = _get_mask_metrics(mask, tracklet.xyxy)
        return (
            metrics is not None
            and metrics[0] >= max(self.minimum_mask_coverage, self.mask_refresh_coverage_threshold)
            and metrics[1] >= self.minimum_mask_fill_ratio
        )

    def _update_mask_trust(self, current_tracklets: list[TrackletSnapshot]) -> None:
        """Assess matched boxes after association, independently of refresh eligibility.

        An occlusion or cooldown can delay a prompt but cannot keep a suspect mask eligible for association. Spontaneous
        recovery never clears exclusion. A validated refresh becomes usable on the next image, after both its seed and
        current propagated output have passed checks against detector boxes.
        """
        output = self._last_mask_output
        if not self.enable_mask_trust_filter or self.mask_manager is None or output is None or output.masks is None:
            return
        for tracklet in current_tracklets:
            key = tracklet.tracker_id
            row = output.tracklet_mask_dict.get(key)
            if row is None:
                continue
            mask = output.masks[row]
            if self._mask_needs_refresh(mask, tracklet):
                self._untrusted_mask_ids.add(key)
            elif (
                key in self._untrusted_mask_ids
                and key in self._refreshed_mask_ids
                and not self.mask_manager._is_tracklet_occluded(tracklet, self._previous_reference_tracklets)
                and self._mask_passes_recovery_checks(mask, tracklet)
            ):
                self._untrusted_mask_ids.remove(key)
        self._refreshed_mask_ids.clear()

    def _get_association_mask_output(self) -> MaskOutput | None:
        """Hide excluded IDs from association without modifying backend masks or ownership."""
        output = self._last_mask_output
        if output is None or not self.enable_mask_trust_filter or not self._untrusted_mask_ids:
            return output
        mapping = {key: row for key, row in output.tracklet_mask_dict.items() if key not in self._untrusted_mask_ids}
        return MaskOutput(
            masks=output.masks,
            tracklet_mask_dict=mapping,
            mask_avg_prob_dict=(
                None
                if output.mask_avg_prob_dict is None
                else {key: score for key, score in output.mask_avg_prob_dict.items() if key in mapping}
            ),
        )

    def _invalidate_mask_observation(self) -> None:
        """Discard stale box prompts, retaining the last backend image and memory."""
        self._last_mask_output = None
        self._previous_tracklets = []
        self._previous_new_tracklets = []
        self._previous_reference_tracklets = []
        self._previous_refresh_tracklets = []
        self._mask_pending_ages = {}
        self._refreshed_mask_ids.clear()

    def _clear_mask_state(self) -> None:
        """Clear session bookkeeping without changing the inference failure streak."""
        self._previous_frame = None
        self._previous_tracklets = []
        self._previous_new_tracklets = []
        self._previous_removed_tracklet_ids = set()
        self._previous_reference_tracklets = []
        self._previous_refresh_tracklets = []
        self._last_mask_output = None
        self._mask_tracklet_ids = set()
        self._untrusted_mask_ids.clear()
        self._refreshed_mask_ids.clear()
        self._mask_pending_ages = {}
        self._mask_last_refresh = {}
        self._mask_image_count = 0

    def _get_iou_matrix(
        self,
        tracklets: list[McBytePlusPlusTracklet],
        detections: np.ndarray,
        tracklet_boxes_by_id: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Compute IoU similarity between tracklet states and detection boxes.

        Returns an ``(N, M)`` matrix where ``N`` is the number of tracklets and ``M`` is
        the number of detections. Empty inputs are handled by returning an empty matrix
        with the expected shape.

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

    def _get_mask_conditioned_associated_indices(
        self,
        similarity_matrix: np.ndarray,
        raw_iou_similarity: np.ndarray,
        tracklets: list[McBytePlusPlusTracklet],
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
            mask_output=self._get_association_mask_output(),
            minimum_similarity=min_similarity_thresh,
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

            # Every unmatched high-confidence detection is returned, with
            # tracker_id -1 when it does not spawn a track. This matches the
            # documented update() contract ("unmatched/unconfirmed detections
            # have tracker_id of -1"), the symmetric handling of unmatched
            # low-confidence detections, and sibling ByteTrack. Only detections
            # clearing the activation threshold spawn a new tracklet; a
            # first-frame instant-activated spawn is emitted with its real ID.
            tracker_id_out = -1
            if conf >= self.track_activation_threshold:
                tracklet = McBytePlusPlusTracklet(
                    initial_bbox=detection_boxes[global_idx],
                    state_estimator_class=self.state_estimator_class,
                )
                if is_first_frame and self.instant_first_frame_activation:
                    tracklet.tracker_id = self._allocate_tracker_id()
                    tracklet.public_id = tracklet.tracker_id
                self.tracks.append(tracklet)
                tracker_id_out = tracklet.tracker_id

            out_det_indices.append(global_idx)
            out_tracker_ids.append(tracker_id_out)

    def reset(self) -> None:
        """Clear sequence state and restore masks disabled by repeated OOM failures."""
        self.tracks = []
        self.frame_id = 0
        self._last_timestamp = None
        self._reset_id_allocator()
        if self.mask_manager is None and self._consecutive_mask_failures >= _MAX_CONSECUTIVE_MASK_FAILURES:
            self.mask_manager = self._mask_manager_original
        if self.mask_manager is not None:
            self.mask_manager.reset()
        if self.cmc is not None:
            self.cmc.reset()
        self._clear_mask_state()
        self._consecutive_mask_failures = 0

    def apply_cmc_batch(self, H: np.ndarray | None) -> None:
        """Apply camera motion compensation to all active tracks.

        Convenience wrapper around :meth:`CMC.apply_batch` for callers that hold
        a tracker instance and an affine transform. ``update()`` applies CMC
        directly and does not rely on this method.

        Args:
            H: 2x3 affine transform matrix returned by CMC.estimate().
                If None, this method is a no-op.

        Examples:
            >>> tracker = McBytePlusPlusTracker()
            >>> tracker.apply_cmc_batch(None)  # no-op
        """
        CMC.apply_batch(H, self.tracks)
