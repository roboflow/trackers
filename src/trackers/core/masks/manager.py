# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.core.masks.base import (
    MaskGenerator,
    MaskOutput,
    MaskPropagator,
    TrackletSnapshot,
)

MASK_CREATION_BBOX_OVERLAP_THRESHOLD = 0.6


def _area(xyxy: np.ndarray) -> float:
    width = max(float(xyxy[2] - xyxy[0]), 0.0)
    height = max(float(xyxy[3] - xyxy[1]), 0.0)
    return width * height


def _intersection_area(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def _validate_mask_resolution(
    mask_output: MaskOutput,
    frame: np.ndarray,
) -> None:
    """Ensure propagated masks share the frame's pixel grid.

    McByte mask-conditioning (:func:`condition_similarity_with_masks`) measures
    mask coverage and fill ratio in ``mask.shape`` pixel space while detection
    boxes are expressed in original-frame pixels. The two only agree when the
    propagated masks have the same height and width as ``frame``. A propagator
    that internally resizes or pads without upsampling back would silently gate
    associations in the wrong coordinate space, so the invariant is enforced
    explicitly.

    Args:
        mask_output: Propagated mask output returned by the mask propagator.
        frame: Current frame whose ``(H, W)`` the masks must match.

    Raises:
        ValueError: If the masks' trailing two dimensions differ from the
            frame's height and width.
    """
    if mask_output.masks is None:
        return

    frame_hw = frame.shape[:2]
    mask_hw = mask_output.masks.shape[-2:]
    if mask_hw != frame_hw:
        raise ValueError(
            "Propagated masks must match the frame resolution. "
            f"Got mask (H, W)={tuple(mask_hw)} for frame (H, W)={tuple(frame_hw)}."
        )


def _get_tracklets_with_lower_bottom(
    tracklet: TrackletSnapshot,
    tracklets: list[TrackletSnapshot],
) -> list[TrackletSnapshot]:
    """Return tracklets whose bottom edge is lower than the target tracklet."""
    target_bottom = float(tracklet.xyxy[3])
    return [
        other for other in tracklets if other.tracker_id != tracklet.tracker_id and float(other.xyxy[3]) > target_bottom
    ]


def _get_overlap_with_lower_bottom_tracklets(
    tracklet: TrackletSnapshot,
    lower_bottom_tracklets: list[TrackletSnapshot],
) -> float:
    """Compute maximum box-overlap ratio with lower-bottom tracklets."""
    tracklet_area = _area(tracklet.xyxy)
    if tracklet_area == 0:
        return 0.0

    max_overlap = 0.0
    for other in lower_bottom_tracklets:
        overlap = _intersection_area(tracklet.xyxy, other.xyxy) / tracklet_area
        max_overlap = max(max_overlap, overlap)
        if max_overlap >= 1.0:
            break

    return max_overlap


class MaskManager:
    """Coordinate McByte mask generation and temporal propagation.

    The manager follows the original McByte timing: masks for frame ``t`` are
    prepared before association on frame ``t``, using tracker outputs from frame
    ``t-1``. New and removed tracklets are therefore passed in as lifecycle
    events from the previous tracker update.

    Mask creation may be delayed for tracklets whose boxes are strongly
    overlapped by tracklets with lower bottom edges. Pending tracklets are kept
    in one shared pool and retried on later frames, both before first Cutie
    initialization and during later mask addition.
    """

    def __init__(
        self,
        mask_generator: MaskGenerator,
        mask_propagator: MaskPropagator | None = None,
        mask_creation_bbox_overlap_threshold: float = MASK_CREATION_BBOX_OVERLAP_THRESHOLD,
    ) -> None:
        self.mask_generator = mask_generator
        self.mask_propagator = mask_propagator
        if not 0.0 <= mask_creation_bbox_overlap_threshold <= 1.0:
            raise ValueError("mask_creation_bbox_overlap_threshold must be between 0 and 1.")
        self.mask_creation_bbox_overlap_threshold = mask_creation_bbox_overlap_threshold
        self._initialized = False
        self._pending_tracklet_ids: set[int] = set()

    @property
    def pending_tracklet_ids(self) -> frozenset[int]:
        """Tracklet IDs whose mask creation is currently deferred.

        A tracklet lands here when its box is strongly overlapped by a tracklet
        with a lower bottom edge, and leaves once it is clean enough to mask or
        the tracker removes it. Exposed read-only so callers can report the
        deferral without reaching into the pool the manager mutates.

        Returns:
            A snapshot of the pending IDs; mutating it does not affect the manager.

        Examples:
            >>> from trackers.core.masks.dummy import DummyBoxMaskGenerator
            >>> MaskManager(DummyBoxMaskGenerator()).pending_tracklet_ids
            frozenset()
        """
        return frozenset(self._pending_tracklet_ids)

    def reset(self) -> None:
        """Reset mask-manager state and the underlying propagator."""
        self._initialized = False
        self._pending_tracklet_ids = set()
        if self.mask_propagator is not None:
            self.mask_propagator.reset()

    def get_updated_masks(
        self,
        frame: np.ndarray,
        previous_frame: np.ndarray | None,
        previous_tracklets: list[TrackletSnapshot],
        new_tracklets: list[TrackletSnapshot] | None = None,
        removed_tracklet_ids: list[int] | None = None,
    ) -> MaskOutput | None:
        """Return propagated masks for the current frame.

        The method consumes tracker state from the previous frame. All tracker inputs
        (``previous_frame``, ``previous_tracklets``, ``new_tracklets`` and
        ``removed_tracklet_ids``) are expected to correspond to the same previous
        tracker update and therefore describe a consistent tracker state.

        On the first valid call, masks are generated from the subset of
        ``previous_tracklets`` whose bounding boxes are considered suitable for mask
        creation. Tracklets whose boxes are too strongly overlapped are placed into a
        shared pending pool and retried on later frames until they become suitable for
        initialization.

        After the propagator has been initialized, newly created tracklets are handled
        in the same way: masks are generated only for suitable tracklets, while
        occluded tracklets remain in the pending pool and are retried on subsequent
        frames. Pending tracklets always use their most recent tracker state.
        Tracklets terminated by the tracker are removed both from the pending pool and
        from the propagator.

        After lifecycle updates are applied, the propagator advances masks to
        ``frame`` and returns the resulting ``MaskOutput``.

        Args:
            frame: Current RGB frame for which masks should be produced.
            previous_frame: Previous RGB frame used for initialization or adding new
                masks. If ``None``, no masks are returned.
            previous_tracklets: All tracklet snapshots produced by the tracker on the
                previous frame. These represent the complete visible tracker state
                associated with ``previous_frame``.
            new_tracklets: Subset of ``previous_tracklets`` corresponding to tracklets
                created on the previous frame. These tracklets are considered for
                adding new masks after the propagator has been initialized.
            removed_tracklet_ids: Tracklet IDs terminated on the previous frame that
                should be removed from both the pending pool and propagation memory.

        Returns:
            Propagated mask output for ``frame``. Returns ``None`` when there is no
            previous frame/tracklet state, no propagator is configured, or
            propagation fails.

        Examples:
            >>> import numpy as np
            >>> from trackers.core.masks import TrackletSnapshot
            >>> from trackers.core.masks.dummy import (
            ...     DummyBoxMaskGenerator,
            ...     DummyIdentityMaskPropagator,
            ... )
            >>> manager = MaskManager(
            ...     mask_generator=DummyBoxMaskGenerator(),
            ...     mask_propagator=DummyIdentityMaskPropagator(),
            ... )
            >>> frame = np.zeros((100, 120, 3), dtype=np.uint8)
            >>> tracklet = TrackletSnapshot(
            ...     tracker_id=1, xyxy=np.array([10, 20, 30, 40], dtype=np.float32)
            ... )
            >>> output = manager.get_updated_masks(
            ...     frame=frame,
            ...     previous_frame=frame,
            ...     previous_tracklets=[tracklet],
            ... )
            >>> output.tracklet_mask_dict
            {1: 0}
            >>> output.masks.shape
            (1, 100, 120)
        """
        if previous_frame is None:
            return None

        new_tracklets = [] if new_tracklets is None else new_tracklets
        removed_tracklet_ids = [] if removed_tracklet_ids is None else removed_tracklet_ids

        self._remove_pending_tracklets(removed_tracklet_ids)

        if not self._initialized and len(previous_tracklets) == 0:
            return None

        if self.mask_propagator is None:
            return None

        if not self._initialized:
            candidates = previous_tracklets
            clean_tracklets = self._select_clean_tracklets(candidates, previous_tracklets)

            if len(clean_tracklets) == 0:
                return None

            mask_output = self.mask_generator.generate(previous_frame, clean_tracklets)
            self.mask_propagator.initialize(previous_frame, mask_output)
            self._initialized = True
        else:
            candidates = self._build_addition_candidates(
                previous_tracklets=previous_tracklets,
                new_tracklets=new_tracklets,
            )
            clean_tracklets = self._select_clean_tracklets(candidates, previous_tracklets)

            if len(clean_tracklets) > 0:
                new_mask_output = self.mask_generator.generate(previous_frame, clean_tracklets)
                self.mask_propagator.add_masks(previous_frame, new_mask_output)

            if len(removed_tracklet_ids) > 0:
                self.mask_propagator.remove_masks(removed_tracklet_ids)

        propagated_output = self.mask_propagator.propagate(frame)
        if propagated_output is not None:
            _validate_mask_resolution(propagated_output, frame)
            return propagated_output

        self._initialized = False
        return None

    def _build_addition_candidates(
        self,
        previous_tracklets: list[TrackletSnapshot],
        new_tracklets: list[TrackletSnapshot],
    ) -> list[TrackletSnapshot]:
        """Build candidate tracklets for adding masks after initialization."""
        candidates = {tracklet.tracker_id: tracklet for tracklet in new_tracklets}
        candidates.update(self._get_visible_pending_tracklets(previous_tracklets))
        return list(candidates.values())

    def _get_visible_pending_tracklets(
        self,
        previous_tracklets: list[TrackletSnapshot],
    ) -> dict[int, TrackletSnapshot]:
        """Return pending tracklets currently visible in tracker outputs."""
        visible_tracklets = {tracklet.tracker_id: tracklet for tracklet in previous_tracklets}
        return {
            tracklet_id: visible_tracklets[tracklet_id]
            for tracklet_id in self._pending_tracklet_ids
            if tracklet_id in visible_tracklets
        }

    def _select_clean_tracklets(
        self,
        candidates: list[TrackletSnapshot],
        reference_tracklets: list[TrackletSnapshot],
    ) -> list[TrackletSnapshot]:
        """Split same-frame candidates into clean and still-pending tracklets."""
        clean_tracklets = []

        for tracklet in candidates:
            if self._is_tracklet_occluded(tracklet, reference_tracklets):
                self._pending_tracklet_ids.add(tracklet.tracker_id)
            else:
                clean_tracklets.append(tracklet)
                self._pending_tracklet_ids.discard(tracklet.tracker_id)

        return clean_tracklets

    def _is_tracklet_occluded(
        self,
        tracklet: TrackletSnapshot,
        reference_tracklets: list[TrackletSnapshot],
    ) -> bool:
        """Return whether a tracklet box is too overlapped for mask creation."""
        lower_bottom_tracklets = _get_tracklets_with_lower_bottom(tracklet, reference_tracklets)
        overlap = _get_overlap_with_lower_bottom_tracklets(tracklet, lower_bottom_tracklets)
        return overlap >= self.mask_creation_bbox_overlap_threshold

    def _remove_pending_tracklets(
        self,
        removed_tracklet_ids: list[int],
    ) -> None:
        """Remove terminated tracklets from the pending mask-creation pool."""
        self._pending_tracklet_ids -= set(removed_tracklet_ids)
