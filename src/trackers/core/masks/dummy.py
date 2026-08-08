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


class DummyBoxMaskGenerator(MaskGenerator):
    """Generate rectangular binary masks from tracklet bounding boxes."""

    def generate(
        self,
        frame: np.ndarray,
        tracklets: list[TrackletSnapshot],
    ) -> MaskOutput:
        """Generate one axis-aligned rectangular mask per tracklet bounding box.

        Args:
            frame: Current video frame, used only for its ``(H, W)`` shape.
            tracklets: Tracklets to generate masks for.

        Returns:
            ``MaskOutput`` with one boolean mask per tracklet, mapping each
            tracklet ID to its mask index. ``mask_avg_prob_dict`` is always
            ``None`` since this generator does not estimate confidence.
        """
        height, width = frame.shape[:2]
        masks = np.zeros((len(tracklets), height, width), dtype=bool)
        tracklet_mask_dict: dict[int, int] = {}

        for mask_index, tracklet in enumerate(tracklets):
            x1, y1, x2, y2 = tracklet.xyxy.astype(int)

            x1 = int(np.clip(x1, 0, width))
            x2 = int(np.clip(x2, 0, width))
            y1 = int(np.clip(y1, 0, height))
            y2 = int(np.clip(y2, 0, height))

            masks[mask_index, y1:y2, x1:x2] = True
            tracklet_mask_dict[tracklet.tracker_id] = mask_index

        return MaskOutput(
            masks=masks,
            tracklet_mask_dict=tracklet_mask_dict,
            mask_avg_prob_dict=None,
        )


class DummyIdentityMaskPropagator(MaskPropagator):
    """In-memory mask propagator used for lightweight MaskManager tests.

    The propagator keeps the last initialized ``MaskOutput`` and returns copies of it during propagation. It also
    supports simple add/remove lifecycle operations so MaskManager tests can validate mask indexing without requiring
    SAM, Cutie, checkpoints, or GPU.
    """

    def __init__(self) -> None:
        self._mask_output: MaskOutput | None = None

    def reset(self) -> None:
        """Clear the stored mask output."""
        self._mask_output = None

    def initialize(
        self,
        frame: np.ndarray,
        mask_output: MaskOutput,
    ) -> None:
        """Store a copy of the initial mask output."""
        self._mask_output = MaskOutput(
            masks=None if mask_output.masks is None else mask_output.masks.copy(),
            tracklet_mask_dict=mask_output.tracklet_mask_dict.copy(),
            mask_avg_prob_dict=(
                None if mask_output.mask_avg_prob_dict is None else mask_output.mask_avg_prob_dict.copy()
            ),
        )

    def propagate(
        self,
        frame: np.ndarray,
    ) -> MaskOutput | None:
        """Return a copy of the current stored mask output."""
        if self._mask_output is None:
            return None

        return MaskOutput(
            masks=None if self._mask_output.masks is None else self._mask_output.masks.copy(),
            tracklet_mask_dict=self._mask_output.tracklet_mask_dict.copy(),
            mask_avg_prob_dict=(
                None if self._mask_output.mask_avg_prob_dict is None else self._mask_output.mask_avg_prob_dict.copy()
            ),
        )

    def add_masks(
        self,
        frame: np.ndarray,
        mask_output: MaskOutput,
    ) -> None:
        """Append new masks and remap their tracklet indices."""
        if self._mask_output is None:
            self.initialize(frame, mask_output)
            return

        if self._mask_output.masks is None or mask_output.masks is None:
            return

        offset = self._mask_output.masks.shape[0]
        added_mapping = {
            tracklet_id: offset + local_index for tracklet_id, local_index in mask_output.tracklet_mask_dict.items()
        }

        self._mask_output = MaskOutput(
            masks=np.concatenate([self._mask_output.masks, mask_output.masks], axis=0),
            tracklet_mask_dict={
                **self._mask_output.tracklet_mask_dict,
                **added_mapping,
            },
            mask_avg_prob_dict=None,
        )

    def remove_masks(
        self,
        tracklet_ids: list[int],
    ) -> None:
        """Remove masks for the given tracklet IDs and compact local indices."""
        if self._mask_output is None or self._mask_output.masks is None:
            return

        tracklet_id_set = set(tracklet_ids)
        remaining_items = [
            (tracklet_id, mask_index)
            for tracklet_id, mask_index in self._mask_output.tracklet_mask_dict.items()
            if tracklet_id not in tracklet_id_set
        ]

        kept_indices = [mask_index for _, mask_index in remaining_items]
        new_mapping = {tracklet_id: new_index for new_index, (tracklet_id, _) in enumerate(remaining_items)}

        self._mask_output = MaskOutput(
            masks=self._mask_output.masks[kept_indices],
            tracklet_mask_dict=new_mapping,
            mask_avg_prob_dict=None,
        )
