# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.core.mcbyte.masks.base import (
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
    """Return the last initialized mask output unchanged."""

    def __init__(self) -> None:
        self._mask_output: MaskOutput | None = None

    def reset(self) -> None:
        self._mask_output = None

    def initialize(
        self,
        frame: np.ndarray,
        mask_output: MaskOutput,
    ) -> None:
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
        if self._mask_output is None:
            return None

        return MaskOutput(
            masks=None if self._mask_output.masks is None else self._mask_output.masks.copy(),
            tracklet_mask_dict=self._mask_output.tracklet_mask_dict.copy(),
            mask_avg_prob_dict=(
                None if self._mask_output.mask_avg_prob_dict is None else self._mask_output.mask_avg_prob_dict.copy()
            ),
        )
