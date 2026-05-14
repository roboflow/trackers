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


class MaskManager:
    """Manage McByte mask generation and propagation.

    The manager follows the original McByte timing: masks for the current frame
    are prepared before association, but they are initialized/updated from tracker 
    outputs produced on the previous frame.
    """

    def __init__(
        self,
        mask_generator: MaskGenerator,
        mask_propagator: MaskPropagator | None = None,
    ) -> None:
        self.mask_generator = mask_generator
        self.mask_propagator = mask_propagator
        self._initialized = False

    def reset(self) -> None:
        self._initialized = False
        if self.mask_propagator is not None:
            self.mask_propagator.reset()

    def get_updated_masks(
        self,
        frame: np.ndarray,
        previous_frame: np.ndarray | None,
        previous_tracklets: list[TrackletSnapshot],
    ) -> MaskOutput | None:
        """Return masks for the current frame.

        No masks are returned until at least one previous frame and previous
        tracker output are available.
        """
        if previous_frame is None or len(previous_tracklets) == 0:
            return None

        if not self._initialized or self.mask_propagator is None:
            mask_output = self.mask_generator.generate(previous_frame, previous_tracklets)
            self._initialized = True

            if self.mask_propagator is not None:
                self.mask_propagator.initialize(previous_frame, mask_output)
                propagated_output = self.mask_propagator.propagate(frame)
                return propagated_output if propagated_output is not None else mask_output

            return mask_output

        propagated_output = self.mask_propagator.propagate(frame)
        if propagated_output is not None:
            return propagated_output

        mask_output = self.mask_generator.generate(previous_frame, previous_tracklets)
        self.mask_propagator.initialize(previous_frame, mask_output)
        return mask_output