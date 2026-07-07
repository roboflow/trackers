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
    """Coordinate McByte mask generation and temporal propagation.

    The manager follows the original McByte timing: masks for frame ``t`` are
    prepared before association on frame ``t``, using tracker outputs from frame
    ``t-1``. New and removed tracklets are therefore passed in as lifecycle
    events from the previous tracker update.
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
        """Reset mask-manager state and the underlying propagator."""
        self._initialized = False
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

        The method consumes tracker state from the previous frame. On the first
        valid call, masks are generated from ``previous_tracklets`` on
        ``previous_frame`` and used to initialize the propagator. On later calls,
        masks for ``new_tracklets`` are generated on ``previous_frame`` and added to
        the propagator, while ``removed_tracklet_ids`` are removed from propagation
        memory.

        After lifecycle updates are applied, the propagator advances masks to
        ``frame`` and returns the resulting ``MaskOutput``.

        Args:
            frame: Current RGB frame for which masks should be produced.
            previous_frame: Previous RGB frame used for initialization or adding new
                masks. If ``None``, no masks are returned.
            previous_tracklets: Tracklet snapshots produced by the tracker on the
                previous frame.
            new_tracklets: Tracklets created on the previous frame that should be
                added to propagation memory.
            removed_tracklet_ids: Tracklet IDs terminated on the previous frame that
                should be removed from propagation memory.

        Returns:
            Propagated mask output for ``frame``. Returns ``None`` when there is no
            previous frame/tracklet state, no propagator is configured, or
            propagation fails.
        """
        if previous_frame is None:
            return None

        if not self._initialized and len(previous_tracklets) == 0:
            return None

        if self.mask_propagator is None:
            return None

        new_tracklets = [] if new_tracklets is None else new_tracklets
        removed_tracklet_ids = [] if removed_tracklet_ids is None else removed_tracklet_ids

        if not self._initialized:
            mask_output = self.mask_generator.generate(previous_frame, previous_tracklets)
            self.mask_propagator.initialize(previous_frame, mask_output)
            self._initialized = True
        else:
            if len(new_tracklets) > 0:
                new_mask_output = self.mask_generator.generate(previous_frame, new_tracklets)
                self.mask_propagator.add_masks(previous_frame, new_mask_output)

            if len(removed_tracklet_ids) > 0:
                self.mask_propagator.remove_masks(removed_tracklet_ids)

        propagated_output = self.mask_propagator.propagate(frame)
        if propagated_output is not None:
            return propagated_output

        self._initialized = False
        return None
