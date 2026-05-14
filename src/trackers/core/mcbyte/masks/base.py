# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackletSnapshot:
    """Minimal tracker state needed by mask components."""

    tracker_id: int
    xyxy: np.ndarray


@dataclass(frozen=True)
class MaskOutput:
    """Mask information produced before McByte association."""

    masks: np.ndarray | None
    tracklet_mask_dict: dict[int, int]
    mask_avg_prob_dict: dict[int, float] | None = None


class MaskGenerator(ABC):
    """Generate masks from tracklet boxes."""

    @abstractmethod
    def generate(
        self,
        frame: np.ndarray,
        tracklets: list[TrackletSnapshot],
    ) -> MaskOutput:
        """Generate masks for the given tracklet snapshots."""


class MaskPropagator(ABC):
    """Propagate masks from one frame to the next."""

    @abstractmethod
    def reset(self) -> None:
        """Reset propagation state."""

    @abstractmethod
    def initialize(
        self,
        frame: np.ndarray,
        mask_output: MaskOutput,
    ) -> None:
        """Initialize propagation state."""

    @abstractmethod
    def propagate(
        self,
        frame: np.ndarray,
    ) -> MaskOutput | None:
        """Propagate masks to the current frame."""