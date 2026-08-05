# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def _resolve_auto_device() -> str:
    """Resolve the ``"auto"`` device sentinel for the mask pipeline.

    Returns ``"cuda"`` when available, otherwise ``"cpu"``. Apple MPS is
    deliberately skipped: the SAM + Cutie inference graph issues hundreds of
    small kernels per frame, and the per-op dispatch/synchronization overhead
    on MPS makes the pipeline roughly an order of magnitude slower than the
    CPU backend on Apple Silicon (measured ~12.7x on torch 2.13, identical
    outputs). Pass ``device="mps"`` explicitly to opt in regardless.

    Returns:
        Device string suitable for ``torch.device``.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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

    @abstractmethod
    def add_masks(
        self,
        frame: np.ndarray,
        mask_output: MaskOutput,
    ) -> None:
        """Add masks to existing propagation state."""

    @abstractmethod
    def remove_masks(
        self,
        tracklet_ids: list[int],
    ) -> None:
        """Remove masks from existing propagation state."""
