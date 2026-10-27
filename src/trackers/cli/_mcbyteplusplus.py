# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""McByte++ configuration and overlays shared by inspection and benchmarking.

These helpers configure CLI runs only; the independent tracker owns its defaults and tracking behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import supervision as sv

from trackers.cli._annotate import annotate_masks, annotate_tracklet_boxes, draw_status_lines
from trackers.core.mcbyteplusplus.tracker import McBytePlusPlusTracker
from trackers.utils.cmc import CMCMethod


@dataclass
class CMCOptions:
    """Camera-motion compensation options.

    Attributes:
        enable: Enable camera-motion compensation.
        method: Motion estimator used by CMC.
        downscale: Image downscale factor used by CMC.
    """

    enable: bool = True
    method: CMCMethod = "orb"
    downscale: int = 4


@dataclass
class TrackerOptions:
    """Association, confirmation, mask creation and refresh settings.

    Attributes:
        lost_track_buffer: Missed-frame budget in 30 FPS units.
        track_activation_threshold: Minimum confidence for a new tracklet.
        high_conf_det_threshold: High-confidence detection cutoff.
        minimum_consecutive_frames: Successful observations for confirmation.
        instant_first_frame_activation: Confirm tracks on the first update.
        minimum_iou_threshold_first_assoc: High-confidence match threshold.
        minimum_iou_threshold_second_assoc: Low-confidence match threshold.
        minimum_iou_threshold_unconfirmed_assoc: Tentative match threshold.
        minimum_mask_coverage: Mask coverage required for association.
        minimum_mask_fill_ratio: Mask foreground fraction for association.
        enable_isolated_mask_matching: Rescue isolated pairs with mask evidence.
        minimum_mask_creation_frames: Real-image matches before mask creation.
        enable_mask_refresh: Re-prompt empty or displaced visible masks.
        mask_refresh_coverage_threshold: Coverage below which masks refresh.
        mask_refresh_interval: Minimum images between per-tracklet refreshes.
        enable_mask_trust_filter: Exclude suspect masks until a validated refresh.
    """

    lost_track_buffer: int = 30
    track_activation_threshold: float = 0.7
    high_conf_det_threshold: float = 0.6
    minimum_consecutive_frames: int = 2
    instant_first_frame_activation: bool = True
    minimum_iou_threshold_first_assoc: float = 0.1
    minimum_iou_threshold_second_assoc: float = 0.5
    minimum_iou_threshold_unconfirmed_assoc: float = 0.3
    minimum_mask_coverage: float = 0.9
    minimum_mask_fill_ratio: float = 0.05
    enable_isolated_mask_matching: bool = False
    minimum_mask_creation_frames: int = 1
    enable_mask_refresh: bool = True
    mask_refresh_coverage_threshold: float = 0.5
    mask_refresh_interval: int = 1
    enable_mask_trust_filter: bool = True


class _HubAuthNoticeFilter(logging.Filter):
    """Save the optional Hub authentication notice without hiding download failures."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if record.levelno == logging.WARNING and "unauthenticated requests to the HF Hub" in message:
            self.messages.append(message)
            return False
        return True


def visualize_mcbyteplusplus(
    frame: np.ndarray, number: int, tracked: sv.Detections, tracker: McBytePlusPlusTracker
) -> np.ndarray:
    """Overlay raw masks and report exclusions after the current detector matches.

    Args:
        frame: RGB uint8 image, copied before drawing.
        number: Original sequence frame number shown in the status overlay.
        tracked: Current detector matches with public tracker IDs.
        tracker: Tracker supplying raw masks and excluded internal mask IDs.

    Returns:
        Annotated RGB image. Excluded masks remain visible for inspection;
        their IDs are listed in the exclusion status line.
    """
    visual = frame.copy()
    output = tracker.last_mask_output
    mask_ids = (
        []
        if output is None
        else [key for key, _ in sorted(output.tracklet_mask_dict.items(), key=lambda item: item[1])]
    )
    if output is not None and output.masks is not None and len(output.masks):
        visual = annotate_masks(visual, output.masks, mask_ids)
    ids = [] if tracked.tracker_id is None else [int(value) for value in tracked.tracker_id]
    if ids:
        visual = annotate_tracklet_boxes(visual, tracked.xyxy, ids)
    return draw_status_lines(
        visual,
        [
            "Tracker: McByte++",
            f"Frame: {number}",
            f"Output IDs: {ids}",
            f"Masks: {mask_ids}",
            f"Excluded mask IDs: {sorted(tracker.untrusted_mask_ids)}",
        ],
    )
