# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""McByte++ configuration and overlays shared by inspection and benchmarking.

These helpers configure CLI runs only; the independent tracker owns its defaults and tracking behavior.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from jsonargparse import ArgumentParser

from trackers.cli._annotate import annotate_masks, annotate_tracklet_boxes, draw_status_lines
from trackers.cli.track import ReIDModelConfig, _load_reid_model, _reid_requested
from trackers.core.mcbyteplusplus.tracker import McBytePlusPlusTracker
from trackers.reid.encoder import ReIDEncoder
from trackers.reid.long_term_config import LongTermReIDConfig
from trackers.reid.long_term_encoder import SPORTS_REID_MODEL
from trackers.utils.cmc import CMCMethod


@dataclass
class CMCOptions:
    """Camera-motion compensation options.

    Attributes:
        enable: Enable camera-motion compensation.
        method: Motion estimator used by CMC.
        downscale: Image downscale factor used by CMC.
        conditional: Enable conditional estimation and validated state
            corrections for all methods; False selects the original shared CMC.
        minimum_matches: Conditional ORB/SIFT candidate-match minimum before
            statistical filtering. At least five must remain afterwards.
        minimum_scale: Minimum accepted conditional affine column norm.
        maximum_scale: Maximum accepted conditional affine column norm.
    """

    enable: bool = True
    method: CMCMethod = "orb"
    downscale: int = 4
    conditional: bool = True
    minimum_matches: int = 12
    minimum_scale: float = 0.3
    maximum_scale: float = 3.0


@dataclass
class TrackerOptions:
    """Association, confirmation, mask lifecycle and optional re-ID joining settings.

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
        reid_joining_config: Optional long-term joining policy; requires reid.model.
        enable_reid_camera_motion_pause: Skip new re-ID feature extraction for
            all subjects and pause pending identity decision ages during strong
            estimated camera motion. Defaults to True; requires conditional CMC
            when CMC and re-ID are enabled. Inactive without a re-ID model or CMC.
            Keep reid_joining_config.buffer_results=True for complete text output;
            without buffering, longer pending periods can discard valid track
            fragments or entire tracklets, so that combination is not recommended
            for accuracy evaluation.
        reid_camera_motion_threshold: Maximum allowed normalized camera
            displacement before pausing; defaults to 0.02 of the image diagonal.
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
    reid_joining_config: LongTermReIDConfig | None = None
    enable_reid_camera_motion_pause: bool = True
    reid_camera_motion_threshold: float = 0.02


def _add_reid_arguments(parser: ArgumentParser) -> list[str]:
    """Register the same encoder group as generic track, including bare flags."""
    added = parser.add_class_arguments(ReIDModelConfig, "reid", skip={"enable"})
    parser.add_argument(
        "--reid.enable", type=bool, default=None, help="Enable long-term joining; requires --reid.model."
    )
    return [*added, "reid.enable"]


def _validate_reid_options(reid: ReIDModelConfig, joining: LongTermReIDConfig | None) -> bool:
    """Reject incomplete McByte++ appearance configuration before model loading."""
    enabled = _reid_requested(reid)
    if enabled and not reid.model:
        raise ValueError("McByte++ ReID requires --reid.model, for example mcbyteplusplus_sports.")
    if joining is not None and not enabled:
        raise ValueError("--tracker.reid_joining_config requires --reid.model.")
    return enabled


def _load_mcbyteplusplus_reid(reid: ReIDModelConfig) -> ReIDEncoder | None:
    """Load one explicitly selected encoder for an inspection or whole benchmark."""
    return _load_reid_model(reid, long_term=True) if _reid_requested(reid) else None


def _tracker_kwargs(options: dict[str, Any]) -> dict[str, Any]:
    """Restore the typed joining configuration after serializing run options."""
    result = dict(options)
    joining = result.get("reid_joining_config")
    if isinstance(joining, dict):
        result["reid_joining_config"] = LongTermReIDConfig(**joining)
    return result


def _reid_provenance(options: ReIDModelConfig, encoder: ReIDEncoder | None) -> dict[str, Any]:
    """Record the selected source, loaded encoder and available preprocessing."""
    try:
        package_version = version("reid")
    except PackageNotFoundError:
        package_version = None
    result: dict[str, Any] = {
        "enabled": encoder is not None,
        "source": options.model,
        "architecture": options.architecture,
        "device": str(getattr(encoder, "device", options.device)) if encoder is not None else None,
        "encoder": f"{type(encoder).__module__}.{type(encoder).__qualname__}" if encoder is not None else None,
        "reid_version": package_version,
    }
    if encoder is None:
        return result
    preprocessing = getattr(encoder, "preprocessing", None)
    if preprocessing is not None and callable(getattr(preprocessing, "to_dict", None)):
        result["preprocessing"] = preprocessing.to_dict()
    if options.model == SPORTS_REID_MODEL:
        from trackers.reid.sports_model import _model_provenance

        result.update(_model_provenance())
    elif options.model is not None and "://" not in options.model:
        source = Path(options.model).expanduser()
        artifacts = [source] if source.is_file() else [source / "reid_config.json", source / "weights.safetensors"]
        hashes = {}
        for artifact in artifacts:
            if artifact.is_file():
                digest = hashlib.sha256()
                with artifact.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[str(artifact.resolve())] = digest.hexdigest()
        if hashes:
            result["artifact_sha256"] = hashes
    return result


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

    Masks and box colours use internal tracklet IDs. Labels show public
    identities; unresolved eligible observations display ``-1 pending (internal_id)``.
    This overlay shows current output and does not redraw earlier frames after
    an identity resolves.

    Args:
        frame: RGB uint8 image, copied before drawing.
        number: Original sequence frame number shown in the status overlay.
        tracked: Current tracker output, including unassigned detections and
            observations whose public identities are still pending.
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
    internal_ids = [int(value) for value in tracked.data.get("tracklet_id", ids)]
    pending = [bool(value) for value in tracked.data.get("reid_pending", [False] * len(ids))]
    if ids:
        labels = [
            f"-1 pending ({internal})" if waiting else str(public)
            for public, internal, waiting in zip(ids, internal_ids, pending)
        ]
        visual = annotate_tracklet_boxes(visual, tracked.xyxy, internal_ids, labels)
    return draw_status_lines(
        visual,
        [
            "Tracker: McByte++",
            f"Frame: {number}",
            f"Output IDs: {ids}",
            *(
                [f"Pending internal IDs: {[identity for identity, waiting in zip(internal_ids, pending) if waiting]}"]
                if tracker.reid_joiner is not None
                else []
            ),
            f"Masks: {mask_ids}",
            f"Excluded mask IDs: {sorted(tracker.untrusted_mask_ids)}",
        ],
    )
