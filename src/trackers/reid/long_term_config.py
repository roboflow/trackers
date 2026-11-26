# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Configuration for joining terminated and newly confirmed tracklets."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LongTermReIDConfig:
    """Configure optional long-term identity joining independently of a tracker.

    Descriptor aggregation and matching thresholds follow the original McByte++
    appearance policy. By default, a subject-occlusion check additionally skips
    new re-ID feature extraction and pauses an affected pending tracklet's
    identity decision deadline. Existing descriptors can still support a join.
    Each successful extraction counts once; one actual descriptor permits matching by default.
    Descriptors are individually L2-normalized, retained in a bounded history,
    averaged and normalized again. This is not exponential moving averaging.

    Attributes:
        minimum_detection_confidence: Minimum matched detection confidence for
            extraction, inclusive. Defaults to 0.6.
        minimum_box_width: Minimum crop width in pixels, inclusive. Defaults to 25.
        minimum_box_height: Minimum crop height in pixels, inclusive. Defaults to 50.
        minimum_box_area: Minimum crop area in pixels, inclusive. Defaults to 1024.
        border_margin: Crop coordinates must lie strictly inside this image
            margin in pixels. Defaults to 4.
        minimum_feature_frame_gap: Minimum processing steps between successful
            extractions for one tracklet. Defaults to 5. Duplicate timestamps
            do not advance this clock; non-duplicate missing-image updates do.
        feature_history_size: Maximum descriptors retained per tracklet. Defaults
            to 50. The count of successful samples is independent of this limit.
        minimum_feature_samples: Actual successful samples required on both
            sides of a join. Defaults to 1.
        minimum_cosine_similarity: Minimum cosine similarity for a mutual-best
            match, inclusive. Defaults to 0.8.
        maximum_reconnection_age: A newly confirmed tracklet can reconnect
            through this many processing steps after its original birth,
            inclusive. Defaults to 10. A currently observed pending identity
            becomes a new public identity once its age exceeds this limit.
            The calculation is
            ``decision_age = current_step - birth_step - paused_steps``.
            This processing clock is separate from the frame references used
            for historical output and does not measure elapsed wall-clock time.
        maximum_reconnection_pause_steps: Maximum extra waiting steps allowed
            per pending tracklet across all pause requests. Defaults to ``None``
            (unlimited); zero disables extension. Once the cap is consumed,
            decision age advances normally even while extraction is suppressed.
            This does not cap unread output or force an appearance match.
        buffer_results: Retain eligible observations returned with public ID
            ``-1`` for later retrieval with their resolved IDs. Defaults to True.
            No images, segmentation masks or collection metadata are retained.
            Keep enabled for complete trajectories, especially when decision
            ages can pause. Without buffering, pending observations are lost
            even if an ID is later assigned at termination or finish. Longer
            pauses can therefore omit valid track fragments or entire tracklets
            from text output; observations with established IDs are unaffected.
        enable_feature_occlusion_filter: Defer extraction when another current
            detector box appears to occlude the subject. Defaults to True.
            Applies to pending and established identities. To disable this
            filter, also set ``pause_reconnection_on_occlusion=False``.
        pause_reconnection_on_occlusion: Also pause the affected pending
            tracklet's joining deadline on an occluded image. Defaults to True;
            requires ``enable_feature_occlusion_filter=True``. On advancing
            updates with an image and host-enabled feature extraction, check
            observed pending tracklets even when their next sample is not due.
            Confidence, size and border rejection alone do not request a pause;
            occlusion can still request one when another rejection reason also
            applies. Existing descriptors can still support a join.
            Occlusion and a host-requested pause count once per step and share
            ``maximum_reconnection_pause_steps``. Missing images and duplicate
            processing steps do not request this additional pause.
        feature_occlusion_overlap_threshold: Maximum individual intersection
            divided by target area over boxes with a greater bottom y coordinate.
            Extraction is deferred at or above this value. Defaults to 0.6.
            This box-only heuristic has no dependency on mask creation settings.

    Raises:
        ValueError: If a setting is outside its supported range.
    """

    minimum_detection_confidence: float = 0.6
    minimum_box_width: float = 25
    minimum_box_height: float = 50
    minimum_box_area: float = 1024
    border_margin: float = 4
    minimum_feature_frame_gap: int = 5
    feature_history_size: int = 50
    minimum_feature_samples: int = 1
    minimum_cosine_similarity: float = 0.8
    maximum_reconnection_age: int = 10
    buffer_results: bool = True
    enable_feature_occlusion_filter: bool = True
    feature_occlusion_overlap_threshold: float = 0.6
    maximum_reconnection_pause_steps: int | None = None
    pause_reconnection_on_occlusion: bool = True

    def __post_init__(self) -> None:
        for name in ("minimum_detection_confidence", "feature_occlusion_overlap_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if not math.isfinite(self.minimum_cosine_similarity) or not -1 <= self.minimum_cosine_similarity <= 1:
            raise ValueError("minimum_cosine_similarity must be finite and in [-1, 1].")
        for name in ("minimum_box_width", "minimum_box_height", "minimum_box_area", "border_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in ("minimum_feature_frame_gap", "feature_history_size", "minimum_feature_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        age = self.maximum_reconnection_age
        if isinstance(age, bool) or not isinstance(age, int) or age < 0:
            raise ValueError("maximum_reconnection_age must be a non-negative integer.")
        cap = self.maximum_reconnection_pause_steps
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 0):
            raise ValueError("maximum_reconnection_pause_steps must be None or a non-negative integer.")
        for name in ("buffer_results", "enable_feature_occlusion_filter", "pause_reconnection_on_occlusion"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if self.pause_reconnection_on_occlusion and not self.enable_feature_occlusion_filter:
            raise ValueError("pause_reconnection_on_occlusion requires enable_feature_occlusion_filter=True.")
