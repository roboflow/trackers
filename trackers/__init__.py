# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from trackers.annotators.trace import MotionAwareTraceAnnotator
from trackers.core.bytetrack.tracker import ByteTrackTracker
from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.core.sort.tracker import SORTTracker
from trackers.datasets.download import download_dataset
from trackers.datasets.manifest import Dataset, DatasetAsset, DatasetSplit
from trackers.io.video import frames_from_source
from trackers.motion.estimator import MotionEstimator
from trackers.motion.transformation import (
    CoordinatesTransformation,
    HomographyTransformation,
    IdentityTransformation,
)
from trackers.utils.converters import xcycsr_to_xyxy, xyxy_to_xcycsr

__all__ = [
    "ByteTrackTracker",
    "CoordinatesTransformation",
    "Dataset",
    "DatasetAsset",
    "DatasetSplit",
    "HomographyTransformation",
    "IdentityTransformation",
    "MotionAwareTraceAnnotator",
    "MotionEstimator",
    "OCSORTTracker",
    "SORTTracker",
    "download_dataset",
    "frames_from_source",
    "xcycsr_to_xyxy",
    "xyxy_to_xcycsr",
]
