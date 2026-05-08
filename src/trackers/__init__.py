# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from trackers.annotators.trace import MotionAwareTraceAnnotator
from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.bytetrack.tracker import ByteTrackTracker
from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.core.sort.tracker import SORTTracker
from trackers.datasets.download import download_dataset
from trackers.datasets.manifest import Dataset, DatasetAsset, DatasetSplit
from trackers.io.mot import load_mot_file
from trackers.io.video import frames_from_source
from trackers.motion.estimator import MotionEstimator
from trackers.motion.transformation import (
    CoordinatesTransformation,
    HomographyTransformation,
    IdentityTransformation,
)
from trackers.utils.converters import xcycsr_to_xyxy, xyxy_to_xcycsr
from trackers.utils.iou import BaseIoU, BIoU, CIoU, DIoU, GIoU, IoU

__all__ = [
    "BIoU",
    "BaseIoU",
    "BoTSORTTracker",
    "ByteTrackTracker",
    "CIoU",
    "CoordinatesTransformation",
    "DIoU",
    "Dataset",
    "DatasetAsset",
    "DatasetSplit",
    "GIoU",
    "HomographyTransformation",
    "IdentityTransformation",
    "IoU",
    "MotionAwareTraceAnnotator",
    "MotionEstimator",
    "OCSORTTracker",
    "SORTTracker",
    "download_dataset",
    "frames_from_source",
    "load_mot_file",
    "xcycsr_to_xyxy",
    "xyxy_to_xcycsr",
]
