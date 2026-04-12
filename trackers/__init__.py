# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from trackers.annotators.trace import MotionAwareTraceAnnotator
from trackers.calibration.base import PitchCalibrator
from trackers.calibration.pitch import PitchModel
from trackers.calibration.projection import (
    apply_homography,
    bottom_center_from_xywh,
    bottom_center_from_xyxy,
    invert_homography,
    project_image_points_to_pitch,
    project_pitch_points_to_image,
)
from trackers.calibration.providers.pnlcalib import PnLCalibProvider
from trackers.calibration.smoothing import HoldLastCalibration, fill_calibration_gaps
from trackers.calibration.types import (
    CalibrationFrame,
    PitchDimensions,
    TrackProjection,
)
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
    "CalibrationFrame",
    "CoordinatesTransformation",
    "Dataset",
    "DatasetAsset",
    "DatasetSplit",
    "HoldLastCalibration",
    "HomographyTransformation",
    "IdentityTransformation",
    "MotionAwareTraceAnnotator",
    "MotionEstimator",
    "OCSORTTracker",
    "PitchCalibrator",
    "PitchDimensions",
    "PitchModel",
    "PnLCalibProvider",
    "SORTTracker",
    "TrackProjection",
    "apply_homography",
    "bottom_center_from_xywh",
    "bottom_center_from_xyxy",
    "download_dataset",
    "fill_calibration_gaps",
    "frames_from_source",
    "invert_homography",
    "project_image_points_to_pitch",
    "project_pitch_points_to_image",
    "xcycsr_to_xyxy",
    "xyxy_to_xcycsr",
]
