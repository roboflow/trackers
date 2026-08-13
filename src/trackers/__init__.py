# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from trackers.annotators.trace import MotionAwareTraceAnnotator
from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.bytetrack.tracker import ByteTrackTracker
from trackers.core.cbiou.tracker import CBIoUTracker
from trackers.core.mcbyte.tracker import McByteMaskConfig, McByteTracker
from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.core.reid import (
    DEFAULT_FRAME_GAP_BANDS,
    AppearanceDistances,
    FeatureBank,
    ReIDEncoder,
    ThresholdLines,
    appearance_similarity,
    extract_detection_embeddings,
    plot_appearance_distances,
    plot_frame_gap_sweep,
    roc_auc,
    sample_appearance_distances,
    sweep_frame_gap,
)
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
from trackers.utils.cmc import CMC, CMCConfig, CMCMethod, CMCTMethod
from trackers.utils.converters import xcycsr_to_xyxy, xyxy_to_xcycsr
from trackers.utils.iou import BaseIoU, BIoU, CIoU, DIoU, GIoU, IoU

__all__ = [
    "CMC",
    "DEFAULT_FRAME_GAP_BANDS",
    "AppearanceDistances",
    "BIoU",
    "BaseIoU",
    "BoTSORTTracker",
    "ByteTrackTracker",
    "CBIoUTracker",
    "CIoU",
    "CMCConfig",
    "CMCMethod",
    "CMCTMethod",
    "CoordinatesTransformation",
    "DIoU",
    "Dataset",
    "DatasetAsset",
    "DatasetSplit",
    "FeatureBank",
    "GIoU",
    "HomographyTransformation",
    "IdentityTransformation",
    "IoU",
    "McByteMaskConfig",
    "McByteTracker",
    "MotionAwareTraceAnnotator",
    "MotionEstimator",
    "OCSORTTracker",
    "ReIDEncoder",
    "SORTTracker",
    "ThresholdLines",
    "appearance_similarity",
    "download_dataset",
    "extract_detection_embeddings",
    "frames_from_source",
    "load_mot_file",
    "plot_appearance_distances",
    "plot_frame_gap_sweep",
    "roc_auc",
    "sample_appearance_distances",
    "sweep_frame_gap",
    "xcycsr_to_xyxy",
    "xyxy_to_xcycsr",
]
