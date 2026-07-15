# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

from trackers.core.reid._lazy import REID_INSTALL_HINT, import_reid_symbol

from trackers.annotators.trace import MotionAwareTraceAnnotator
from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.bytetrack.tracker import ByteTrackTracker
from trackers.core.cbiou.tracker import CBIoUTracker
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
from trackers.utils.cmc import CMC, CMCConfig, CMCMethod, CMCTMethod
from trackers.utils.converters import xcycsr_to_xyxy, xyxy_to_xcycsr
from trackers.utils.iou import BaseIoU, BIoU, CIoU, DIoU, GIoU, IoU

if TYPE_CHECKING:
    from trackers.core.reid.model import ReIDModel

__all__ = [
    "CMC",
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
    "GIoU",
    "HomographyTransformation",
    "IdentityTransformation",
    "IoU",
    "MotionAwareTraceAnnotator",
    "MotionEstimator",
    "OCSORTTracker",
    "ReIDModel",
    "SORTTracker",
    "download_dataset",
    "frames_from_source",
    "load_mot_file",
    "xcycsr_to_xyxy",
    "xyxy_to_xcycsr",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ReIDModel": ("trackers.core.reid.model", "ReIDModel"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        try:
            value = import_reid_symbol(module_name, attr_name)
        except ImportError as exc:
            raise ImportError(REID_INSTALL_HINT) from exc
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
