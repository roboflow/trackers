# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Model loading: preprocessing, registry, and checkpoint loaders."""

from trackers.core.reid.models.loaders import (
    KeyReport,
    load_fastreid_sbs_state_dict_into,
    load_state_dict_for_architecture,
    load_state_dict_into,
    remap_fastreid_sbs_state_dict,
    resolve_weights,
)
from trackers.core.reid.models.preprocessing import ReIDPreprocessing
from trackers.core.reid.models.registry import (
    DEFAULT_MODEL,
    FASTREID_MOT17_SBS50,
    ModelCard,
    default_preprocessing_for_architecture,
    load_model_config,
    resolve_model_card,
    save_model_config,
)

__all__ = [
    "DEFAULT_MODEL",
    "FASTREID_MOT17_SBS50",
    "KeyReport",
    "ModelCard",
    "ReIDPreprocessing",
    "default_preprocessing_for_architecture",
    "load_fastreid_sbs_state_dict_into",
    "load_model_config",
    "load_state_dict_for_architecture",
    "load_state_dict_into",
    "remap_fastreid_sbs_state_dict",
    "resolve_model_card",
    "resolve_weights",
    "save_model_config",
]
