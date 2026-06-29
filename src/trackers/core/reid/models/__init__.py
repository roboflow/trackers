# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Model loading: preprocessing, registry, and checkpoint loaders."""

from trackers.core.reid.models.loaders import KeyReport, load_state_dict_into, resolve_weights
from trackers.core.reid.models.preprocessing import ReIDPreprocessing
from trackers.core.reid.models.registry import (
    DEFAULT_MODEL,
    ModelCard,
    load_model_config,
    resolve_model_card,
    save_model_config,
)

__all__ = [
    "DEFAULT_MODEL",
    "KeyReport",
    "ModelCard",
    "ReIDPreprocessing",
    "load_model_config",
    "load_state_dict_into",
    "resolve_model_card",
    "resolve_weights",
    "save_model_config",
]
