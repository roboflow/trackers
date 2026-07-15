# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Model loading helpers: preprocessing, curated registry, and checkpoint I/O.

The registry (``registry.py``) picks a pretrained *recipe*. Loaders fetch and
apply weight files. Preprocessing owns crop / resize / embedding norms.
Architecture builders live under ``trackers.core.reid.architectures``.
"""

from trackers.core.reid.models.loaders import (
    KeyReport,
    load_state_dict_for_architecture,
    load_state_dict_into,
    resolve_weights,
)
from trackers.core.reid.models.preprocessing import ReIDPreprocessing
from trackers.core.reid.models.registry import (
    DEFAULT_MODEL,
    ModelCard,
    default_preprocessing_for_architecture,
    load_model_config,
    resolve_model_card,
    save_model_config,
)

__all__ = [
    "DEFAULT_MODEL",
    "KeyReport",
    "ModelCard",
    "ReIDPreprocessing",
    "default_preprocessing_for_architecture",
    "load_model_config",
    "load_state_dict_for_architecture",
    "load_state_dict_into",
    "resolve_model_card",
    "resolve_weights",
    "save_model_config",
]
