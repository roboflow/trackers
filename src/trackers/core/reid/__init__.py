# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""NumPy-only appearance-ReID association glue.

This package intentionally contains no model stack. The encoder, weights,
preprocessing, and gallery evaluation live in the standalone ``reid`` package
(``pip install 'trackers[reid]'``); import ``ReIDModel`` and evaluation helpers
from there. Everything exported here is importable without torch.
"""

from __future__ import annotations

from trackers.core.reid.appearance import appearance_similarity, extract_detection_embeddings
from trackers.core.reid.encoder import ReIDEncoder
from trackers.core.reid.feature_bank import FeatureBank

__all__ = [
    "FeatureBank",
    "ReIDEncoder",
    "appearance_similarity",
    "extract_detection_embeddings",
]
