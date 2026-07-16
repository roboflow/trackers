# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Re-ID training utilities (patch generation, fine-tuning, retrieval splits)."""

from typing import TYPE_CHECKING

from trackers.core.reid.training.patches import (
    PatchGenerationStats,
    generate_mot_patches,
)
from trackers.core.reid.training.retrieval import (
    build_identity_holdout_split,
    build_retrieval_split,
)

if TYPE_CHECKING:
    from trackers.core.reid.training.trainer import (
        TrainConfig,
        TrainResult,
        train_reid,
    )

__all__ = [
    "PatchGenerationStats",
    "TrainConfig",
    "TrainResult",
    "build_identity_holdout_split",
    "build_retrieval_split",
    "generate_mot_patches",
    "train_reid",
]

# Lazily expose the torch-backed training entry points so that importing this
# package (e.g. for patch generation) does not require the optional [reid]
# dependencies. They are imported on first attribute access.
_LAZY_TRAINER_SYMBOLS = {"TrainConfig", "TrainResult", "train_reid"}


def __getattr__(name: str):
    if name in _LAZY_TRAINER_SYMBOLS:
        from trackers.core.reid.training import trainer

        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
