# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID training utilities (patch generation, fine-tuning, retrieval splits)."""

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

_LAZY_TRAINER_SYMBOLS = frozenset({"TrainConfig", "TrainResult", "train_reid"})


def __getattr__(name: str):
    if name in _LAZY_TRAINER_SYMBOLS:
        from trackers.core.reid.training import trainer

        value = getattr(trainer, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
