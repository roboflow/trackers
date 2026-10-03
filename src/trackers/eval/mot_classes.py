# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Ground-truth class conventions used by MOT evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MOTClassPreset = Literal["mot17", "mot20"]


@dataclass(frozen=True)
class MOTClassConfig:
    """Configure which MOT ground-truth classes are scored or treated as distractors."""

    scored_classes: tuple[int, ...] = (1,)
    distractor_classes: tuple[int, ...] = (2, 7, 8, 12)


MOT_CLASS_PRESETS: dict[MOTClassPreset, MOTClassConfig] = {
    "mot17": MOTClassConfig(),
    "mot20": MOTClassConfig(distractor_classes=(2, 6, 7, 8, 12)),
}


def resolve_mot_class_config(class_config: MOTClassPreset | MOTClassConfig) -> MOTClassConfig:
    """Resolve a preset name or return a caller-supplied MOT class configuration.

    Args:
        class_config: Preset name or explicit class configuration.

    Returns:
        Resolved MOT class configuration.

    Raises:
        ValueError: If ``class_config`` is not a supported preset or config object.
    """
    if isinstance(class_config, MOTClassConfig):
        return class_config

    if isinstance(class_config, str) and class_config in MOT_CLASS_PRESETS:
        return MOT_CLASS_PRESETS[class_config]

    supported_names = ", ".join(MOT_CLASS_PRESETS)
    raise ValueError(f"Unsupported MOT class preset {class_config!r}. Supported presets: {supported_names}")
