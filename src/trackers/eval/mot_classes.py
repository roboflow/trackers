# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

MOTClassPreset = Literal["mot17", "mot20"]


@dataclass(frozen=True)
class MOTClassConfig:
    """Configuration of scored and distractor ground-truth classes in MOT evaluation.

    Attributes:
        scored_classes: Ground-truth class IDs considered true targets (e.g. pedestrian = 1).
            Matched detections count as true positives.
        distractor_classes: Ground-truth class IDs considered distractors.
            Detections matching these regions are suppressed without penalty (neither FP nor TP).
    """

    scored_classes: tuple[int, ...] = (1,)
    distractor_classes: tuple[int, ...] = (2, 7, 8, 12)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scored_classes", tuple(self.scored_classes))
        object.__setattr__(self, "distractor_classes", tuple(self.distractor_classes))


MOT_CLASS_PRESETS: dict[MOTClassPreset, MOTClassConfig] = {
    "mot17": MOTClassConfig(),
    "mot20": MOTClassConfig(distractor_classes=(2, 6, 7, 8, 12)),
}


def resolve_mot_class_config(
    config: MOTClassPreset | MOTClassConfig | dict[str, Any] | None = None,
) -> MOTClassConfig:
    """Resolve a preset name, dict, or config instance into a `MOTClassConfig`.

    Args:
        config: Either a preset name ("mot17", "mot20"), a `MOTClassConfig` instance,
            a dict with class configuration, or None (defaults to "mot17").

    Returns:
        The resolved `MOTClassConfig`.

    Raises:
        ValueError: If an unknown preset name is provided.
        TypeError: If an unsupported config type is provided.
    """
    if config is None:
        return MOT_CLASS_PRESETS["mot17"]
    if isinstance(config, MOTClassConfig):
        return config
    if isinstance(config, str):
        if config in MOT_CLASS_PRESETS:
            return MOT_CLASS_PRESETS[config]  # type: ignore[index]
        supported = ", ".join(f"'{p}'" for p in sorted(MOT_CLASS_PRESETS))
        raise ValueError(f"Unknown MOT class preset: '{config}'. Supported presets: {supported}.")
    if isinstance(config, dict):
        scored = tuple(config.get("scored_classes", (1,)))
        distractor = tuple(config.get("distractor_classes", (2, 7, 8, 12)))
        return MOTClassConfig(scored_classes=scored, distractor_classes=distractor)
    raise TypeError(f"Expected MOTClassPreset or MOTClassConfig, got {type(config).__name__}")
