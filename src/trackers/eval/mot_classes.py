# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Ground-truth class conventions used by MOT evaluation."""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

MOTClassPreset = Literal["mot17", "mot20"]


def _coerce_class_ids(field_name: str, values: Iterable[int]) -> tuple[int, ...]:
    """Normalize an iterable of class IDs into a tuple of Python ints.

    Args:
        field_name: Name of the configuration field, used in error messages.
        values: Class IDs. Python and NumPy integers are accepted.

    Returns:
        Class IDs as a tuple of Python ints, in the order given.

    Raises:
        ValueError: If ``values`` is not an iterable of integer class IDs.
    """
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be an iterable of integer class IDs, got {values!r}")
    try:
        items = tuple(values)
    except TypeError as error:
        raise ValueError(f"{field_name} must be an iterable of integer class IDs, got {values!r}") from error

    class_ids: list[int] = []
    for item in items:
        try:
            class_id = operator.index(item)
        except TypeError as error:
            raise ValueError(f"{field_name} must contain only integer class IDs, got {item!r}") from error
        if isinstance(item, bool):
            raise ValueError(f"{field_name} must contain only integer class IDs, got {item!r}")
        class_ids.append(class_id)
    return tuple(class_ids)


@dataclass(frozen=True)
class MOTClassConfig:
    """Configure which MOT ground-truth classes are scored or treated as distractors."""

    scored_classes: tuple[int, ...] = (1,)
    distractor_classes: tuple[int, ...] = (2, 7, 8, 12)

    def __post_init__(self) -> None:
        """Normalize both class sets to int tuples and reject contradictory configurations.

        Raises:
            ValueError: If a field is not an iterable of integer class IDs, if ``scored_classes`` is empty, or if
                a class appears in both ``scored_classes`` and ``distractor_classes``.
        """
        scored = _coerce_class_ids("scored_classes", self.scored_classes)
        distractor = _coerce_class_ids("distractor_classes", self.distractor_classes)
        if not scored:
            raise ValueError("scored_classes must contain at least one class ID")
        overlap = sorted(set(scored) & set(distractor))
        if overlap:
            raise ValueError(f"Classes cannot be both scored and distractor: {overlap}")
        # frozen dataclass: bypass __setattr__ so the instance stays hashable with normalized tuples
        object.__setattr__(self, "scored_classes", scored)
        object.__setattr__(self, "distractor_classes", distractor)


MOT_CLASS_PRESETS: Mapping[MOTClassPreset, MOTClassConfig] = MappingProxyType(
    {
        "mot17": MOTClassConfig(),
        "mot20": MOTClassConfig(distractor_classes=(2, 6, 7, 8, 12)),
    }
)


def resolve_mot_class_config(class_config: MOTClassPreset | MOTClassConfig) -> MOTClassConfig:
    """Resolve a preset name or return a caller-supplied MOT class configuration.

    Args:
        class_config: Preset name (case-insensitive, surrounding whitespace ignored) or explicit class configuration.

    Returns:
        Resolved MOT class configuration.

    Raises:
        ValueError: If ``class_config`` is neither a supported preset name nor a ``MOTClassConfig``.
    """
    if isinstance(class_config, MOTClassConfig):
        return class_config

    supported_names = ", ".join(MOT_CLASS_PRESETS)
    if not isinstance(class_config, str):
        raise ValueError(
            f"Expected a MOT class preset name or MOTClassConfig, got {type(class_config).__name__}. "
            f"Supported presets: {supported_names}"
        )

    normalized_name = class_config.strip().lower()
    for preset_name, preset_config in MOT_CLASS_PRESETS.items():
        if preset_name == normalized_name:
            return preset_config

    raise ValueError(f"Unsupported MOT class preset {class_config!r}. Supported presets: {supported_names}")
