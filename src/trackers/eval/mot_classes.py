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

#: Names of the built-in MOT class presets, accepted case-insensitively by `resolve_mot_class_config`.
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
    """Configure which MOT ground-truth classes are scored and which act as distractors.

    A ground-truth row is scored when its confidence flag is non-zero and its class is in ``scored_classes``. Tracker
    detections that best match a row of a distractor class are removed during preprocessing, so they count as neither
    false positives nor true positives. The defaults reproduce TrackEval's MOT17 convention
    (``trackeval/datasets/mot_challenge_2d_box.py``).

    Both fields are normalized to tuples of ints on creation, so lists and NumPy integers are accepted and the instance
    stays hashable.

    Attributes:
        scored_classes: Class IDs scored as ground truth. Must be non-empty. TrackEval scores only pedestrians
            (class 1); scoring several classes is an extension of this configuration.
        distractor_classes: Class IDs whose regions suppress matching tracker detections. Must not overlap
            ``scored_classes``. MOT17 uses 2 (person_on_vehicle), 7 (static_person), 8 (distractor) and
            12 (reflection); MOT20 also adds 6 (non_mot_vehicle).

    Raises:
        ValueError: If a field is not an iterable of integer class IDs, if ``scored_classes`` is empty, or if a class
            appears in both fields.

    Example:
        >>> from trackers.eval.mot_classes import MOTClassConfig
        >>> MOTClassConfig(scored_classes=[1], distractor_classes=[2, 6])
        MOTClassConfig(scored_classes=(1,), distractor_classes=(2, 6))
    """

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


#: Read-only registry of the built-in presets: ``mot17`` (TrackEval MOT17 rules) and ``mot20`` (also treats class 6 as
#: a distractor).
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

    Example:
        >>> from trackers.eval.mot_classes import resolve_mot_class_config
        >>> resolve_mot_class_config("mot20").distractor_classes
        (2, 6, 7, 8, 12)
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
