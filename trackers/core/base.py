# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import inspect
import re
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Union, get_args, get_origin

import numpy as np
import supervision as sv


@dataclass
class ParameterInfo:
    """Definition of a tracker parameter."""

    param_type: type
    default_value: Any
    description: str


@dataclass
class TrackerInfo:
    """Metadata about a registered tracker."""

    tracker_class: type[BaseTracker]
    parameters: dict[str, ParameterInfo]


# Pattern: leading whitespace, optional backticks, param name (supports dotted),
# optional (type info), colon, and captures description
_PARAM_START_PATTERN = re.compile(
    r"^\s*`?(\w+(?:\.\w+)*)`?\s*(?:\([^)]*\))?\s*:\s*(.*)$"
)


def _parse_docstring_arguments(docstring: str) -> dict[str, str]:
    """Parse Google-style docstring Args section.

    Handles various formats:
        param: description
        param (type): description
        param (type, optional): description
        `param`: description
        param.sub: dotted parameter names

    Args:
        docstring: The docstring to parse.

    Returns:
        Dict mapping parameter names to their descriptions.
    """
    if not docstring:
        return {}

    result: dict[str, str] = {}
    lines = docstring.splitlines()
    i = 0
    n = len(lines)

    # Find Args: section
    while i < n:
        if lines[i].strip() == "Args:":
            i += 1
            break
        i += 1

    if i == n:
        return {}

    current_param: str | None = None
    current_desc_parts: list[str] = []

    while i < n:
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped in (
            "Returns:",
            "Yields:",
            "Raises:",
            "Attributes:",
            "Note:",
            "Notes:",
            "Example:",
            "Examples:",
            "See Also:",
        ):
            break

        match = _PARAM_START_PATTERN.match(line)
        if match:
            if current_param:
                result[current_param] = " ".join(current_desc_parts).strip()
            current_param = match.group(1)
            desc_first = match.group(2).strip()
            current_desc_parts = [desc_first] if desc_first else []
        elif current_param:
            current_desc_parts.append(stripped)

        i += 1

    if current_param:
        result[current_param] = " ".join(current_desc_parts).strip()

    return result


def _normalize_type(annotation: Any, default: Any) -> Any:
    """Convert complex type annotations to simple types for CLI/config.

    Handles Optional[T], T | None, List[T], etc. and extracts the base type.

    Args:
        annotation: The type annotation to normalize.
        default: The default value (used for fallback type inference).

    Returns:
        A simple type suitable for CLI argument parsing, or Any if unknown.
    """
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        if annotation is Any and default is not None:
            return type(default)
        return annotation if isinstance(annotation, type) else Any

    # Handle Union types (typing.Union and Python 3.10+ int | None syntax)
    union_type = getattr(types, "UnionType", None)
    if origin is Union or (union_type is not None and origin is union_type):
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _normalize_type(non_none[0], default)
        return Any

    if origin in (list, tuple, set, frozenset):
        return origin

    if origin is dict:
        return dict

    if default is not None:
        return type(default)
    return Any


def _extract_params_from_init(cls: type) -> dict[str, ParameterInfo]:
    """Extract parameters from a class's __init__ signature and docstring.

    Args:
        cls: The class to extract parameters from.

    Returns:
        Dict mapping parameter names to ParameterInfo objects.
    """
    sig = inspect.signature(cls.__init__)  # type: ignore[misc]

    try:
        from typing import get_type_hints

        type_hints = get_type_hints(cls.__init__)  # type: ignore[misc]
    except Exception:
        type_hints = {}

    # Check __init__ docstring first, then fall back to class docstring
    init_doc = cls.__init__.__doc__ or ""  # type: ignore[misc]
    class_doc = cls.__doc__ or ""
    param_docs = _parse_docstring_arguments(init_doc) or _parse_docstring_arguments(
        class_doc
    )

    params: dict[str, ParameterInfo] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue

        default = (
            param.default if param.default is not inspect.Parameter.empty else None
        )

        annotation = type_hints.get(name, Any)
        param_type = _normalize_type(annotation, default)

        # Fallback: infer from default if annotation is Any
        if param_type is Any and default is not None:
            param_type = type(default)

        description = param_docs.get(name, "")

        params[name] = ParameterInfo(
            param_type=param_type, default_value=default, description=description
        )

    return params


class BaseTracker(ABC):
    """Base class for all trackers with auto-registration.

    Subclasses that define a `tracker_id` class variable will be automatically
    registered and discoverable via `BaseTracker.get_info()` and
    `BaseTracker.available()`.

    Example:
        class MyTracker(BaseTracker):
            tracker_id = "mytracker"

            def __init__(self, param1: int = 10) -> None:
                '''
                Args:
                    param1: Description of param1.
                '''
                self.param1 = param1
    """

    _registry: ClassVar[dict[str, TrackerInfo]] = {}
    tracker_id: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register subclasses that define tracker_id."""
        super().__init_subclass__(**kwargs)

        tracker_id = getattr(cls, "tracker_id", None)
        if tracker_id is not None:
            BaseTracker._registry[tracker_id] = TrackerInfo(
                tracker_class=cls,
                parameters=_extract_params_from_init(cls),
            )

    @classmethod
    def get_info(cls, tracker_id: str) -> TrackerInfo:
        """Get tracker info by ID.

        Args:
            tracker_id: The tracker identifier (e.g., "bytetrack", "sort").

        Returns:
            TrackerInfo containing the tracker class and its parameters.

        Raises:
            ValueError: If tracker_id is not found in the registry.
        """
        if tracker_id not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"Unknown tracker ID: {tracker_id!r}\nAvailable trackers: {available}"
            )
        return cls._registry[tracker_id]

    @classmethod
    def available(cls) -> list[str]:
        """List available tracker IDs.

        Returns:
            List of registered tracker identifiers.
        """
        return sorted(cls._registry.keys())

    @abstractmethod
    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker with new detections and return tracked objects.

        Args:
            detections (sv.Detections): New detections for the current frame
                (xyxy, class_id, confidence, etc.).

        Returns:
            sv.Detections: The input detections enriched with tracker_id attribute.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""
        pass


class BaseTrackerWithFeatures(ABC):
    """Base class for trackers that require image features (e.g., ReID)."""

    @abstractmethod
    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """Update tracker with detections and frame image.

        Args:
            detections: New detections for the current frame.
            frame: The current video frame.

        Returns:
            Detections with tracker_id assigned.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""
        pass
