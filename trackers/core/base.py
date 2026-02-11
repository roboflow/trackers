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

import supervision as sv


@dataclass
class ParameterInfo:
    """Holds metadata for a single tracker parameter.

    Stores the type, default value, and description extracted from the
    tracker's __init__ signature and docstring.
    """

    param_type: type
    default_value: Any
    description: str


@dataclass
class TrackerInfo:
    """Holds a tracker class and its extracted parameter metadata.

    Used by the CLI to discover available trackers and their configurable
    options without instantiating them.
    """

    tracker_class: type[BaseTracker]
    parameters: dict[str, ParameterInfo]


# Pattern: leading whitespace, optional backticks, param name (supports dotted),
# optional (type info), colon, and captures description
_PARAM_START_PATTERN = re.compile(
    r"^\s*`?(\w+(?:\.\w+)*)`?\s*(?:\([^)]*\))?\s*:\s*(.*)$"
)


def _parse_docstring_arguments(docstring: str) -> dict[str, str]:
    """Extract parameter-to-description mapping from Google-style Args section.

    Supports multiple formats including `param: desc`, `param (type): desc`,
    and multi-line descriptions with proper continuation handling.

    Args:
        docstring: Raw docstring text to parse.

    Returns:
        Mapping of parameter names to their description strings.
        Empty dict if no Args section found.
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
    """Unwrap Optional/Union/generics to base type for CLI argument parsing.

    Converts complex annotations like Optional[int], list[str], or int | None
    to their base types (int, list, int) suitable for argparse type conversion.

    Args:
        annotation: Type annotation to simplify.
        default: Default value used for fallback type inference when
            annotation is Any or cannot be resolved.

    Returns:
        Simplified type (e.g., int, str, list) or Any if unresolvable.
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
    """Introspect __init__ signature and docstring to build parameter metadata.

    Combines type hints, default values, and docstring descriptions into a
    structured format. Falls back to class docstring if __init__ has none.

    Args:
        cls: Class whose __init__ to analyze.

    Returns:
        Mapping of parameter names to ParameterInfo objects.
        Excludes 'self' parameter.
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
    """Abstract tracker with auto-registration via tracker_id class variable.

    Subclasses that define `tracker_id` are automatically registered and
    become discoverable. Parameter metadata is extracted from __init__ for
    CLI integration.
    """

    _registry: ClassVar[dict[str, TrackerInfo]] = {}
    tracker_id: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclass in the tracker registry if it defines tracker_id.

        Extracts parameter metadata from __init__ at class definition time.
        """
        super().__init_subclass__(**kwargs)

        tracker_id = getattr(cls, "tracker_id", None)
        if tracker_id is not None:
            BaseTracker._registry[tracker_id] = TrackerInfo(
                tracker_class=cls,
                parameters=_extract_params_from_init(cls),
            )

    @classmethod
    def _lookup_tracker(cls, name: str) -> TrackerInfo | None:
        """Look up registered tracker by name.

        Internal method used by CLI for tracker discovery and instantiation.

        Args:
            name: Tracker identifier (e.g., "bytetrack", "sort").

        Returns:
            TrackerInfo containing class and parameters if found,
            None otherwise.
        """
        return cls._registry.get(name)

    @classmethod
    def _registered_trackers(cls) -> list[str]:
        """List all registered tracker names.

        Internal method used by CLI for help text and argument validation.

        Returns:
            Alphabetically sorted list of tracker identifiers.
        """
        return sorted(cls._registry.keys())

    @abstractmethod
    def update(self, detections: sv.Detections) -> sv.Detections:
        """Process new detections and assign track IDs.

        Matches incoming detections to existing tracks, creates new tracks
        for unmatched detections, and handles track lifecycle management.

        Args:
            detections: Current frame detections with xyxy, confidence, class_id.

        Returns:
            Same detections enriched with tracker_id attribute for each box.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal tracking state.

        Call between videos or when tracking should restart from scratch.
        """
        pass
