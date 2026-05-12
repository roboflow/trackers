# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import inspect
import re
import types
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, Union, get_args, get_origin

import numpy as np
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


class TrackerParameters(dict[str, ParameterInfo]):
    """Tracker parameter mapping with CLI-only filtering for IoU metrics."""

    def items(self) -> Iterator[tuple[str, ParameterInfo]]:  # type: ignore[override]
        try:
            from trackers.utils.iou import BaseIoU
        except ImportError:
            yield from super().items()
            return

        for name, param_info in super().items():
            param_type = param_info.param_type
            if isinstance(param_type, type) and issubclass(param_type, BaseIoU):
                continue
            yield name, param_info


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
_PARAM_START_PATTERN = re.compile(r"^\s*`?(\w+(?:\.\w+)*)`?\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def _parse_docstring_arguments(docstring: str) -> dict[str, str]:
    """Extract parameter-to-description mapping from Google-style Args section.

    Supports multiple formats including `param: desc`, `param (type): desc`,
    and multi-line descriptions with proper continuation handling.

    Args:
        docstring: Raw docstring text to parse.

    Returns:
        Mapping of parameter names to their description strings.
        Empty dict if no Args section is found in the docstring.
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
        Simplified type (e.g., int, str, list) suitable for argparse type
        conversion, or Any if the annotation cannot be resolved to a concrete
        type.
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
        Mapping of parameter names to ParameterInfo objects, excluding
        the ``self`` parameter.
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
    param_docs = _parse_docstring_arguments(init_doc) or _parse_docstring_arguments(class_doc)

    params: dict[str, ParameterInfo] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue

        default = param.default if param.default is not inspect.Parameter.empty else None

        annotation = type_hints.get(name, Any)
        param_type = _normalize_type(annotation, default)

        # Fallback: infer from default if annotation is Any
        if param_type is Any and default is not None:
            param_type = type(default)

        description = param_docs.get(name, "")

        params[name] = ParameterInfo(param_type=param_type, default_value=default, description=description)

    return params


_VALID_SPACE_TYPES: frozenset[str] = frozenset({"randint", "uniform", "choice"})


def _validate_search_space_entry(cls_name: str, key: str, spec: Any, init_params: set[str]) -> None:
    if key not in init_params:
        raise ValueError(
            f"{cls_name}: search_space key {key!r} is not a "
            f"parameter of __init__. "
            f"Valid parameters: {sorted(init_params)}"
        )
    if not isinstance(spec, dict):
        raise ValueError(f"{cls_name}: search_space[{key!r}] must be a dict, got {type(spec).__name__!r}")
    if "type" not in spec:
        raise ValueError(
            f"{cls_name}: search_space[{key!r}] missing required key 'type'. Valid types: {sorted(_VALID_SPACE_TYPES)}"
        )
    if spec["type"] not in _VALID_SPACE_TYPES:
        raise ValueError(
            f"{cls_name}: search_space[{key!r}]['type'] = "
            f"{spec['type']!r} is not valid. "
            f"Valid types: {sorted(_VALID_SPACE_TYPES)}"
        )
    space_type = spec["type"]
    if space_type == "choice":
        if "options" not in spec:
            raise ValueError(f"{cls_name}: search_space[{key!r}] with type 'choice' missing required key 'options'")
        opts = spec["options"]
        if isinstance(opts, (str, bytes)):
            raise ValueError(
                f"{cls_name}: search_space[{key!r}]['options'] must be "
                f"a sequence of choices, not {type(opts).__name__!r}"
            )
        try:
            n_opts = len(opts)
        except TypeError as exc:
            raise ValueError(
                f"{cls_name}: search_space[{key!r}]['options'] must be a sized sequence, got {type(opts).__name__!r}"
            ) from exc
        if n_opts < 1:
            raise ValueError(f"{cls_name}: search_space[{key!r}]['options'] must be non-empty, got {opts!r}")
        return

    if "range" not in spec:
        raise ValueError(f"{cls_name}: search_space[{key!r}] missing required key 'range'")
    rng = spec["range"]
    if not (hasattr(rng, "__len__") and len(rng) == 2):
        raise ValueError(f"{cls_name}: search_space[{key!r}]['range'] must be a 2-element sequence, got {rng!r}")
    if rng[0] >= rng[1]:
        raise ValueError(f"{cls_name}: search_space[{key!r}]['range'] must have low < high, got {rng!r}")


class TrackletProtocol(Protocol):
    """Contract every tracklet in ``BaseTracker.tracks`` must satisfy."""

    tracker_id: int

    def get_state_bbox(self) -> np.ndarray: ...


class BaseTracker(ABC):
    """Abstract tracker with auto-registration via tracker_id class variable.

    Subclasses that define `tracker_id` are automatically registered and
    become discoverable. Parameter metadata is extracted from __init__ for
    CLI integration.

    Attributes:
        tracker_id: Unique identifier for the tracker. Subclasses must define
            this to be registered.
        search_space: Hyperparameter search space for tuning. Each key must
            match an `__init__` parameter. Values are dicts with `type`
            ``"randint"`` or ``"uniform"`` and ``range`` ``[low, high]``, or
            `type` ``"choice"`` and ``options`` (non-empty sequence of
            categorical values for Optuna).
        tracks: List of alive tracklets after each `update()`. Each element
            must satisfy `TrackletProtocol` (exposes `.tracker_id: int` and
            `.get_state_bbox() -> np.ndarray`). Subclasses must initialise
            this as an empty list in `__init__`. Override `tracked_objects`
            if using a different internal container.
    """

    _registry: ClassVar[dict[str, TrackerInfo]] = {}
    tracker_id: ClassVar[str | None] = None
    search_space: ClassVar[dict[str, dict] | None] = None
    # list[Any]: elements satisfy TrackletProtocol; list is invariant so
    # list[ConcreteTracklet] in subclasses rejects list[TrackletProtocol] base.
    tracks: list[Any]
    maximum_frames_without_update: int

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclass in the tracker registry if it defines tracker_id.

        Extracts parameter metadata from __init__ at class definition time.
        Validates search_space (if present) against __init__ parameters.
        """
        super().__init_subclass__(**kwargs)

        # Validate search_space keys match __init__ parameters (search_space optional)
        search_space = getattr(cls, "search_space", None)
        if search_space is not None and len(search_space) > 0:
            init_params = {n for n in inspect.signature(cls.__init__).parameters if n != "self"}
            for key, spec in search_space.items():
                _validate_search_space_entry(cls.__name__, key, spec, init_params)

        tracker_id = getattr(cls, "tracker_id", None)
        if tracker_id is not None:
            BaseTracker._registry[tracker_id] = TrackerInfo(
                tracker_class=cls,
                parameters=TrackerParameters(_extract_params_from_init(cls)),
            )

    @classmethod
    def _lookup_tracker(cls, name: str) -> TrackerInfo | None:
        """Look up registered tracker by name.

        Internal method used by CLI for tracker discovery and instantiation.

        Args:
            name: Tracker identifier (e.g., "bytetrack", "sort").

        Returns:
            TrackerInfo containing the tracker class and its parameter
            metadata, or None if no tracker is registered under that name.
        """
        return cls._registry.get(name)

    @classmethod
    def _registered_trackers(cls) -> list[str]:
        """List all registered tracker names.

        Internal method used by CLI for help text and argument validation.

        Returns:
            Alphabetically sorted list of registered tracker identifiers
            (e.g., ``["bytetrack", "ocsort", "sort"]``).
        """
        return sorted(cls._registry.keys())

    def _warn_if_frame_unused(self, frame: np.ndarray | None) -> None:
        """Emit a UserWarning when a frame is passed to a tracker that ignores it.

        Subclasses that do not perform camera motion compensation should call this
        at the top of their ``update()`` implementation.

        Args:
            frame: Value passed to ``update(frame=...)``.
        """
        if frame is not None:
            warnings.warn(
                f"{type(self).__name__}.update() received a frame argument but does not use it.",
                UserWarning,
                stacklevel=3,
            )

    @abstractmethod
    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
    ) -> sv.Detections:
        """Process new detections and assign track IDs.

        Matches incoming detections to existing tracks, creates new tracks
        for unmatched detections, and handles track lifecycle management.

        Args:
            detections: Current frame detections with xyxy, confidence, class_id.
            frame: Current video frame in BGR format (H, W, 3), or ``None``.
                Used by trackers with camera motion compensation (e.g. BoTSORT).

        Returns:
            sv.Detections enriched with tracker_id assigned for each
            detection box.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal tracking state.

        Call between videos or when tracking should restart from scratch.
        """
        pass

    @property
    def tracked_objects(self) -> sv.Detections:
        """All confirmed alive tracks with Kalman-predicted bounding boxes.

        Exposes every confirmed track (tracker_id != -1) that the tracker
        still considers alive after the most recent `update()` call, including
        tracks not matched to a detection on the current frame (e.g.
        temporarily occluded or missed by the detector). Tracks are dropped
        once the time since the last matching detection exceeds
        `lost_track_buffer` (scaled by `frame_rate`).

        Unlike the `update()` return value, the result omits `confidence` and
        `class_id` (both remain `None`). Kalman-predicted boxes have no
        associated detection score or class label.

        Note:
            `sv.LabelAnnotator` and other supervision annotators that read
            `class_id` or `confidence` cannot be used directly on this result
            and will raise `TypeError`. Guard with
            ``if detections.class_id is not None`` before annotating.

        Returns:
            sv.Detections with Kalman-predicted xyxy and tracker_id for each
            confirmed alive track. Returns an empty sv.Detections (with an
            empty int tracker_id array) when no confirmed tracks are alive.
            The exact set depends on each tracker's pruning logic.

        Raises:
            AttributeError: If a `BaseTracker` subclass does not initialise
                `self.tracks` as a list of objects satisfying
                `TrackletProtocol` in `__init__`.
        """
        tracklets = [t for t in self.tracks if t.tracker_id != -1]
        xyxy = (
            np.array([t.get_state_bbox() for t in tracklets], dtype=np.float32)
            if tracklets
            else np.empty((0, 4), dtype=np.float32)
        )
        tracker_ids = np.array([t.tracker_id for t in tracklets], dtype=int)
        result = sv.Detections(xyxy=xyxy)
        result.tracker_id = tracker_ids
        return result
