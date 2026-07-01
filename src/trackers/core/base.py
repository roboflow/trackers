# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import inspect
import math
import re
import types
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, Union, cast, get_args, get_origin

import numpy as np
import supervision as sv

from trackers.utils.base_tracklet import BaseTracklet
from trackers.utils.predict_timing import PredictTiming


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
    maximum_time_without_update: float | None
    _next_track_id: int

    @staticmethod
    def _compute_maximum_frames_without_update(
        lost_track_buffer: int,
        frame_rate: float,
    ) -> int:
        """Scale positive lost-track buffers without changing explicit zero-buffer configs.

        Args:
            lost_track_buffer: Non-negative buffer length expressed in 30 FPS frames.
                Zero means no missed-frame grace period.
            frame_rate: Actual video frame rate in frames per second. Must be
                finite and strictly positive.

        Returns:
            Scaled maximum number of missed frames before a confirmed track expires.
            Returns zero when ``lost_track_buffer`` is zero; otherwise returns
            ``max(1, ceil(frame_rate / 30.0 * lost_track_buffer))`` to ensure at
            least one frame of grace for any positive buffer at any frame rate.

        Raises:
            ValueError: If ``lost_track_buffer`` is negative.
            ValueError: If ``frame_rate`` is not finite or is not strictly positive.
            ValueError: If the scaled product overflows to infinity for extreme inputs.

        Examples:
            >>> BaseTracker._compute_maximum_frames_without_update(30, 30.0)
            30
            >>> BaseTracker._compute_maximum_frames_without_update(30, 60.0)
            60
            >>> BaseTracker._compute_maximum_frames_without_update(3, 15.0)
            2
            >>> BaseTracker._compute_maximum_frames_without_update(0, 30.0)
            0
        """
        if lost_track_buffer < 0:
            raise ValueError("lost_track_buffer must be greater than or equal to 0")
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("frame_rate must be a finite positive value")
        if lost_track_buffer == 0:
            return 0
        scaled = frame_rate / 30.0 * lost_track_buffer
        if not math.isfinite(scaled):
            raise ValueError("Scaled lost_track_buffer overflows: frame_rate / 30.0 * lost_track_buffer must be finite")
        return max(1, math.ceil(scaled))

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

    _frame_rate: float = 1.0
    _last_timestamp: float | None = None

    def _init_timestamp_state(self, frame_rate: float) -> None:
        """Register reference FPS and reset timestamp bookkeeping.

        Call from ``__init__`` on all concrete trackers.

        Args:
            frame_rate: Reference frames per second for bootstrap elapsed time.
        """
        self._frame_rate = frame_rate
        self._last_timestamp = None

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

    def _predict_timing(self, timestamp: float | None) -> PredictTiming:
        """Build predict timing from an optional timestamp.

        All timestamp ordering checks live here: fixed-rate mode, bootstrap,
        backwards (skip whole update), duplicate (skip predict only), normal gap.
        ``_last_timestamp`` advances only on bootstrap and strictly increasing times.
        """
        if timestamp is None:
            self._last_timestamp = None
            return PredictTiming(frame_step=1.0, elapsed_seconds=None)

        if not np.isfinite(timestamp):
            warnings.warn(
                f"{type(self).__name__}: timestamp {timestamp!r} is not finite; skipping update.",
                UserWarning,
                stacklevel=3,
            )
            return PredictTiming(frame_step=0.0, elapsed_seconds=None, skip_update=True)

        last = self._last_timestamp

        if last is None:
            # Bootstrap: no prior timestamp, so we cannot compute t - t_prev.
            # Use one nominal frame period (1 / frame_rate) so the first Kalman
            # step is frame_step=1.0 — matching fixed-rate behaviour rather than
            # using the absolute timestamp value (e.g. 37.2 s would break tuning).
            self._last_timestamp = timestamp
            elapsed = 1.0 / self._frame_rate
            return PredictTiming(
                frame_step=elapsed * self._frame_rate,
                elapsed_seconds=elapsed,
            )

        if timestamp < last:
            warnings.warn(
                f"{type(self).__name__}: timestamp {timestamp} is earlier than the "
                f"previous timestamp {last}. Skipping update; pass capture times in "
                "non-decreasing order.",
                UserWarning,
                stacklevel=3,
            )
            return PredictTiming(frame_step=0.0, elapsed_seconds=None, skip_update=True)

        if timestamp == last:
            warnings.warn(
                f"{type(self).__name__}: duplicate timestamp {timestamp}; skipping predict for this step.",
                UserWarning,
                stacklevel=3,
            )
            return PredictTiming(frame_step=0.0, elapsed_seconds=0.0)

        elapsed = timestamp - last
        self._last_timestamp = timestamp
        return PredictTiming(
            frame_step=elapsed * self._frame_rate,
            elapsed_seconds=elapsed,
        )

    def _detections_for_skipped_update(self, detections: sv.Detections) -> sv.Detections:
        """Return detections unchanged except tracker_id=-1; do not mutate tracks."""
        if len(detections) == 0:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result
        result = cast(sv.Detections, detections[:])
        result.tracker_id = np.full(len(result), -1, dtype=int)
        return result

    def _predict_tracklets(self, tracklets: list[Any], timing: PredictTiming) -> None:
        """Predict all tracklets unless the timestamp did not advance."""
        if timing.skip_predict:
            return
        for tracklet in tracklets:
            tracklet.predict(timing)

    def _lost_track_time_budget(
        self,
        timing: PredictTiming,
        seconds_budget: float | None,
    ) -> float | None:
        """Return the seconds lost-track budget when timestamps are in use."""
        return seconds_budget if timing.uses_elapsed_time else None

    def _prune_lost_tracks(self, timing: PredictTiming) -> None:
        """Remove tracks that exceed their lost-track budget (ghost-ID prevention).

        Applies a budget-only filter so immature tracks stay alive for matching.
        Call after ``_predict_tracklets`` and before association.

        At fixed frame rate (no timestamps) this is a no-op — the frame-count
        budget is enforced post-association, preserving the last-frame re-association
        opportunity that the original trackers relied on.  In variable-FPS mode the
        time budget can differ from the frame budget, so expired-by-time tracks are
        removed here before they can be matched and revived with a stale ID.
        """
        budget = self._lost_track_time_budget(timing, self.maximum_time_without_update)
        if budget is None:
            return
        self.tracks = [
            t
            for t in self.tracks
            if BaseTracklet.within_lost_track_budget(
                t,
                maximum_frames_without_update=self.maximum_frames_without_update,
                maximum_time_without_update=budget,
            )
        ]

    @abstractmethod
    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        """Process new detections and assign track IDs.

        Matches incoming detections to existing tracks, creates new tracks
        for unmatched detections, and handles track lifecycle management.

        Args:
            detections: Current frame detections with xyxy, confidence, class_id.
            frame: Current video frame in BGR format (H, W, 3), or ``None``.
                Used by trackers with camera motion compensation (e.g. BoTSORT).
            timestamp: Absolute time of the current frame in seconds, or
                ``None`` for fixed-rate mode (``frame_step = 1.0`` per call).
                Must be non-negative. When provided, elapsed seconds are
                converted to Kalman frame units via ``* frame_rate``; pruning
                uses seconds directly. Must be non-decreasing in capture time.
                Passing ``None`` resets the internal timestamp anchor so the
                next timestamped call is treated as a fresh bootstrap.

        Returns:
            sv.Detections enriched with tracker_id assigned for each
            detection box. When the update is skipped (backwards or
            non-finite timestamp), all ``tracker_id`` values are ``-1``.

        Warns:
            UserWarning: If ``timestamp`` is earlier than the previous call
                (backwards order); the whole update is skipped and all output
                IDs are ``-1``. If ``timestamp`` equals the previous call
                (duplicate); predict is skipped but association still runs on
                the last state (``elapsed_seconds = 0.0``).

        Note:
            Mixing timestamped and non-timestamped calls in the same session is
            unsupported. Calling ``update(detections)`` (no timestamp) resets
            ``_last_timestamp`` to ``None``; the next timestamped call is then
            treated as a fresh bootstrap (``frame_step = 1 / frame_rate``) rather
            than measuring the real gap from the previous call. If you switch from
            ``update(d, timestamp=t)`` to ``update(d)`` and then back to
            ``update(d, timestamp=t2)``, the elapsed gap ``t2 - t`` is silently
            discarded and the Kalman step is reset to one nominal frame.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal tracking state.

        Call between videos or when tracking should restart from scratch.
        """
        pass

    def _reset_id_allocator(self) -> None:
        """Restart this tracker instance's ID allocation from zero."""
        self._next_track_id = 0

    def _allocate_tracker_id(self) -> int:
        """Return the next tracker ID (zero-indexed) and advance the internal counter."""
        next_track_id = self._next_track_id
        self._next_track_id = next_track_id + 1
        return next_track_id

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
