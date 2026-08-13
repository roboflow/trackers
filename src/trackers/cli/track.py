#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""``trackers track`` subcommand — run a detector + tracker over a video source."""

from __future__ import annotations

import sys
import warnings
from contextlib import nullcontext, suppress
from dataclasses import dataclass, fields, make_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_type_hints

import numpy as np
import supervision as sv

from trackers import frames_from_source
from trackers.cli._annotate import COLOR_PALETTE as _COLOR_PALETTE
from trackers.cli._progress import _classify_source, _SourceInfo, _TrackingProgress
from trackers.core.base import BaseTracker, ParameterInfo
from trackers.io.mot import _mot_frame_to_detections, _MOTOutput, load_mot_file
from trackers.io.paths import _resolve_video_output_path, _validate_output_path
from trackers.io.video import _DEFAULT_OUTPUT_FPS, _DisplayWindow, _VideoOutput
from trackers.utils.device import _best_device
from trackers.utils.iou import variant_from_name

if TYPE_CHECKING:
    from inference_models import AnyModel

DEFAULT_MODEL = "rfdetr-nano"
DEFAULT_TRACKER = "bytetrack"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_DEVICE = "auto"

# Re-exported: the palette moved to _annotate so that inspect draws tracklets in
# the same colours, and track's own callers keep importing it from here.
COLOR_PALETTE = _COLOR_PALETTE


@dataclass
class DetectionOptions:
    """Detection model and inference settings.

    Attributes:
        model: Model ID (e.g. ``rfdetr-nano``) or
            ``workspace/project/version`` for a Roboflow custom model.
            Ignored when ``mot_file`` is set.
        mot_file: Path to a pre-computed MOT-format detector-output file.
            Mutually exclusive with ``model``; supply one or the other.
        confidence: Detection confidence threshold.
        device: Inference device: ``auto``, ``cpu``, ``cuda``, ``cuda:0``,
            ``mps``.
        api_key: Roboflow API key (required for private custom models).
    """

    model: str = DEFAULT_MODEL
    mot_file: Path | None = None
    confidence: float = DEFAULT_CONFIDENCE
    device: str = DEFAULT_DEVICE
    api_key: str | None = None


@dataclass
class ReIDOptions:
    """Appearance ReID settings, for trackers that accept an encoder.

    Requires the ``reid`` extra (``pip install "trackers[reid]"``). BoT-SORT is
    the only tracker that accepts an encoder today.

    Attributes:
        enable: Enable appearance association using the default encoder.
        model: Checkpoint source: curated alias, ``hf://`` URL, local path, or
            ``save_pretrained`` directory. Implies ``enable``. Default alias
            when omitted: ``osnet_x1_0_msmt17_combineall``.
        device: ReID compute device: ``auto``, ``cpu``, ``cuda``, ``mps``.
        architecture: Backbone for bare ``.pth``/``.safetensors`` weights (e.g.
            ``osnet_x1_0``, ``fastreid_sbs_resnest50``). Required when ``model``
            points at a bare weights file.
    """

    enable: bool = False
    model: str | None = None
    device: str = DEFAULT_DEVICE
    architecture: str | None = None


@dataclass
class FilterOptions:
    """Detection and track filters.

    Both fields are lists, matching the list-valued options of ``eval`` and
    ``tune``. On the command line they accept bracket shorthand, so
    ``--filters.classes=[person,car]`` and ``--filters.track_ids=[1,3,5]``
    need no quoting.

    Attributes:
        classes: Class names or IDs to keep (e.g. ``[person,car]``, ``[0,2]``,
            or the mixed form ``[person,2]``).
        track_ids: Track IDs to keep in the output (e.g. ``[1,3,5]``).
    """

    classes: list[str | int] | None = None
    track_ids: list[str | int] | None = None


@dataclass
class OutputOptions:
    """Output paths and write policy.

    Attributes:
        video: Annotated-video output path.
        mot_results: MOT-format predictions output path.
        overwrite: Overwrite existing output files without prompting.
    """

    video: Path | None = None
    mot_results: Path | None = None
    overwrite: bool = False


@dataclass
class ShowOptions:
    """Annotation elements to draw on each frame.

    Attributes:
        boxes: Draw bounding boxes around detections.
        masks: Draw segmentation masks (segmentation models only).
        labels: Draw class labels.
        ids: Draw track IDs.
        confidence: Draw detection confidence scores.
        trajectories: Draw track trajectory trails.
    """

    boxes: bool = True
    masks: bool = False
    labels: bool = False
    ids: bool = True
    confidence: bool = False
    trajectories: bool = False


# Standard leading tokens abbreviated on the CLI, mapping long Python prefix to
# short CLI prefix. Domain words (``threshold``, ``consecutive``) stay in full.
_CLI_PARAMETER_ABBREVIATIONS = {"minimum_": "min_", "maximum_": "max_"}

# Exact-name CLI renames, checked before the prefix rule above. ``mask_config``
# never shipped under its Python name on any CLI release, so unlike the
# prefixes it carries no deprecation alias — see `_abbreviated_tracker_parameters`.
_CLI_PARAMETER_RENAMES = {"mask_config": "mask"}
_CLI_PARAMETER_RENAMES_REVERSED = {short: long for long, short in _CLI_PARAMETER_RENAMES.items()}

# Tracker registry parameters that never reach the CLI. Both take a live object
# rather than a value a command line can express: ``mask_manager`` a
# ``MaskManager``, whose CLI-facing switch is ``enable_mask_manager``, and
# ``reid_model`` an encoder, built from the ``ReIDOptions`` group instead.
_EXCLUDED_TRACKER_PARAMETERS = frozenset({"mask_manager", "reid_model"})


def _abbreviate_parameter_name(name: str) -> str:
    """Return the CLI spelling of one tracker ``__init__`` parameter name.

    Args:
        name: Python keyword name (e.g. ``minimum_iou_threshold``).

    Returns:
        Renamed or abbreviated CLI name, or ``name`` unchanged when neither
        rule applies.

    Examples:
        >>> _abbreviate_parameter_name("minimum_iou_threshold")
        'min_iou_threshold'
        >>> _abbreviate_parameter_name("mask_config")
        'mask'
        >>> _abbreviate_parameter_name("lost_track_buffer")
        'lost_track_buffer'
    """
    if name in _CLI_PARAMETER_RENAMES:
        return _CLI_PARAMETER_RENAMES[name]
    for long_prefix, short_prefix in _CLI_PARAMETER_ABBREVIATIONS.items():
        if name.startswith(long_prefix):
            return f"{short_prefix}{name.removeprefix(long_prefix)}"
    return name


def _expand_parameter_name(name: str) -> str:
    """Return the tracker ``__init__`` keyword name for one CLI parameter name.

    Inverse of :func:`_abbreviate_parameter_name`.

    Args:
        name: Abbreviated or renamed CLI name (e.g. ``min_iou_threshold``).

    Returns:
        Python keyword name, or ``name`` unchanged when neither rule applies.

    Examples:
        >>> _expand_parameter_name("min_iou_threshold")
        'minimum_iou_threshold'
        >>> _expand_parameter_name("mask")
        'mask_config'
        >>> _expand_parameter_name("lost_track_buffer")
        'lost_track_buffer'
    """
    if name in _CLI_PARAMETER_RENAMES_REVERSED:
        return _CLI_PARAMETER_RENAMES_REVERSED[name]
    for long_prefix, short_prefix in _CLI_PARAMETER_ABBREVIATIONS.items():
        if name.startswith(short_prefix):
            return f"{long_prefix}{name.removeprefix(short_prefix)}"
    return name


def _tracker_parameter_union() -> list[tuple[str, ParameterInfo, Any]]:
    """Collect every CLI-eligible parameter across the tracker registry.

    Trackers are visited in :func:`BaseTracker._registered_trackers` order,
    each tracker's own parameters in ``__init__`` order. The first tracker to
    define a given name provides its description and annotation, so a
    parameter shared by several trackers (e.g. ``lost_track_buffer``) is not
    duplicated.

    Returns:
        ``(python_name, parameter_info, annotation)`` triples. The annotation
        comes from :func:`typing.get_type_hints` rather than
        ``parameter_info.param_type``, which is normalized for argparse and
        would collapse ``type[BaseStateEstimator]`` to ``abc.ABCMeta`` and a
        ``Literal`` to ``str``.
    """
    seen: dict[str, tuple[ParameterInfo, Any]] = {}
    for tracker_id in BaseTracker._registered_trackers():
        info = BaseTracker._lookup_tracker(tracker_id)
        if info is None:
            continue
        try:
            hints = get_type_hints(info.tracker_class.__init__)
        except Exception:
            hints = {}
        for name, parameter_info in info.parameters.items():
            if name in _EXCLUDED_TRACKER_PARAMETERS or name in seen:
                continue
            seen[name] = (parameter_info, hints.get(name, parameter_info.param_type))
    return [(name, parameter_info, annotation) for name, (parameter_info, annotation) in seen.items()]


def _build_tracker_options() -> type:
    """Generate the ``TrackerOptions`` dataclass from the tracker registry.

    Union of parameters across all registered trackers; each tracker only
    receives the keys it knows about (:func:`_init_tracker` drops the rest).
    Every field defaults to ``None`` so an unset option leaves the tracker's
    own default alone. CLI names abbreviate the standard leading token only —
    ``minimum_`` becomes ``min_`` and ``maximum_`` becomes ``max_`` — plus the
    one exact-name rename, ``mask_config`` to ``mask``; the Python keyword
    names are unchanged, and :func:`_init_tracker` maps a short CLI name back
    to its long keyword.

    Generating this from the registry means a newly registered tracker
    parameter is reachable from the CLI without a manual edit here.

    Returns:
        A dataclass equivalent to a hand-written ``@dataclass class
        TrackerOptions``, with a per-field ``Attributes:`` docstring section
        jsonargparse reads for ``--help`` text.
    """
    field_specs: list[tuple[str, Any, Any]] = [("name", str, DEFAULT_TRACKER)]
    doc_lines = [
        "Optional tracker-specific parameters.",
        "",
        "Union of parameters across all registered trackers; each tracker only",
        "receives the keys it knows about. Fields left as ``None`` are dropped",
        "before instantiation so the tracker's own defaults apply. Generated",
        "from the tracker registry, so a newly registered tracker parameter is",
        "reachable from the CLI without a manual edit here.",
        "",
        "CLI names abbreviate the standard leading token only — ``minimum_``",
        "becomes ``min_`` and ``maximum_`` becomes ``max_`` — plus one",
        "exact-name rename, ``mask_config`` to ``mask``. The Python keyword",
        "names are unchanged; :func:`_init_tracker` maps the short CLI name",
        "back to the long keyword.",
        "",
        "Attributes:",
        "    name: Tracking algorithm ID. Discoverable via",
        "        ``BaseTracker._registered_trackers()``. ``--tracker <id>`` is the",
        "        shorthand spelling of ``--tracker.name <id>``.",
    ]
    for python_name, parameter_info, annotation in _tracker_parameter_union():
        cli_name = _abbreviate_parameter_name(python_name)
        # ``annotation | None`` rather than ``typing.Optional[annotation]``: a
        # generic alias such as ``type[BaseStateEstimator]`` supports ``|``
        # directly, and an annotation already optional (``McByteMaskConfig |
        # None``) collapses back to itself instead of double-wrapping.
        field_specs.append((cli_name, annotation | None, None))
        doc_lines.append(f"    {cli_name}: {parameter_info.description}")
    field_specs.append(("iou_variant", str | None, None))
    doc_lines.append(
        "    iou_variant: IoU similarity metric for data association. One of "
        "``iou`` (standard), ``giou``, ``diou``, ``ciou``, ``biou``. Applies "
        "to all trackers. Defaults to ``iou``."
    )

    cls = make_dataclass("TrackerOptions", field_specs)
    cls.__module__ = __name__
    cls.__doc__ = "\n".join(doc_lines)
    return cls


# Not guarded by ``if TYPE_CHECKING`` — jsonargparse's own parameter resolver
# walks this module's source for ``track_command`` and treats a
# ``TYPE_CHECKING`` branch as taken the same way a type checker does, which
# would make it resolve ``tracker`` as ``Any`` and silently stop validating or
# instantiating ``--config`` values for that group. A single unconditional
# assignment keeps that resolution correct; the `# type: ignore[valid-type]`
# comments where ``TrackerOptions`` is used as an annotation cover the mypy
# cost instead (see :func:`_build_tracker_options`).
TrackerOptions = _build_tracker_options()


def _abbreviated_tracker_parameters() -> dict[str, str]:
    """Map every abbreviated :class:`TrackerOptions` field to its former CLI name.

    Derived from the dataclass fields so the deprecation aliases in the CLI
    entry point cannot drift from the option definitions. The exact-name
    ``mask`` rename is excluded: ``mask_config`` never shipped under its
    Python name on any CLI release, so warning that it is deprecated would be
    a lie.

    Returns:
        Mapping of long (deprecated) name to short (current) name.

    Examples:
        >>> _abbreviated_tracker_parameters()["minimum_iou_threshold"]
        'min_iou_threshold'
    """
    renamed = {}
    for field in fields(TrackerOptions):
        if field.name in _CLI_PARAMETER_RENAMES_REVERSED:
            continue
        expanded = _expand_parameter_name(field.name)
        if expanded != field.name:
            renamed[expanded] = field.name
    return renamed


def track_command(
    source: str | None = None,
    detection: DetectionOptions | None = None,
    filters: FilterOptions | None = None,
    tracker: TrackerOptions | None = None,  # type: ignore[valid-type]
    reid: ReIDOptions | None = None,
    output: OutputOptions | None = None,
    display: bool = False,
    show: ShowOptions | None = None,
) -> int:
    """Run detection and tracking over a video, webcam, RTSP, or image directory.

    Args:
        source: Video file, webcam index (e.g. ``"0"``), RTSP URL, or image
            directory. Required unless ``detection.mot_file`` is supplied.
        detection: Detection model and inference options.
        filters: Class and track-ID filters applied to detections and tracks.
        tracker: Algorithm ID plus optional parameter overrides; only fields
            matching the chosen tracker's ``__init__`` are forwarded.
        reid: Appearance ReID encoder options. Requires ``source``, since
            embeddings are extracted from frames.
        output: Output paths.
        display: Show a live preview window during tracking.
        show: Annotation elements to draw on each frame.

    Returns:
        Exit code: ``0`` on success, ``1`` on validation error.
    """
    if detection is None:
        detection = DetectionOptions()
    if filters is None:
        filters = FilterOptions()
    if tracker is None:
        tracker = TrackerOptions()
    if reid is None:
        reid = ReIDOptions()
    if output is None:
        output = OutputOptions()
    if show is None:
        show = ShowOptions()
    needs_frames = output.video is not None or display

    if source is None and detection.mot_file is None:
        print("Error: --source is required when not using --detection.mot_file.", file=sys.stderr)
        return 1
    if needs_frames and source is None:
        print("Error: --source is required when using --output.video or --display.", file=sys.stderr)
        return 1
    if _reid_requested(reid) and source is None:
        print(
            "Error: ReID requires --source (video/webcam/images) so appearance "
            "embeddings can be extracted from frames.",
            file=sys.stderr,
        )
        return 1

    if output.video:
        _validate_output_path(_resolve_video_output_path(output.video), overwrite=output.overwrite)
    if output.mot_results:
        _validate_output_path(output.mot_results, overwrite=output.overwrite)

    # Built before the detection model so an unknown tracker ID is rejected
    # without first paying for a model download and load.
    try:
        tracker_obj = _init_tracker(tracker, reid)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if detection.mot_file is not None:
        model_obj: AnyModel | None = None
        detections_data: dict | None = load_mot_file(detection.mot_file)
        class_names: list[str] = []
    else:
        model_obj = _init_model(detection.model, device=detection.device, api_key=detection.api_key)
        detections_data = None
        class_names = getattr(model_obj, "class_names", [])

    class_filter = _resolve_class_filter(filters.classes, class_names)
    track_id_filter = _resolve_track_id_filter(filters.track_ids)

    if source is not None:
        return _run_with_source(
            source=source,
            model=model_obj,
            confidence=detection.confidence,
            detections_data=detections_data,
            class_names=class_names,
            class_filter=class_filter,
            track_id_filter=track_id_filter,
            tracker=tracker_obj,
            output=output.video,
            mot_results=output.mot_results,
            display=display,
            show=show,
        )

    return _run_frameless(
        detections_data=detections_data,
        class_filter=class_filter,
        track_id_filter=track_id_filter,
        tracker=tracker_obj,
        mot_results=output.mot_results,
    )


def _run_frameless(
    *,
    detections_data: dict | None,
    class_filter: list[int] | None,
    track_id_filter: list[int] | None,
    tracker: BaseTracker,
    mot_results: Path | None,
) -> int:
    """Run tracking from pre-computed detections without a frame source."""
    if not detections_data:
        print("Error: No detections found in file.", file=sys.stderr)
        return 1

    total_frames = max(detections_data.keys())
    source_info = _SourceInfo(source_type="video", total_frames=total_frames)

    with suppress(KeyboardInterrupt):
        with _MOTOutput(mot_results) as mot, _TrackingProgress(source_info) as progress:
            for frame_idx in range(1, total_frames + 1):
                if frame_idx in detections_data:
                    dets = _mot_frame_to_detections(detections_data[frame_idx])
                else:
                    dets = sv.Detections.empty()

                if class_filter is not None and len(dets) > 0 and dets.class_id is not None:
                    mask = np.isin(dets.class_id, class_filter)
                    dets = dets[mask]  # type: ignore[assignment]

                tracked = tracker.update(dets)

                if track_id_filter is not None and len(tracked) > 0 and tracked.tracker_id is not None:
                    mask = np.isin(tracked.tracker_id.astype(int), track_id_filter)
                    tracked = tracked[mask]  # type: ignore[assignment]

                mot.write(frame_idx, tracked)
                progress.update()

            progress.complete(interrupted=False)

    return 0


def _run_with_source(
    *,
    source: str,
    model: AnyModel | None,
    confidence: float,
    detections_data: dict | None,
    class_names: list[str],
    class_filter: list[int] | None,
    track_id_filter: list[int] | None,
    tracker: BaseTracker,
    output: Path | None,
    mot_results: Path | None,
    display: bool,
    show: ShowOptions,
) -> int:
    """Run tracking with a frame source (video, webcam, images)."""
    frame_gen = frames_from_source(source)
    source_info = _classify_source(source)

    annotators, label_annotator = _init_annotators(show)
    trace_annotator = (
        sv.TraceAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.TRACK) if show.trajectories else None
    )
    display_ctx = _DisplayWindow() if display else nullcontext()

    with suppress(KeyboardInterrupt):
        with (
            _VideoOutput(output, fps=source_info.fps or _DEFAULT_OUTPUT_FPS) as video,
            _MOTOutput(mot_results) as mot,
            display_ctx as display_win,
            _TrackingProgress(source_info) as progress,
        ):
            interrupted = False
            for frame_idx, frame in frame_gen:
                if model is not None:
                    dets = _run_model(model, frame, confidence)
                elif detections_data is not None and frame_idx in detections_data:
                    dets = _mot_frame_to_detections(detections_data[frame_idx])
                else:
                    dets = sv.Detections.empty()

                if class_filter is not None and len(dets) > 0 and dets.class_id is not None:
                    mask = np.isin(dets.class_id, class_filter)
                    dets = dets[mask]  # type: ignore[assignment]

                tracked = tracker.update(dets, frame)

                if track_id_filter is not None and len(tracked) > 0 and tracked.tracker_id is not None:
                    mask = np.isin(tracked.tracker_id.astype(int), track_id_filter)
                    tracked = tracked[mask]  # type: ignore[assignment]

                mot.write(frame_idx, tracked)
                progress.update()

                if display or output:
                    annotated = frame.copy()
                    if trace_annotator is not None:
                        annotated = trace_annotator.annotate(annotated, tracked)
                    for ann in annotators:
                        annotated = ann.annotate(annotated, tracked)
                    if label_annotator is not None:
                        labeled = tracked[tracked.tracker_id != -1]
                        labels = _format_labels(labeled, class_names, show)
                        annotated = label_annotator.annotate(annotated, labeled, labels)

                    video.write(annotated)

                    if display_win is not None:
                        display_win.show(annotated)
                        if display_win.quit_requested:
                            interrupted = True
                            break

            progress.complete(interrupted=interrupted)

    return 0


def _resolve_track_id_filter(track_ids_arg: list[str | int] | None) -> list[int] | None:
    """Resolve parsed ``track_ids`` tokens to a list of integer IDs.

    Tokens arrive already split by jsonargparse, so a numeric token is an
    ``int`` while a malformed one stays a ``str``. Malformed tokens are printed
    as warnings and skipped rather than aborting the run.

    Args:
        track_ids_arg: Parsed ``--filters.track_ids`` tokens (e.g. ``[1, 3, 5]``).
            ``None`` or empty means no filter.

    Returns:
        List of integer track IDs, or ``None`` when no valid filter remains.
    """
    if not track_ids_arg:
        return None

    track_ids: list[int] = []
    for raw in track_ids_arg:
        token = str(raw).strip()
        try:
            track_ids.append(int(token))
        except ValueError:
            print(f"Warning: '{token}' is not a valid track ID, skipping.", file=sys.stderr)
    return track_ids or None


def _resolve_class_filter(classes_arg: list[str | int] | None, class_names: list[str]) -> list[int] | None:
    """Resolve parsed ``classes`` tokens to a list of integer IDs.

    Each token is checked independently: if it parses as an ``int`` it is used
    directly as a class ID; otherwise it is looked up by name in ``class_names``.
    Unknown names are printed as warnings and skipped. Names and IDs may be
    mixed in one filter, which is why tokens stay loosely typed.

    Args:
        classes_arg: Parsed ``--filters.classes`` tokens (e.g. ``["person", "car"]``,
            ``[0, 2]``, or the mixed ``["person", 2]``). ``None`` or empty means
            no filter.
        class_names: Ordered list of class names where the index equals the
            class ID (as provided by the model).

    Returns:
        List of integer class IDs, or ``None`` when no valid filter remains.
    """
    if not classes_arg:
        return None

    name_to_id = {name: i for i, name in enumerate(class_names)}
    class_filter: list[int] = []
    for raw in classes_arg:
        token = str(raw).strip()
        try:
            class_filter.append(int(token))
        except ValueError:
            if token in name_to_id:
                class_filter.append(name_to_id[token])
            else:
                print(f"Warning: class '{token}' not found in model class list, skipping.", file=sys.stderr)
    return class_filter or None


def _init_model(model_id: str, *, device: str = DEFAULT_DEVICE, api_key: str | None = None) -> AnyModel:
    """Load detection model via ``inference-models``.

    Args:
        model_id: Model identifier (e.g. ``rfdetr-nano`` or
            ``workspace/project/version``).
        device: Device to load model on (``auto``, ``cpu``, ``cuda``, ``mps``).
        api_key: Roboflow API key for custom models.

    Returns:
        Loaded model instance.
    """
    try:
        from inference_models import AutoModel
    except ImportError as e:
        print(
            "Error: inference-models is required for model-based detection.\n"
            "Install with: pip install 'trackers[detection]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    resolved_device = _best_device() if device == DEFAULT_DEVICE else device
    return AutoModel.from_pretrained(model_id, api_key=api_key, device=resolved_device)


def _run_model(model: AnyModel, frame: np.ndarray, confidence: float) -> sv.Detections:
    """Run model inference, filter by confidence, return ``sv.Detections``."""
    predictions = model(frame)
    if not predictions:
        return sv.Detections.empty()

    dets = predictions[0].to_supervision()
    if len(dets) > 0 and dets.confidence is not None:
        dets = dets[dets.confidence >= confidence]
    return dets


def _tracker_options_as_dict(params: TrackerOptions | None) -> dict[str, Any]:  # type: ignore[valid-type]
    """Return ``params`` as a shallow field-name-to-value mapping.

    Deliberately not :func:`dataclasses.asdict`: that recurses into nested
    dataclass fields such as ``mask``, handing the tracker a plain ``dict``
    instead of the ``McByteMaskConfig`` instance it expects.

    Args:
        params: Tracker selection and parameter overrides, or ``None``.

    Returns:
        One entry per field, values unconverted.
    """
    if params is None:
        return {}
    return {f.name: getattr(params, f.name) for f in fields(params)}


def _resolve_tracker_kwargs(raw: dict[str, Any], accepted: set[str]) -> tuple[dict[str, Any], list[str]]:
    """Split parsed CLI overrides into tracker kwargs and unsupported names.

    Abbreviated CLI names are resolved back to their tracker ``__init__``
    keyword before the membership check, so ``min_iou_threshold`` reaches the
    tracker as ``minimum_iou_threshold`` and ``mask`` as ``mask_config``.
    Without that step a renamed CLI option would silently fail the membership
    test and leave the tracker on its own default.

    Args:
        raw: Parsed ``TrackerOptions`` fields, ``name`` and ``iou_variant``
            already removed.
        accepted: Parameter names the selected tracker's ``__init__`` accepts.

    Returns:
        ``(kwargs, dropped)`` — forwarded keyword arguments, and the CLI names
        of overrides the selected tracker does not accept.
    """
    kwargs: dict[str, Any] = {}
    dropped: list[str] = []
    for name, value in raw.items():
        if value is None:
            continue
        keyword = name if name in accepted else _expand_parameter_name(name)
        if keyword in accepted:
            kwargs[keyword] = value
        else:
            dropped.append(name)
    return kwargs, dropped


def _warn_dropped_tracker_overrides(tracker_id: str, dropped: list[str]) -> None:
    """Warn once per CLI override the selected tracker silently ignored.

    Every :class:`TrackerOptions` field defaults to ``None``, so a non-``None``
    value absent from the tracker's accepted parameters means the user set an
    option this tracker does not support — most consequently the mcbyte
    mask settings, where a dropped override reads as "masks enabled" while the
    tracker silently ran without them.

    Args:
        tracker_id: Selected tracker's registry ID.
        dropped: CLI names of overrides the tracker does not accept.
    """
    for name in dropped:
        warnings.warn(
            f"Tracker '{tracker_id}' does not support --tracker.{name}; it will be ignored.",
            UserWarning,
            stacklevel=3,
        )


def _reid_requested(reid: ReIDOptions) -> bool:
    """Whether the command line asked for appearance association."""
    return reid.enable or reid.model is not None


def _load_reid_model(reid: ReIDOptions) -> Any:
    """Build the ReID encoder described by ``reid``.

    Args:
        reid: Encoder selection and loading options.

    Returns:
        A ``reid.ReIDModel`` satisfying the ``ReIDEncoder`` protocol.

    Raises:
        ValueError: If the ``reid`` extra is not installed, ``architecture`` was
            given without ``model``, or the checkpoint fails to load.
    """
    try:
        from reid import ReIDModel
    except ImportError as exc:
        raise ValueError(
            "ReID tracking requires the optional `trackers[reid]` extra.\nInstall with: pip install 'trackers[reid]'"
        ) from exc

    if reid.architecture is not None and reid.model is None:
        raise ValueError("--reid.architecture requires --reid.model (bare weights need a checkpoint path).")

    load_kwargs: dict[str, Any] = {"device": reid.device}
    if reid.model is not None:
        load_kwargs["source"] = reid.model
    if reid.architecture is not None:
        load_kwargs["architecture"] = reid.architecture

    try:
        return ReIDModel.from_pretrained(**load_kwargs)
    except KeyboardInterrupt:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Failed to load ReID model: {exc}") from exc


def _init_tracker(
    params: TrackerOptions | None,  # type: ignore[valid-type]
    reid: ReIDOptions | None = None,
) -> BaseTracker:
    """Create a tracker instance from the registry.

    ``params.name`` selects the algorithm; every other field is a parameter
    override. Only fields the chosen tracker accepts are forwarded; ``None``
    values are always dropped so the tracker's own defaults apply. A non-
    ``None`` value the chosen tracker does not accept is dropped with a
    warning rather than silently, via :func:`_warn_dropped_tracker_overrides`.

    Args:
        params: Tracker selection and parameter overrides.
        reid: Appearance options. When ReID is requested, the encoder is built
            here and injected as the ``reid_model`` keyword, mirroring how
            ``iou_variant`` becomes the ``iou`` keyword. Loading happens after
            the tracker ID resolves, so a bad ID fails before a checkpoint
            download.

    Returns:
        Initialised tracker instance.

    Raises:
        ValueError: If ``params.name`` is not registered, the chosen tracker
            does not accept an encoder, or the encoder fails to load.
    """
    raw = _tracker_options_as_dict(params)
    tracker_id = raw.pop("name", DEFAULT_TRACKER)
    info = BaseTracker._lookup_tracker(tracker_id)
    if info is None:
        available = ", ".join(BaseTracker._registered_trackers())
        raise ValueError(f"Unknown tracker: '{tracker_id}'. Available: {available}")

    iou_variant = raw.pop("iou_variant", None)
    accepted = set(info.parameters)
    kwargs, dropped = _resolve_tracker_kwargs(raw, accepted)
    _warn_dropped_tracker_overrides(tracker_id, dropped)
    if iou_variant is not None:
        if "iou" in accepted:
            kwargs["iou"] = variant_from_name(iou_variant)
        else:
            warnings.warn(
                f"Tracker '{tracker_id}' does not support iou_variant; '{iou_variant}' will be ignored.",
                UserWarning,
                stacklevel=2,
            )
    if reid is not None and _reid_requested(reid):
        if "reid_model" not in accepted:
            raise ValueError(f"--reid.* options apply only to a tracker that accepts an encoder, got '{tracker_id}'.")
        kwargs["reid_model"] = _load_reid_model(reid)
    return info.tracker_class(**kwargs)


def _init_annotators(show: ShowOptions) -> tuple[list, sv.LabelAnnotator | None]:
    """Initialise supervision annotators based on display options.

    Args:
        show: Annotation elements to draw on each frame.

    Returns:
        Tuple of (annotators list, label_annotator or None). Label annotator is
        separate because it needs custom labels per frame.

    Examples:
        >>> annotators, label_annotator = _init_annotators(ShowOptions(boxes=False, ids=False))
        >>> annotators, label_annotator
        ([], None)
    """
    annotators: list = []
    label_annotator: sv.LabelAnnotator | None = None

    if show.boxes:
        annotators.append(sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.TRACK))
    if show.masks:
        annotators.append(sv.MaskAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.TRACK))
    if show.labels or show.ids or show.confidence:
        label_annotator = sv.LabelAnnotator(
            color=COLOR_PALETTE,
            text_color=sv.Color.BLACK,
            text_position=sv.Position.TOP_LEFT,
            color_lookup=sv.ColorLookup.TRACK,
        )
    return annotators, label_annotator


def _format_labels(detections: sv.Detections, class_names: list[str], show: ShowOptions) -> list[str]:
    """Generate label strings for each detection.

    Args:
        detections: Detections to generate labels for.
        class_names: List of class names for lookup.
        show: Annotation elements to draw on each frame.

    Returns:
        List of label strings, one per detection.

    Examples:
        >>> import supervision as sv
        >>> _format_labels(sv.Detections.empty(), [], ShowOptions())
        []
    """
    labels = []
    for i in range(len(detections)):
        parts: list[str] = []
        if show.ids and detections.tracker_id is not None:
            parts.append(f"#{int(detections.tracker_id[i])}")
        if show.labels and detections.class_id is not None:
            class_id = int(detections.class_id[i])
            if class_names and 0 <= class_id < len(class_names):
                parts.append(class_names[class_id])
            else:
                parts.append(str(class_id))
        if show.confidence and detections.confidence is not None:
            parts.append(f"{detections.confidence[i]:.2f}")
        labels.append(" ".join(parts))
    return labels
