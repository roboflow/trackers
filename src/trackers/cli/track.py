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
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

from trackers import frames_from_source
from trackers.cli.progress import _classify_source, _SourceInfo, _TrackingProgress
from trackers.core.base import BaseTracker
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

COLOR_PALETTE = sv.ColorPalette.from_hex(
    [
        "#ffff00",
        "#ff9b00",
        "#ff8080",
        "#ff66b2",
        "#ff66ff",
        "#b266ff",
        "#9999ff",
        "#3399ff",
        "#66ffff",
        "#33ff99",
        "#66ff66",
        "#99ff00",
    ]
)


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


@dataclass
class TrackerOptions:
    """Optional tracker-specific parameters.

    Union of parameters across all registered trackers; each tracker only
    receives the keys it knows about. Fields left as ``None`` are dropped
    before instantiation so the tracker's own defaults apply.

    CLI names abbreviate the standard leading token only — ``minimum_`` becomes
    ``min_`` and ``maximum_`` becomes ``max_``. Domain words such as
    ``threshold`` stay spelled out. The Python keyword names are unchanged;
    :func:`_init_tracker` maps the short CLI name back to the long keyword.

    Attributes:
        name: Tracking algorithm ID. Discoverable via
            ``BaseTracker._registered_trackers()``. ``--tracker <id>`` is the
            shorthand spelling of ``--tracker.name <id>``.
        lost_track_buffer: Frames to keep a lost track before discarding.
        frame_rate: Source frame rate for time-based logic.
        track_activation_threshold: Detection score needed to spawn a track.
        min_consecutive_frames: Consecutive matches to confirm a track.
        min_iou_threshold: IoU threshold for SORT/OC-SORT association.
        min_iou_threshold_first_assoc: BoT-SORT first-stage IoU.
        min_iou_threshold_second_assoc: BoT-SORT second-stage IoU.
        min_iou_threshold_unconfirmed_assoc: BoT-SORT unconfirmed IoU.
        high_conf_det_threshold: High-confidence detection threshold.
        direction_consistency_weight: OC-SORT direction consistency weight.
        delta_t: OC-SORT velocity delta horizon.
        enable_cmc: BoT-SORT camera motion compensation toggle.
        cmc_method: BoT-SORT CMC method name.
        cmc_downscale: BoT-SORT CMC downscale factor.
        instant_first_frame_activation: BoT-SORT first-frame activation toggle.
        iou_variant: IoU similarity metric for data association. One of
            ``iou`` (standard), ``giou``, ``diou``, ``ciou``, ``biou``.
            Applies to all trackers. Defaults to ``iou``.
    """

    name: str = DEFAULT_TRACKER
    lost_track_buffer: int | None = None
    frame_rate: float | None = None
    track_activation_threshold: float | None = None
    min_consecutive_frames: int | None = None
    min_iou_threshold: float | None = None
    min_iou_threshold_first_assoc: float | None = None
    min_iou_threshold_second_assoc: float | None = None
    min_iou_threshold_unconfirmed_assoc: float | None = None
    high_conf_det_threshold: float | None = None
    direction_consistency_weight: float | None = None
    delta_t: int | None = None
    enable_cmc: bool | None = None
    cmc_method: str | None = None
    cmc_downscale: int | None = None
    instant_first_frame_activation: bool | None = None
    iou_variant: str | None = None


# Standard leading tokens abbreviated on the CLI, mapping long Python prefix to
# short CLI prefix. Domain words (``threshold``, ``consecutive``) stay in full.
_CLI_PARAMETER_ABBREVIATIONS = {"minimum_": "min_", "maximum_": "max_"}


def _abbreviate_parameter_name(name: str) -> str:
    """Return the CLI spelling of one tracker ``__init__`` parameter name.

    Args:
        name: Python keyword name (e.g. ``minimum_iou_threshold``).

    Returns:
        Abbreviated CLI name, or ``name`` unchanged when no prefix applies.

    Examples:
        >>> _abbreviate_parameter_name("minimum_iou_threshold")
        'min_iou_threshold'
        >>> _abbreviate_parameter_name("lost_track_buffer")
        'lost_track_buffer'
    """
    for long_prefix, short_prefix in _CLI_PARAMETER_ABBREVIATIONS.items():
        if name.startswith(long_prefix):
            return f"{short_prefix}{name.removeprefix(long_prefix)}"
    return name


def _expand_parameter_name(name: str) -> str:
    """Return the tracker ``__init__`` keyword name for one CLI parameter name.

    Inverse of :func:`_abbreviate_parameter_name`.

    Args:
        name: Abbreviated CLI name (e.g. ``min_iou_threshold``).

    Returns:
        Python keyword name, or ``name`` unchanged when no prefix applies.

    Examples:
        >>> _expand_parameter_name("min_iou_threshold")
        'minimum_iou_threshold'
        >>> _expand_parameter_name("lost_track_buffer")
        'lost_track_buffer'
    """
    for long_prefix, short_prefix in _CLI_PARAMETER_ABBREVIATIONS.items():
        if name.startswith(short_prefix):
            return f"{long_prefix}{name.removeprefix(short_prefix)}"
    return name


def _abbreviated_tracker_parameters() -> dict[str, str]:
    """Map every abbreviated :class:`TrackerOptions` field to its former CLI name.

    Derived from the dataclass fields so the deprecation aliases in the CLI
    entry point cannot drift from the option definitions.

    Returns:
        Mapping of long (deprecated) name to short (current) name.

    Examples:
        >>> _abbreviated_tracker_parameters()["minimum_iou_threshold"]
        'min_iou_threshold'
    """
    renamed = {}
    for field in fields(TrackerOptions):
        expanded = _expand_parameter_name(field.name)
        if expanded != field.name:
            renamed[expanded] = field.name
    return renamed


def track_command(
    source: str | None = None,
    detection: DetectionOptions | None = None,
    filters: FilterOptions | None = None,
    tracker: TrackerOptions | None = None,
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

    if output.video:
        _validate_output_path(_resolve_video_output_path(output.video), overwrite=output.overwrite)
    if output.mot_results:
        _validate_output_path(output.mot_results, overwrite=output.overwrite)

    # Built before the detection model so an unknown tracker ID is rejected
    # without first paying for a model download and load.
    try:
        tracker_obj = _init_tracker(tracker)
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


def _init_tracker(params: TrackerOptions | None) -> BaseTracker:
    """Create a tracker instance from the registry.

    ``params.name`` selects the algorithm; every other field is a parameter
    override. Only fields the chosen tracker accepts are forwarded; ``None``
    values are always dropped so the tracker's own defaults apply.

    Abbreviated CLI names are resolved back to their tracker ``__init__``
    keyword before the forwarding check, so ``min_iou_threshold`` reaches the
    tracker as ``minimum_iou_threshold``. Without that step a renamed CLI
    option would silently fail the membership test and leave the tracker on its
    own default. ``iou_variant`` is the same kind of alias for ``iou``.

    Args:
        params: Tracker selection and parameter overrides.

    Returns:
        Initialised tracker instance.

    Raises:
        ValueError: If ``params.name`` is not registered.
    """
    raw = asdict(params) if params is not None else {}
    tracker_id = raw.pop("name", DEFAULT_TRACKER)
    info = BaseTracker._lookup_tracker(tracker_id)
    if info is None:
        available = ", ".join(BaseTracker._registered_trackers())
        raise ValueError(f"Unknown tracker: '{tracker_id}'. Available: {available}")

    iou_variant = raw.pop("iou_variant", None)
    accepted = set(info.parameters)
    kwargs = {}
    for name, value in raw.items():
        if value is None:
            continue
        keyword = name if name in accepted else _expand_parameter_name(name)
        if keyword in accepted:
            kwargs[keyword] = value
    if iou_variant is not None:
        if "iou" in accepted:
            kwargs["iou"] = variant_from_name(iou_variant)
        else:
            warnings.warn(
                f"Tracker '{tracker_id}' does not support iou_variant; '{iou_variant}' will be ignored.",
                UserWarning,
                stacklevel=2,
            )
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
