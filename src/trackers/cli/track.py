#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""``trackers track`` subcommand — run a detector + tracker over a video source."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
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

if TYPE_CHECKING:
    from inference_models import AnyModel

# Defaults
DEFAULT_MODEL = "rfdetr-nano"
DEFAULT_TRACKER = "bytetrack"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_DEVICE = "auto"

# Visualization
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
            Ignored when ``detections`` is set.
        detections: Path to a pre-computed MOT-format detections file.
            Mutually exclusive with ``model``; supply one or the other.
        confidence: Detection confidence threshold.
        device: Inference device: ``auto``, ``cpu``, ``cuda``, ``cuda:0``,
            ``mps``.
        api_key: Roboflow API key (required for private custom models).
    """

    model: str = DEFAULT_MODEL
    detections: Path | None = None
    confidence: float = DEFAULT_CONFIDENCE
    device: str = DEFAULT_DEVICE
    api_key: str | None = None


@dataclass
class FilteringOptions:
    """Detection and track filters.

    Attributes:
        classes: Comma-separated class names or IDs to keep
            (e.g. ``person,car`` or ``0,2``).
        track_ids: Comma-separated track IDs to keep in the output
            (e.g. ``1,3,5``).
    """

    classes: str | None = None
    track_ids: str | None = None


@dataclass
class OutputOptions:
    """Output paths and write options.

    Attributes:
        output: Annotated-video output path.
        mot_results: MOT-format predictions output path.
        overwrite: Overwrite existing output files without prompting.
    """

    output: Path | None = None
    mot_results: Path | None = None
    overwrite: bool = False


@dataclass
class VisualizationOptions:
    """Live preview and display settings.

    Attributes:
        display: Show a live preview window during tracking.
    """

    display: bool = False


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
class TrackerParams:
    """Optional tracker-specific parameters.

    Union of parameters across all registered trackers; each tracker only
    receives the keys it knows about. Fields left as ``None`` are dropped
    before instantiation so the tracker's own defaults apply.

    Attributes:
        lost_track_buffer: Frames to keep a lost track before discarding.
        frame_rate: Source frame rate for time-based logic.
        track_activation_threshold: Detection score needed to spawn a track.
        minimum_consecutive_frames: Consecutive matches to confirm a track.
        minimum_iou_threshold: IoU threshold for SORT/OC-SORT association.
        minimum_iou_threshold_first_assoc: BoT-SORT first-stage IoU.
        minimum_iou_threshold_second_assoc: BoT-SORT second-stage IoU.
        minimum_iou_threshold_unconfirmed_assoc: BoT-SORT unconfirmed IoU.
        high_conf_det_threshold: High-confidence detection threshold.
        direction_consistency_weight: OC-SORT direction consistency weight.
        delta_t: OC-SORT velocity delta horizon.
        enable_cmc: BoT-SORT camera motion compensation toggle.
        cmc_method: BoT-SORT CMC method name.
        cmc_downscale: BoT-SORT CMC downscale factor.
        instant_first_frame_activation: BoT-SORT first-frame activation toggle.
    """

    lost_track_buffer: int | None = None
    frame_rate: float | None = None
    track_activation_threshold: float | None = None
    minimum_consecutive_frames: int | None = None
    minimum_iou_threshold: float | None = None
    minimum_iou_threshold_first_assoc: float | None = None
    minimum_iou_threshold_second_assoc: float | None = None
    minimum_iou_threshold_unconfirmed_assoc: float | None = None
    high_conf_det_threshold: float | None = None
    direction_consistency_weight: float | None = None
    delta_t: int | None = None
    enable_cmc: bool | None = None
    cmc_method: str | None = None
    cmc_downscale: int | None = None
    instant_first_frame_activation: bool | None = None


def track(
    source: str | None = None,
    detection: DetectionOptions = DetectionOptions(),
    filters: FilteringOptions = FilteringOptions(),
    tracker: str = DEFAULT_TRACKER,
    tracker_params: TrackerParams | None = None,
    out: OutputOptions = OutputOptions(),
    vis: VisualizationOptions = VisualizationOptions(),
    show: ShowOptions = ShowOptions(),
) -> int:
    """Run detection and tracking over a video, webcam, RTSP, or image directory.

    Args:
        source: Video file, webcam index (e.g. ``"0"``), RTSP URL, or image
            directory. Required unless ``detection.detections`` is supplied.
        detection: Detection model and inference options.
        filters: Class and track-ID filters applied to detections and tracks.
        tracker: Tracking algorithm ID. Discoverable via
            ``BaseTracker._registered_trackers()``.
        tracker_params: Optional tracker parameter overrides; only fields
            matching the chosen tracker's ``__init__`` are forwarded.
        out: Output path and overwrite options.
        vis: Live preview and display options.
        show: Annotation elements to draw on each frame.

    Returns:
        Exit code: ``0`` on success, ``1`` on validation error.
    """
    model = detection.model
    detections = detection.detections
    confidence = detection.confidence
    device = detection.device
    api_key = detection.api_key
    classes = filters.classes
    track_ids = filters.track_ids
    output = out.output
    mot_results = out.mot_results
    overwrite = out.overwrite
    display = vis.display
    show_boxes = show.boxes
    show_masks = show.masks
    show_labels = show.labels
    show_ids = show.ids
    show_confidence = show.confidence
    show_trajectories = show.trajectories

    needs_frames = output is not None or display

    if source is None and detections is None:
        print("Error: --source is required when not using --detections.", file=sys.stderr)
        return 1
    if needs_frames and source is None:
        print("Error: --source is required when using --output or --display.", file=sys.stderr)
        return 1

    if output:
        _validate_output_path(_resolve_video_output_path(output), overwrite=overwrite)
    if mot_results:
        _validate_output_path(mot_results, overwrite=overwrite)

    if detections is not None:
        model_obj: AnyModel | None = None
        detections_data: dict | None = load_mot_file(detections)
        class_names: list[str] = []
    else:
        model_obj = _init_model(model, device=device, api_key=api_key)
        detections_data = None
        class_names = getattr(model_obj, "class_names", [])

    class_filter = _resolve_class_filter(classes, class_names)
    track_id_filter = _resolve_track_id_filter(track_ids)
    tracker_obj = _init_tracker(tracker, tracker_params)

    if source is not None:
        return _run_with_source(
            source=source,
            model=model_obj,
            confidence=confidence,
            detections_data=detections_data,
            class_names=class_names,
            class_filter=class_filter,
            track_id_filter=track_id_filter,
            tracker=tracker_obj,
            output=output,
            mot_results=mot_results,
            display=display,
            show_boxes=show_boxes,
            show_masks=show_masks,
            show_labels=show_labels,
            show_ids=show_ids,
            show_confidence=show_confidence,
            show_trajectories=show_trajectories,
        )

    return _run_frameless(
        detections_data=detections_data,
        class_filter=class_filter,
        track_id_filter=track_id_filter,
        tracker=tracker_obj,
        mot_results=mot_results,
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

    try:
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
    except KeyboardInterrupt:
        pass

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
    show_boxes: bool,
    show_masks: bool,
    show_labels: bool,
    show_ids: bool,
    show_confidence: bool,
    show_trajectories: bool,
) -> int:
    """Run tracking with a frame source (video, webcam, images)."""
    frame_gen = frames_from_source(source)
    source_info = _classify_source(source)

    annotators, label_annotator = _init_annotators(
        show_boxes=show_boxes,
        show_masks=show_masks,
        show_labels=show_labels,
        show_ids=show_ids,
        show_confidence=show_confidence,
    )
    trace_annotator = (
        sv.TraceAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.TRACK) if show_trajectories else None
    )
    display_ctx = _DisplayWindow() if display else nullcontext()

    try:
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
                        labels = _format_labels(
                            labeled,
                            class_names,
                            show_ids=show_ids,
                            show_labels=show_labels,
                            show_confidence=show_confidence,
                        )
                        annotated = label_annotator.annotate(annotated, labeled, labels)

                    video.write(annotated)

                    if display_win is not None:
                        display_win.show(annotated)
                        if display_win.quit_requested:
                            interrupted = True
                            break

            progress.complete(interrupted=interrupted)
    except KeyboardInterrupt:
        pass

    return 0


def _resolve_track_id_filter(track_ids_arg: str | None) -> list[int] | None:
    """Resolve a comma-separated ``track_ids`` string to a list of integer IDs.

    Args:
        track_ids_arg: Raw ``--track_ids`` string (e.g. ``"1,3,5"``). ``None``
            means no filter.

    Returns:
        List of integer track IDs, or ``None`` when no valid filter remains.
    """
    if not track_ids_arg:
        return None

    track_ids: list[int] = []
    for raw in track_ids_arg.split(","):
        token = raw.strip()
        try:
            track_ids.append(int(token))
        except ValueError:
            print(f"Warning: '{token}' is not a valid track ID, skipping.", file=sys.stderr)
    return track_ids or None


def _resolve_class_filter(classes_arg: str | None, class_names: list[str]) -> list[int] | None:
    """Resolve a comma-separated ``classes`` string to a list of integer IDs.

    Each token is checked independently: if it parses as an ``int`` it is used
    directly as a class ID; otherwise it is looked up by name in ``class_names``.
    Unknown names are printed as warnings and skipped.

    Args:
        classes_arg: Raw ``--classes`` string (e.g. ``"person,car"`` or
            ``"0,2"`` or ``"person,2"``). ``None`` means no filter.
        class_names: Ordered list of class names where the index equals the
            class ID (as provided by the model).

    Returns:
        List of integer class IDs, or ``None`` when no valid filter remains.
    """
    if not classes_arg:
        return None

    name_to_id = {name: i for i, name in enumerate(class_names)}
    class_filter: list[int] = []
    for raw in classes_arg.split(","):
        token = raw.strip()
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


def _init_tracker(tracker_id: str, params: TrackerParams | None) -> BaseTracker:
    """Create a tracker instance from the registry.

    Only fields the chosen tracker accepts are forwarded; ``None`` values are
    always dropped so the tracker's own defaults apply.

    Args:
        tracker_id: Registered tracker name (e.g. ``bytetrack``, ``sort``).
        params: Optional tracker parameter overrides.

    Returns:
        Initialised tracker instance.

    Raises:
        ValueError: If ``tracker_id`` is not registered.
    """
    info = BaseTracker._lookup_tracker(tracker_id)
    if info is None:
        available = ", ".join(BaseTracker._registered_trackers())
        raise ValueError(f"Unknown tracker: '{tracker_id}'. Available: {available}")

    raw = asdict(params) if params is not None else {}
    accepted = set(info.parameters)
    kwargs = {k: v for k, v in raw.items() if v is not None and k in accepted}
    return info.tracker_class(**kwargs)


def _init_annotators(
    show_boxes: bool = False,
    show_masks: bool = False,
    show_labels: bool = False,
    show_ids: bool = False,
    show_confidence: bool = False,
) -> tuple[list, sv.LabelAnnotator | None]:
    """Initialise supervision annotators based on display options.

    Args:
        show_boxes: Create ``BoxAnnotator``.
        show_masks: Create ``MaskAnnotator``.
        show_labels: Include class labels (triggers ``LabelAnnotator``).
        show_ids: Include track IDs (triggers ``LabelAnnotator``).
        show_confidence: Include confidence scores (triggers ``LabelAnnotator``).

    Returns:
        Tuple of (annotators list, label_annotator or None). Label annotator is
        separate because it needs custom labels per frame.
    """
    annotators: list = []
    label_annotator: sv.LabelAnnotator | None = None

    if show_boxes:
        annotators.append(sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.TRACK))
    if show_masks:
        annotators.append(sv.MaskAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.TRACK))
    if show_labels or show_ids or show_confidence:
        label_annotator = sv.LabelAnnotator(
            color=COLOR_PALETTE,
            text_color=sv.Color.BLACK,
            text_position=sv.Position.TOP_LEFT,
            color_lookup=sv.ColorLookup.TRACK,
        )
    return annotators, label_annotator


def _format_labels(
    detections: sv.Detections,
    class_names: list[str],
    *,
    show_ids: bool = False,
    show_labels: bool = False,
    show_confidence: bool = False,
) -> list[str]:
    """Generate label strings for each detection.

    Args:
        detections: Detections to generate labels for.
        class_names: List of class names for lookup.
        show_ids: Include tracker IDs in labels.
        show_labels: Include class names in labels.
        show_confidence: Include confidence scores in labels.

    Returns:
        List of label strings, one per detection.
    """
    labels = []
    for i in range(len(detections)):
        parts: list[str] = []
        if show_ids and detections.tracker_id is not None:
            parts.append(f"#{int(detections.tracker_id[i])}")
        if show_labels and detections.class_id is not None:
            class_id = int(detections.class_id[i])
            if class_names and 0 <= class_id < len(class_names):
                parts.append(class_names[class_id])
            else:
                parts.append(str(class_id))
        if show_confidence and detections.confidence is not None:
            parts.append(f"{detections.confidence[i]:.2f}")
        labels.append(" ".join(parts))
    return labels
