#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import sys
from contextlib import nullcontext
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


def track(
    source: str | None = None,
    detection_model: str = DEFAULT_MODEL,
    detections: Path | None = None,
    detection_confidence: float = DEFAULT_CONFIDENCE,
    detection_device: str = DEFAULT_DEVICE,
    detection_api_key: str | None = None,
    filters_classes: str | None = None,
    filters_track_ids: str | None = None,
    tracker: str = DEFAULT_TRACKER,
    tracker_lost_track_buffer: int | None = None,
    tracker_frame_rate: float | None = None,
    tracker_track_activation_threshold: float | None = None,
    tracker_minimum_consecutive_frames: int | None = None,
    tracker_minimum_iou_threshold: float | None = None,
    tracker_high_conf_det_threshold: float | None = None,
    tracker_minimum_iou_threshold_first_assoc: float | None = None,
    tracker_minimum_iou_threshold_second_assoc: float | None = None,
    tracker_minimum_iou_threshold_unconfirmed_assoc: float | None = None,
    tracker_enable_cmc: bool | None = None,
    tracker_cmc_method: str | None = None,
    tracker_cmc_downscale: int | None = None,
    tracker_instant_first_frame_activation: bool | None = None,
    tracker_direction_consistency_weight: float | None = None,
    tracker_delta_t: int | None = None,
    out_output: Path | None = None,
    out_mot_results: Path | None = None,
    out_overwrite: bool = False,
    show_display: bool = False,
    show_boxes: bool = True,
    show_masks: bool = False,
    show_labels: bool = False,
    show_ids: bool = True,
    show_confidence: bool = False,
    show_trajectories: bool = False,
) -> int:
    """Track objects in video using detection and tracking.

    Tracker-specific parameters are exposed as ``--tracker.<param>`` flags.
    Each flag defaults to ``None`` (meaning "use the tracker's own default");
    only flags the user supplies are forwarded to the tracker constructor.

    Args:
        source: Video file, webcam index (0), RTSP URL, or image directory.
        detection_model: Model ID for detection (e.g. rfdetr-nano, rfdetr-base, workspace/project/version).
        detections: Load pre-computed detections from MOT format file (mutually exclusive with detection model).
        detection_confidence: Detection confidence threshold.
        detection_device: Device to run model on (auto, cpu, cuda, cuda:0, mps).
        detection_api_key: Roboflow API key for custom models.
        filters_classes: Filter by class names or IDs (comma-separated, e.g. person,car).
        filters_track_ids: Filter output by track IDs (comma-separated, e.g. 1,3,5).
        tracker: Tracking algorithm ID (``bytetrack``, ``sort``, ``ocsort``, ``botsort``).
        tracker_lost_track_buffer: Frames a lost track is kept before deletion. Common to all trackers.
        tracker_frame_rate: Source frame rate used by the tracker for time-based logic. Common to all trackers.
        tracker_track_activation_threshold: Detection confidence to start a new track. Applies to ``bytetrack``,
            ``sort``, ``botsort``.
        tracker_minimum_consecutive_frames: Frames a new track must be matched before being confirmed. Common to
            all trackers.
        tracker_minimum_iou_threshold: IoU threshold for association. Applies to ``bytetrack``, ``sort``, ``ocsort``.
        tracker_high_conf_det_threshold: High-confidence detection threshold for the first association pass.
            Applies to ``bytetrack``, ``ocsort``, ``botsort``.
        tracker_minimum_iou_threshold_first_assoc: IoU threshold for the first association pass. Applies to
            ``botsort`` only.
        tracker_minimum_iou_threshold_second_assoc: IoU threshold for the second association pass. Applies to
            ``botsort`` only.
        tracker_minimum_iou_threshold_unconfirmed_assoc: IoU threshold for unconfirmed-track association.
            Applies to ``botsort`` only.
        tracker_enable_cmc: Enable camera-motion compensation. Applies to ``botsort`` only.
        tracker_cmc_method: Camera-motion compensation method (e.g. ``sparseOptFlow``). Applies to ``botsort`` only.
        tracker_cmc_downscale: Frame downscale factor used by CMC. Applies to ``botsort`` only.
        tracker_instant_first_frame_activation: Activate tracks immediately on the first frame. Applies to
            ``botsort`` only.
        tracker_direction_consistency_weight: Weight of the direction-consistency term during association.
            Applies to ``ocsort`` only.
        tracker_delta_t: Frame gap used for OC-SORT's observation-centric update. Applies to ``ocsort`` only.
        out_output: Output video file path.
        out_mot_results: Output MOT format file path.
        out_overwrite: Overwrite existing output files.
        show_display: Show preview window.
        show_boxes: Draw bounding boxes.
        show_masks: Draw segmentation masks (segmentation models only).
        show_labels: Show class labels.
        show_ids: Show track IDs.
        show_confidence: Show confidence scores.
        show_trajectories: Draw track trajectories.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    needs_frames = out_output or show_display

    if source is None and not detections:
        print(
            "Error: --source is required when not using --detections.",
            file=sys.stderr,
        )
        return 1

    if detection_model != DEFAULT_MODEL and detections is not None:
        print(
            "Error: --detection.model and --detections are mutually exclusive.",
            file=sys.stderr,
        )
        return 1

    if needs_frames and source is None:
        print(
            "Error: --source is required when using --out.output or --show.display.",
            file=sys.stderr,
        )
        return 1

    if out_output:
        _validate_output_path(_resolve_video_output_path(out_output), overwrite=out_overwrite)
    if out_mot_results:
        _validate_output_path(out_mot_results, overwrite=out_overwrite)

    if detections:
        loaded_model = None
        detections_data = load_mot_file(detections)
        class_names: list[str] = []
    else:
        loaded_model = _init_model(
            detection_model,
            device=detection_device,
            api_key=detection_api_key,
        )
        detections_data = None
        class_names = getattr(loaded_model, "class_names", [])

    class_filter = _resolve_class_filter(filters_classes, class_names)
    track_id_filter = _resolve_track_id_filter(filters_track_ids)

    tracker_kwargs: dict[str, object] = {
        k: v
        for k, v in {
            "lost_track_buffer": tracker_lost_track_buffer,
            "frame_rate": tracker_frame_rate,
            "track_activation_threshold": tracker_track_activation_threshold,
            "minimum_consecutive_frames": tracker_minimum_consecutive_frames,
            "minimum_iou_threshold": tracker_minimum_iou_threshold,
            "high_conf_det_threshold": tracker_high_conf_det_threshold,
            "minimum_iou_threshold_first_assoc": tracker_minimum_iou_threshold_first_assoc,
            "minimum_iou_threshold_second_assoc": tracker_minimum_iou_threshold_second_assoc,
            "minimum_iou_threshold_unconfirmed_assoc": tracker_minimum_iou_threshold_unconfirmed_assoc,
            "enable_cmc": tracker_enable_cmc,
            "cmc_method": tracker_cmc_method,
            "cmc_downscale": tracker_cmc_downscale,
            "instant_first_frame_activation": tracker_instant_first_frame_activation,
            "direction_consistency_weight": tracker_direction_consistency_weight,
            "delta_t": tracker_delta_t,
        }.items()
        if v is not None
    }
    tracker_obj = _init_tracker(tracker, **tracker_kwargs)

    if source is not None:
        return _run_with_source(
            source=source,
            loaded_model=loaded_model,
            detections_data=detections_data,
            class_names=class_names,
            class_filter=class_filter,
            track_id_filter=track_id_filter,
            tracker=tracker_obj,
            out_output=out_output,
            out_mot_results=out_mot_results,
            show_display=show_display,
            detection_confidence=detection_confidence,
            show_boxes=show_boxes,
            show_masks=show_masks,
            show_labels=show_labels,
            show_ids=show_ids,
            show_confidence=show_confidence,
            show_trajectories=show_trajectories,
        )
    else:
        return _run_frameless(
            detections_data=detections_data,
            class_filter=class_filter,
            track_id_filter=track_id_filter,
            tracker=tracker_obj,
            out_mot_results=out_mot_results,
        )


def _run_frameless(
    detections_data: dict | None,
    class_filter: list[int] | None,
    track_id_filter: list[int] | None,
    tracker: BaseTracker,
    out_mot_results: Path | None,
) -> int:
    """Run tracking from pre-computed detections without a frame source."""
    if detections_data is None or not detections_data:
        print("Error: No detections found in file.", file=sys.stderr)
        return 1

    total_frames = max(detections_data.keys())
    source_info = _SourceInfo(source_type="video", total_frames=total_frames)

    try:
        with (
            _MOTOutput(out_mot_results) as mot,
            _TrackingProgress(source_info) as progress,
        ):
            interrupted = False
            for frame_idx in range(1, total_frames + 1):
                if frame_idx in detections_data:
                    dets = _mot_frame_to_detections(detections_data[frame_idx])
                else:
                    dets = sv.Detections.empty()

                if class_filter is not None and len(dets) > 0 and dets.class_id is not None:
                    mask = np.isin(dets.class_id, class_filter)
                    dets = dets[mask]  # type: ignore[assignment]

                tracked = tracker.update(dets)

                if track_id_filter is not None and len(tracked) > 0:
                    if tracked.tracker_id is not None:
                        mask = np.isin(tracked.tracker_id.astype(int), track_id_filter)
                        tracked = tracked[mask]  # type: ignore[assignment]

                mot.write(frame_idx, tracked)
                progress.update()

            progress.complete(interrupted=interrupted)

    except KeyboardInterrupt:
        pass

    return 0


def _run_with_source(
    source: str,
    loaded_model: AnyModel | None,
    detections_data: dict | None,
    class_names: list[str],
    class_filter: list[int] | None,
    track_id_filter: list[int] | None,
    tracker: BaseTracker,
    out_output: Path | None,
    out_mot_results: Path | None,
    show_display: bool,
    detection_confidence: float,
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
    trace_annotator = None
    if show_trajectories:
        trace_annotator = sv.TraceAnnotator(
            color=COLOR_PALETTE,
            color_lookup=sv.ColorLookup.TRACK,
        )

    display_ctx = _DisplayWindow() if show_display else nullcontext()

    try:
        with (
            _VideoOutput(
                out_output,
                fps=source_info.fps or _DEFAULT_OUTPUT_FPS,
            ) as video,
            _MOTOutput(out_mot_results) as mot,
            display_ctx as display_win,
            _TrackingProgress(source_info) as progress,
        ):
            interrupted = False
            for frame_idx, frame in frame_gen:
                if loaded_model is not None:
                    dets = _run_model(loaded_model, frame, detection_confidence)
                elif detections_data is not None and frame_idx in detections_data:
                    dets = _mot_frame_to_detections(detections_data[frame_idx])
                else:
                    dets = sv.Detections.empty()

                if class_filter is not None and len(dets) > 0 and dets.class_id is not None:
                    mask = np.isin(dets.class_id, class_filter)
                    dets = dets[mask]  # type: ignore[assignment]

                tracked = tracker.update(dets, frame)

                if track_id_filter is not None and len(tracked) > 0:
                    if tracked.tracker_id is not None:
                        mask = np.isin(tracked.tracker_id.astype(int), track_id_filter)
                        tracked = tracked[mask]  # type: ignore[assignment]

                mot.write(frame_idx, tracked)
                progress.update()

                if show_display or out_output:
                    annotated = frame.copy()
                    if trace_annotator is not None:
                        annotated = trace_annotator.annotate(annotated, tracked)
                    for annotator in annotators:
                        annotated = annotator.annotate(annotated, tracked)
                    if label_annotator is not None:
                        labeled: sv.Detections = tracked[tracked.tracker_id != -1]  # type: ignore[assignment]
                        labels = _format_labels(
                            labeled,
                            class_names,
                            show_ids=show_ids,
                            show_labels=show_labels,
                            show_confidence=show_confidence,
                        )
                        annotated = label_annotator.annotate(annotated, labeled, labels)  # type: ignore[assignment,arg-type]

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
    """Resolve a comma-separated ``--filters.track-ids`` value to a list of integer IDs.

    Args:
        track_ids_arg: Raw ``--filters.track-ids`` string (e.g. ``"1,3,5"``). ``None``
            means no filter.

    Returns:
        List of integer track IDs, or ``None`` when no valid filter remains.

    Examples:
        >>> _resolve_track_id_filter(None) is None
        True
        >>> _resolve_track_id_filter("1,3") == [1, 3]
        True
    """
    if not track_ids_arg:
        return None

    track_ids: list[int] = []
    for token in track_ids_arg.split(","):
        token = token.strip()
        try:
            track_ids.append(int(token))
        except ValueError:
            print(
                f"Warning: '{token}' is not a valid track ID, skipping.",
                file=sys.stderr,
            )
    return track_ids if track_ids else None


def _resolve_class_filter(
    classes_arg: str | None,
    class_names: list[str],
) -> list[int] | None:
    """Resolve a comma-separated ``--filters.classes`` value to a list of integer class IDs.

    Each token is checked independently: if it parses as an ``int`` it is used
    directly as a class ID; otherwise it is looked up by name in *class_names*.
    Unknown names are printed as warnings and skipped.

    Args:
        classes_arg: Raw ``--filters.classes`` string (e.g. ``"person,car"`` or
            ``"0,2"`` or ``"person,2"``). ``None`` means no filter.
        class_names: Ordered list of class names where the index equals the
            class ID (as provided by the model).

    Returns:
        List of integer class IDs, or ``None`` when no valid filter remains.

    Examples:
        >>> _resolve_class_filter(None, []) is None
        True
        >>> _resolve_class_filter("0,2", ["person", "bicycle", "car"]) == [0, 2]
        True
    """
    if not classes_arg:
        return None

    requested = [token.strip() for token in classes_arg.split(",")]
    name_to_id = {name: i for i, name in enumerate(class_names)}
    class_filter: list[int] = []
    for token in requested:
        try:
            class_filter.append(int(token))
        except ValueError:
            if token in name_to_id:
                class_filter.append(name_to_id[token])
            else:
                print(
                    f"Warning: class '{token}' not found in model class list, skipping.",
                    file=sys.stderr,
                )
    return class_filter if class_filter else None


def _init_model(
    model_id: str,
    *,
    device: str = DEFAULT_DEVICE,
    api_key: str | None = None,
) -> AnyModel:
    """Load a detection model via inference-models.

    Args:
        model_id: Model identifier (e.g. ``'rfdetr-nano'`` or
            ``'workspace/project/version'``).
        device: Device to load model on (``'auto'``, ``'cpu'``, ``'cuda'``, ``'mps'``).
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

    return AutoModel.from_pretrained(
        model_id,
        api_key=api_key,
        device=resolved_device,
    )


def _run_model(model: AnyModel, frame: np.ndarray, confidence: float) -> sv.Detections:
    """Run model inference and return sv.Detections."""
    predictions = model(frame)
    if not predictions:
        return sv.Detections.empty()

    detections = predictions[0].to_supervision()

    if len(detections) > 0 and detections.confidence is not None:
        mask = detections.confidence >= confidence
        detections = detections[mask]

    return detections


def _init_tracker(tracker_id: str, **kwargs: object) -> BaseTracker:
    """Create a tracker instance from the registry.

    Args:
        tracker_id: Registered tracker name (e.g. ``'bytetrack'``, ``'sort'``).
        **kwargs: Tracker-specific parameters.

    Returns:
        Initialized tracker instance.

    Raises:
        ValueError: If *tracker_id* is not registered.
    """
    info = BaseTracker._lookup_tracker(tracker_id)
    if info is None:
        available = ", ".join(BaseTracker._registered_trackers())
        raise ValueError(f"Unknown tracker: '{tracker_id}'. Available: {available}")

    return info.tracker_class(**kwargs)


def _init_annotators(
    show_boxes: bool = False,
    show_masks: bool = False,
    show_labels: bool = False,
    show_ids: bool = False,
    show_confidence: bool = False,
) -> tuple[list, sv.LabelAnnotator | None]:
    """Initialize supervision annotators based on display options.

    Args:
        show_boxes: Create BoxAnnotator.
        show_masks: Create MaskAnnotator.
        show_labels: Include class labels (triggers LabelAnnotator).
        show_ids: Include track IDs (triggers LabelAnnotator).
        show_confidence: Include confidence scores (triggers LabelAnnotator).

    Returns:
        Tuple of (annotators list, label_annotator or None).

    Examples:
        >>> annotators, label_annotator = _init_annotators(show_boxes=True)
        >>> len(annotators)
        1
        >>> label_annotator is None
        True
    """
    annotators: list = []
    label_annotator: sv.LabelAnnotator | None = None

    if show_boxes:
        annotators.append(
            sv.BoxAnnotator(
                color=COLOR_PALETTE,
                color_lookup=sv.ColorLookup.TRACK,
            )
        )

    if show_masks:
        annotators.append(
            sv.MaskAnnotator(
                color=COLOR_PALETTE,
                color_lookup=sv.ColorLookup.TRACK,
            )
        )

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

    Examples:
        >>> import supervision as sv
        >>> import numpy as np
        >>> dets = sv.Detections(xyxy=np.array([[0, 0, 1, 1]]))
        >>> _format_labels(dets, [])
        ['']
    """
    labels = []

    for i in range(len(detections)):
        parts = []

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
