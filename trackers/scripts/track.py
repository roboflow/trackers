#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv

from trackers import frames_from_source
from trackers.core.base import BaseTracker
from trackers.io.mot import _load_mot_file, _mot_frame_to_detections, _MOTOutput
from trackers.io.paths import _resolve_video_output_path, _validate_output_path
from trackers.io.video import _DEFAULT_OUTPUT_FPS, _DisplayWindow, _VideoOutput
from trackers.scripts.progress import _classify_source, _TrackingProgress
from trackers.utils.device import _best_device

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
    source: str,
    model: str | None = None,
    detections: Path | None = None,
    model_confidence: float = DEFAULT_CONFIDENCE,
    model_device: str = DEFAULT_DEVICE,
    model_api_key: str | None = None,
    classes: str | None = None,
    tracker: str = DEFAULT_TRACKER,
    tracker_params: dict[str, Any] | None = None,
    output: Path | None = None,
    mot_output: Path | None = None,
    overwrite: bool = False,
    display: bool = False,
    show_boxes: bool = True,
    show_masks: bool = False,
    show_labels: bool = False,
    show_ids: bool = True,
    show_confidence: bool = False,
    show_trajectories: bool = False,
) -> int:
    """Track objects in video using detection and tracking.

    Args:
        source: Video file, webcam index (0), RTSP URL, or image directory.
        model: Model ID for detection. Pretrained: rfdetr-nano, rfdetr-base, etc.
            Custom: workspace/project/version. Defaults to rfdetr-nano when
            --detections is not provided.
        detections: Load pre-computed detections from MOT format file.
        model_confidence: Detection confidence threshold.
        model_device: Device: auto, cpu, cuda, cuda:0, mps.
        model_api_key: Roboflow API key for custom models.
        classes: Filter by class names or IDs (comma-separated, e.g., person,car).
        tracker: Tracking algorithm.
        tracker_params: Tracker-specific parameters as key-value pairs.
        output: Output video file path.
        mot_output: Output MOT format file path.
        overwrite: Overwrite existing output files.
        display: Show preview window.
        show_boxes: Draw bounding boxes.
        show_masks: Draw segmentation masks (seg models only).
        show_labels: Show class labels.
        show_ids: Show track IDs.
        show_confidence: Show confidence scores.
        show_trajectories: Draw track trajectories.
    """
    if detections is not None and model is not None:
        raise ValueError(
            "Arguments --model and --detections are mutually exclusive. "
            "Provide only one."
        )

    # Validate output paths
    if output:
        _validate_output_path(_resolve_video_output_path(output), overwrite=overwrite)
    if mot_output:
        _validate_output_path(mot_output, overwrite=overwrite)

    # Create detection source
    if detections is not None:
        model_obj = None
        detections_data = _load_mot_file(detections)
        class_names: list[str] = []
    else:
        model_id = model or DEFAULT_MODEL
        model_obj = _init_model(
            model_id,
            device=model_device,
            api_key=model_api_key,
        )
        detections_data = None
        class_names = getattr(model_obj, "class_names", [])

    # Resolve class filter (names and/or integer IDs)
    class_filter = _resolve_class_filter(classes, class_names)

    # Create tracker
    tracker_obj = _init_tracker(tracker, **(tracker_params or {}))

    # Create frame generator
    frame_gen = frames_from_source(source)

    source_info = _classify_source(source)

    # Setup annotators
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

    display_ctx = _DisplayWindow() if display else nullcontext()

    try:
        with (
            _VideoOutput(
                output,
                fps=source_info.fps or _DEFAULT_OUTPUT_FPS,
            ) as video,
            _MOTOutput(mot_output) as mot,
            display_ctx as display_win,
            _TrackingProgress(source_info) as progress,
        ):
            interrupted = False
            for frame_idx, frame in frame_gen:
                # Get detections
                if model_obj is not None:
                    dets = _run_model(model_obj, frame, model_confidence)
                elif detections_data is not None and frame_idx in detections_data:
                    dets = _mot_frame_to_detections(detections_data[frame_idx])
                else:
                    dets = sv.Detections.empty()

                # Filter by class
                if class_filter is not None and len(dets) > 0:
                    mask = np.isin(dets.class_id, class_filter)
                    dets = dets[mask]  # type: ignore[assignment]

                # Run tracker
                tracked = tracker_obj.update(dets)

                # Write MOT output
                mot.write(frame_idx, tracked)

                progress.update()

                # Annotate and display/save frame
                if display or output:
                    annotated = frame.copy()
                    if trace_annotator is not None:
                        annotated = trace_annotator.annotate(annotated, tracked)
                    for annotator in annotators:
                        annotated = annotator.annotate(annotated, tracked)
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
        pass  # progress.__exit__ already printed the final line

    return 0


def _resolve_class_filter(
    classes_arg: str | None,
    class_names: list[str],
) -> list[int] | None:
    """Resolve a comma-separated `--classes` value to a list of integer IDs.

    Each token is checked independently: if it parses as an `int` it is used
    directly as a class ID; otherwise it is looked up by name in *class_names*.
    Unknown names are printed as warnings and skipped.

    Args:
        classes_arg: Raw `--classes` string (e.g. `"person,car"` or
            `"0,2"` or `"person,2"`). `None` means no filter.
        class_names: Ordered list of class names where the index equals the
            class ID (as provided by the model).

    Returns:
        List of integer class IDs, or `None` when no valid filter remains.
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
                    f"Warning: class '{token}' not found in model class "
                    "list, skipping.",
                    file=sys.stderr,
                )
    return class_filter if class_filter else None


def _init_model(
    model_id: str,
    *,
    device: str = DEFAULT_DEVICE,
    api_key: str | None = None,
):
    """Load detection model via inference-models.

    Args:
        model_id: Model identifier (e.g., 'rfdetr-nano' or 'workspace/project/version').
        device: Device to load model on ('auto', 'cpu', 'cuda', 'mps').
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


def _run_model(model, frame: np.ndarray, confidence: float) -> sv.Detections:
    """Run model inference and return sv.Detections."""
    predictions = model(frame)
    if not predictions:
        return sv.Detections.empty()

    detections = predictions[0].to_supervision()

    # Filter by confidence
    if len(detections) > 0 and detections.confidence is not None:
        mask = detections.confidence >= confidence
        detections = detections[mask]

    return detections


def _init_tracker(tracker_id: str, **kwargs) -> BaseTracker:
    """Create tracker instance from registry.

    Args:
        tracker_id: Registered tracker name (e.g., 'bytetrack', 'sort').
        **kwargs: Tracker-specific parameters.

    Returns:
        Initialized tracker instance.

    Raises:
        ValueError: If tracker_id is not registered.
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
        Label annotator is separate because it needs custom labels per frame.
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
