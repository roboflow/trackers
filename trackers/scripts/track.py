#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Track objects in video using detection models and tracking algorithms."""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import supervision as sv

from trackers import frames_from_source
from trackers.core.base import BaseTracker
from trackers.io import (
    DisplayWindow,
    MOTOutput,
    VideoOutput,
    load_mot_file,
    resolve_video_output_path,
    validate_output_path,
)
from trackers.io.mot import _mot_frame_to_detections
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


def add_track_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add the track subcommand to the argument parser."""
    parser = subparsers.add_parser(
        "track",
        help="Track objects in video using detection and tracking.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Source options
    source_group = parser.add_argument_group("source")
    source_group.add_argument(
        "--source",
        type=str,
        required=True,
        metavar="PATH",
        help="Video file, webcam index (0), RTSP URL, or image directory.",
    )

    # Detection options (mutually exclusive)
    detection_group = parser.add_argument_group("detection")
    det_mutex = detection_group.add_mutually_exclusive_group(required=True)
    det_mutex.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        metavar="ID",
        help=(
            "Model ID for detection. Pretrained: rfdetr-nano, rfdetr-base, etc. "
            f"Custom: workspace/project/version. Default: {DEFAULT_MODEL}"
        ),
    )
    det_mutex.add_argument(
        "--detections",
        type=Path,
        metavar="PATH",
        help="Load pre-computed detections from MOT format file.",
    )

    # Model options
    model_group = parser.add_argument_group("model options")
    model_group.add_argument(
        "--model.confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        dest="model_confidence",
        metavar="FLOAT",
        help=f"Detection confidence threshold. Default: {DEFAULT_CONFIDENCE}",
    )
    model_group.add_argument(
        "--model.device",
        type=str,
        default=DEFAULT_DEVICE,
        dest="model_device",
        metavar="DEVICE",
        help=f"Device: auto, cpu, cuda, cuda:0, mps. Default: {DEFAULT_DEVICE}",
    )
    model_group.add_argument(
        "--model.api_key",
        type=str,
        default=None,
        dest="model_api_key",
        metavar="KEY",
        help="Roboflow API key for custom models.",
    )

    # Filtering options
    filter_group = parser.add_argument_group("filtering")
    filter_group.add_argument(
        "--classes",
        type=str,
        default=None,
        metavar="IDS",
        help="Filter by class IDs (comma-separated, e.g., 0,1,2).",
    )

    # Tracker options
    tracker_group = parser.add_argument_group("tracker options")
    available_trackers = BaseTracker._registered_trackers()
    tracker_group.add_argument(
        "--tracker",
        type=str,
        default=DEFAULT_TRACKER,
        choices=available_trackers if available_trackers else [DEFAULT_TRACKER, "sort"],
        metavar="ID",
        help=f"Tracking algorithm. Default: {DEFAULT_TRACKER}",
    )

    # Add dynamic tracker parameters
    _add_tracker_params(tracker_group)

    # Output options
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output video file path.",
    )
    output_group.add_argument(
        "--mot-output",
        type=Path,
        default=None,
        dest="mot_output",
        metavar="PATH",
        help="Output MOT format file path.",
    )
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    # Visualization options
    vis_group = parser.add_argument_group("visualization")
    vis_group.add_argument(
        "--display",
        action="store_true",
        help="Show preview window.",
    )
    vis_group.add_argument(
        "--show-boxes",
        action="store_true",
        default=True,
        dest="show_boxes",
        help="Draw bounding boxes. Default: True",
    )
    vis_group.add_argument(
        "--no-boxes",
        action="store_false",
        dest="show_boxes",
        help="Disable bounding boxes.",
    )
    vis_group.add_argument(
        "--show-masks",
        action="store_true",
        dest="show_masks",
        help="Draw segmentation masks (seg models only).",
    )
    vis_group.add_argument(
        "--show-labels",
        action="store_true",
        dest="show_labels",
        help="Show class labels.",
    )
    vis_group.add_argument(
        "--show-ids",
        action="store_true",
        dest="show_ids",
        help="Show track IDs.",
    )
    vis_group.add_argument(
        "--show-confidence",
        action="store_true",
        dest="show_confidence",
        help="Show confidence scores.",
    )
    vis_group.add_argument(
        "--show-trajectories",
        action="store_true",
        dest="show_trajectories",
        help="Draw track trajectories.",
    )

    parser.set_defaults(func=run_track)


def _add_tracker_params(group: argparse._ArgumentGroup) -> None:
    """Add tracker-specific parameters from registry to argument group."""
    for tracker_id in BaseTracker._registered_trackers():
        info = BaseTracker._lookup_tracker(tracker_id)
        if info is None:
            continue

        for param_name, param_info in info.parameters.items():
            arg_name = f"--tracker.{param_name}"
            dest_name = f"tracker_{param_name}"

            kwargs: dict = {
                "dest": dest_name,
                "default": param_info.default_value,
                "help": f"{param_info.description} Default: {param_info.default_value}",
            }

            if param_info.param_type is bool:
                kwargs["action"] = (
                    "store_false" if param_info.default_value else "store_true"
                )
            else:
                kwargs["type"] = param_info.param_type
                kwargs["metavar"] = param_info.param_type.__name__.upper()

            try:
                group.add_argument(arg_name, **kwargs)
            except argparse.ArgumentError:
                # Parameter already added by another tracker
                pass


def run_track(args: argparse.Namespace) -> int:
    """Execute the track command."""
    # Validate output paths
    if args.output:
        validate_output_path(
            resolve_video_output_path(args.output), overwrite=args.overwrite
        )
    if args.mot_output:
        validate_output_path(args.mot_output, overwrite=args.overwrite)

    # Parse class filter
    class_filter = None
    if args.classes:
        class_filter = [int(c.strip()) for c in args.classes.split(",")]

    # Create detection source
    if args.detections:
        model = None
        detections_data = load_mot_file(args.detections)
        class_names: list[str] = []
    else:
        model = _init_model(
            args.model,
            device=args.model_device,
            api_key=args.model_api_key,
        )
        detections_data = None
        class_names = getattr(model, "class_names", [])

    # Create tracker
    tracker_params = _extract_tracker_params(args.tracker, args)
    tracker = _init_tracker(args.tracker, **tracker_params)

    # Create frame generator
    frame_gen = frames_from_source(args.source)

    # Setup annotators
    annotators, label_annotator = _init_annotators(
        show_boxes=args.show_boxes,
        show_masks=args.show_masks,
        show_labels=args.show_labels,
        show_ids=args.show_ids,
        show_confidence=args.show_confidence,
    )
    trace_annotator = None
    if args.show_trajectories:
        trace_annotator = sv.TraceAnnotator(
            color=COLOR_PALETTE,
            color_lookup=sv.ColorLookup.TRACK,
        )

    display_ctx = DisplayWindow() if args.display else nullcontext()

    try:
        with (
            VideoOutput(args.output) as video,
            MOTOutput(args.mot_output) as mot,
            display_ctx as display,
        ):
            for frame_idx, frame in frame_gen:
                # Get detections
                if model is not None:
                    detections = _run_model(model, frame, args.model_confidence)
                elif detections_data is not None and frame_idx in detections_data:
                    detections = _mot_frame_to_detections(detections_data[frame_idx])
                else:
                    detections = sv.Detections.empty()

                # Filter by class
                if class_filter is not None and len(detections) > 0:
                    mask = np.isin(detections.class_id, class_filter)
                    detections = detections[mask]  # type: ignore[assignment]

                # Run tracker
                tracked = tracker.update(detections)

                # Write MOT output
                mot.write(frame_idx, tracked)

                # Annotate and display/save frame
                if args.display or args.output:
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
                            show_ids=args.show_ids,
                            show_labels=args.show_labels,
                            show_confidence=args.show_confidence,
                        )
                        annotated = label_annotator.annotate(annotated, labeled, labels)

                    video.write(annotated)

                    if display is not None:
                        display.show(annotated)
                        if display.quit_requested:
                            break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    return 0


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


def _extract_tracker_params(
    tracker_id: str, args: argparse.Namespace
) -> dict[str, object]:
    """Extract tracker parameters from CLI args.

    Args:
        tracker_id: Registered tracker name.
        args: Parsed CLI arguments.

    Returns:
        Dictionary of tracker parameters with non-None values.
    """
    info = BaseTracker._lookup_tracker(tracker_id)
    if info is None:
        return {}

    params = {}
    for param_name in info.parameters:
        dest_name = f"tracker_{param_name}"
        if hasattr(args, dest_name):
            value = getattr(args, dest_name)
            if value is not None:
                params[param_name] = value
    return params


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
