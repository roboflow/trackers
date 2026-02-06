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
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np
import supervision as sv

from trackers.core.base import BaseTracker

if TYPE_CHECKING:
    from trackers.eval.io import MOTFrameData


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
    det_mutex = detection_group.add_mutually_exclusive_group()
    det_mutex.add_argument(
        "--model",
        type=str,
        default="rfdetr-nano",
        metavar="ID",
        help=(
            "Model ID for detection. Pretrained: rfdetr-nano, rfdetr-base, etc. "
            "Custom: workspace/project/version. Default: rfdetr-nano"
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
        default=0.5,
        dest="model_confidence",
        metavar="FLOAT",
        help="Detection confidence threshold. Default: 0.5",
    )
    model_group.add_argument(
        "--model.device",
        type=str,
        default="auto",
        dest="model_device",
        metavar="DEVICE",
        help="Device: auto, cpu, cuda, cuda:0, mps. Default: auto",
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
        default="bytetrack",
        choices=available_trackers if available_trackers else ["bytetrack", "sort"],
        metavar="ID",
        help="Tracking algorithm. Default: bytetrack",
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
        _check_output_writable(
            _resolve_video_output_path(args.output), overwrite=args.overwrite
        )
    if args.mot_output:
        _check_output_writable(args.mot_output, overwrite=args.overwrite)

    # Parse class filter
    class_filter = None
    if args.classes:
        class_filter = [int(c.strip()) for c in args.classes.split(",")]

    # Create detection source
    if args.detections:
        model = None
        detections_data = _load_mot_detections(args.detections)
        class_names: list[str] = []
    else:
        model = _create_model(args)
        detections_data = None
        class_names = getattr(model, "class_names", [])

    # Create tracker
    tracker = _create_tracker(args)

    # Create frame generator
    frame_gen = create_frame_generator(args.source)

    # Setup output writers
    video_writer = None
    mot_file = None

    # Setup annotators
    annotators, label_annotator = _create_annotators(args)
    trace_annotator = None
    if args.show_trajectories:
        trace_annotator = sv.TraceAnnotator(
            color=COLOR_PALETTE,
            color_lookup=sv.ColorLookup.TRACK,
        )

    try:
        if args.mot_output:
            args.mot_output.parent.mkdir(parents=True, exist_ok=True)
            mot_file = open(args.mot_output, "w")

        for frame_idx, frame in frame_gen:
            # Get detections
            if model is not None:
                detections = _run_model_inference(model, frame, args.model_confidence)
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
            if mot_file is not None:
                _write_mot_frame(mot_file, frame_idx, tracked)

            # Annotate frame
            if args.display or args.output:
                annotated = frame.copy()
                if trace_annotator is not None:
                    annotated = trace_annotator.annotate(annotated, tracked)
                for annotator in annotators:
                    annotated = annotator.annotate(annotated, tracked)
                if label_annotator is not None:
                    labeled = tracked[tracked.tracker_id != -1]
                    labels = _generate_labels(
                        labeled,
                        class_names,
                        show_ids=args.show_ids,
                        show_labels=args.show_labels,
                        show_confidence=args.show_confidence,
                    )
                    annotated = label_annotator.annotate(annotated, labeled, labels)

                # Setup video writer on first frame
                if args.output and video_writer is None:
                    video_writer = _create_video_writer(args.output, frame)

                if video_writer is not None:
                    video_writer.write(annotated)

                if args.display:
                    _show_frame(annotated)
                    if _check_quit():
                        break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if mot_file is not None:
            mot_file.close()
        if video_writer is not None:
            video_writer.release()
        if args.display:
            _close_display()

    return 0


def _resolve_video_output_path(path: Path) -> Path:
    """Resolve video output path, handling directories.

    If path is an existing directory, generates 'output.mp4' inside it.
    If path has no extension, adds '.mp4'.
    """
    if path.is_dir():
        return path / "output.mp4"
    if not path.suffix:
        return path.with_suffix(".mp4")
    return path


def _check_output_writable(path: Path, *, overwrite: bool = False) -> None:
    """Raise FileExistsError if path exists and overwrite is False.

    Args:
        path: Path to check.
        overwrite: If True, allow overwriting existing files.

    Raises:
        FileExistsError: If path exists and overwrite is False.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file '{path}' already exists. Use --overwrite to replace."
        )


def _detect_device() -> str:
    """Auto-detect the best available device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _create_model(args: argparse.Namespace):
    """Load detection model via inference-models."""
    try:
        from inference_models import AutoModel
    except ImportError as e:
        print(
            "Error: inference-models is required for model-based detection.\n"
            "Install with: pip install 'trackers[detection]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    device = _detect_device() if args.model_device == "auto" else args.model_device

    model = AutoModel.from_pretrained(
        args.model,
        api_key=args.model_api_key,
        device=device,
    )
    return model


def _run_model_inference(model, frame: np.ndarray, confidence: float) -> sv.Detections:
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


def _create_tracker(args: argparse.Namespace) -> BaseTracker:
    """Create tracker instance from registry with CLI parameters."""
    info = BaseTracker._lookup_tracker(args.tracker)
    if info is None:
        available = ", ".join(BaseTracker._registered_trackers())
        raise ValueError(f"Unknown tracker: '{args.tracker}'. Available: {available}")

    # Build kwargs from CLI args
    kwargs = {}
    for param_name in info.parameters:
        dest_name = f"tracker_{param_name}"
        if hasattr(args, dest_name):
            value = getattr(args, dest_name)
            if value is not None:
                kwargs[param_name] = value

    return info.tracker_class(**kwargs)


def _load_mot_detections(path: Path) -> dict[int, "MOTFrameData"]:
    """Load pre-computed detections from MOT format file."""
    from trackers.eval.io import load_mot_file

    return load_mot_file(path)


def _mot_frame_to_detections(frame_data: "MOTFrameData") -> sv.Detections:
    """Convert MOTFrameData to sv.Detections.

    Args:
        frame_data: MOT frame data containing boxes in xywh format.

    Returns:
        Detections object with boxes converted to xyxy format.
    """
    return sv.Detections(
        xyxy=sv.xywh_to_xyxy(frame_data.boxes),
        confidence=frame_data.confidences,
        class_id=frame_data.classes.astype(int),
    )


def create_frame_generator(source: str) -> Iterator[tuple[int, np.ndarray]]:
    """Create frame generator from video, webcam, stream, or image directory.

    Args:
        source: Video file path, webcam index (as string), RTSP URL,
            or directory containing images.

    Yields:
        Tuple of (frame_index, frame) where frame_index is 1-based.
    """
    try:
        import cv2
    except ImportError as e:
        print(
            "Error: opencv-python is required for video processing.\n"
            "Install with: pip install 'trackers[cli]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    source_path = Path(source)

    # Check if it's a directory of images
    if source_path.is_dir():
        yield from _generate_from_directory(source_path)
        return

    # Try to parse as webcam index
    video_source: str | int = int(source) if source.isdigit() else source

    # Open video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source}")

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            yield frame_idx, frame
    finally:
        cap.release()


def _generate_from_directory(directory: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Generate frames from a directory of images."""
    import cv2

    # Find image files and sort them
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = sorted(
        f for f in directory.iterdir() if f.suffix.lower() in extensions
    )

    if not image_files:
        raise ValueError(f"No image files found in directory: {directory}")

    for idx, img_path in enumerate(image_files, start=1):
        frame = cv2.imread(str(img_path))
        if frame is not None:
            yield idx, frame


COLOR_PALETTE = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00",
])


def _create_annotators(
    args: argparse.Namespace,
) -> tuple[list, sv.LabelAnnotator | None]:
    """Create list of supervision annotators based on args.

    Returns:
        Tuple of (annotators list, label_annotator or None).
        Label annotator is separate because it needs custom labels per frame.
    """
    annotators: list = []
    label_annotator: sv.LabelAnnotator | None = None

    if args.show_boxes:
        annotators.append(sv.BoxAnnotator(
            color=COLOR_PALETTE,
            color_lookup=sv.ColorLookup.TRACK,
        ))

    if args.show_masks:
        annotators.append(sv.MaskAnnotator(
            color=COLOR_PALETTE,
            color_lookup=sv.ColorLookup.TRACK,
        ))

    if args.show_labels or args.show_ids or args.show_confidence:
        label_annotator = sv.LabelAnnotator(
            color=COLOR_PALETTE,
            text_color=sv.Color.BLACK,
            text_position=sv.Position.TOP_LEFT,
            color_lookup=sv.ColorLookup.TRACK,
        )

    return annotators, label_annotator


def _generate_labels(
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


def _write_mot_frame(f, frame_idx: int, detections: sv.Detections) -> None:
    """Write detections for a frame in MOT format."""
    if len(detections) == 0:
        return

    for i in range(len(detections)):
        # MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
        x1, y1, x2, y2 = detections.xyxy[i]
        w = x2 - x1
        h = y2 - y1

        track_id = (
            int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
        )
        conf = (
            float(detections.confidence[i])
            if detections.confidence is not None
            else -1.0
        )

        f.write(
            f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},"
            f"-1,-1,-1\n"
        )


def _create_video_writer(output_path: Path, frame: np.ndarray):
    """Create OpenCV VideoWriter for output."""
    import cv2

    resolved = _resolve_video_output_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]

    return cv2.VideoWriter(str(resolved), fourcc, 30.0, (w, h))


def _show_frame(frame: np.ndarray) -> None:
    """Display frame in window."""
    import cv2

    cv2.imshow("Tracking", frame)


def _check_quit() -> bool:
    """Check if user pressed 'q' to quit."""
    import cv2

    return cv2.waitKey(1) & 0xFF == ord("q")


def _close_display() -> None:
    """Close display window."""
    import cv2

    cv2.destroyAllWindows()
