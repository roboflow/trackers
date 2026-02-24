# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Gradio app for the trackers library — run object tracking on uploaded videos."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from inference_models import AutoModel
from tqdm import tqdm

from trackers import ByteTrackTracker, SORTTracker, frames_from_source

MAX_DURATION_SECONDS = 30

MODELS = [
    "rfdetr-nano",
    "rfdetr-small",
    "rfdetr-medium",
    "rfdetr-large",
    "rfdetr-seg-nano",
    "rfdetr-seg-small",
    "rfdetr-seg-medium",
    "rfdetr-seg-large",
]

TRACKERS = ["bytetrack", "sort"]

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "truck",
    "cat",
    "dog",
    "sports ball",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {len(MODELS)} models on {DEVICE}...")
LOADED_MODELS = {}
for model_id in MODELS:
    print(f"  Loading {model_id}...")
    LOADED_MODELS[model_id] = AutoModel.from_pretrained(model_id, device=DEVICE)
print("All models loaded.")

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

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def _init_annotators(
    show_boxes: bool = False,
    show_masks: bool = False,
    show_labels: bool = False,
    show_ids: bool = False,
    show_confidence: bool = False,
) -> tuple[list, sv.LabelAnnotator | None]:
    """Initialize supervision annotators based on display options."""
    annotators: list = []
    label_annotator: sv.LabelAnnotator | None = None

    if show_masks:
        annotators.append(
            sv.MaskAnnotator(
                color=COLOR_PALETTE,
                color_lookup=sv.ColorLookup.TRACK,
            )
        )

    if show_boxes:
        annotators.append(
            sv.BoxAnnotator(
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
    """Generate label strings for each detection."""
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


VIDEO_EXAMPLES = [
    [
        "https://storage.googleapis.com/com-roboflow-marketing/supervision/video-examples/bikes-1280x720-1.mp4",
        "rfdetr-small",
        "bytetrack",
        0.2,
        30,
        0.3,
        3,
        0.1,
        0.6,
        [],
        True,
        True,
        False,
        False,
        True,
        False,
    ],
    [
        "https://storage.googleapis.com/com-roboflow-marketing/supervision/video-examples/bikes-1280x720-2.mp4",
        "rfdetr-seg-small",
        "sort",
        0.2,
        30,
        0.3,
        3,
        0.3,
        0.6,
        [],
        True,
        True,
        False,
        False,
        True,
        True,
    ],
    [
        "https://storage.googleapis.com/com-roboflow-marketing/supervision/video-examples/cars-1280x720-1.mp4",
        "rfdetr-small",
        "bytetrack",
        0.2,
        30,
        0.3,
        3,
        0.1,
        0.6,
        ["car"],
        True,
        True,
        False,
        True,
        False,
        False,
    ],
    [
        "https://storage.googleapis.com/com-roboflow-marketing/supervision/video-examples/jets-1280x720-1.mp4",
        "rfdetr-small",
        "bytetrack",
        0.2,
        30,
        0.3,
        3,
        0.1,
        0.6,
        [],
        True,
        True,
        False,
        False,
        False,
        False,
    ],
    [
        "https://storage.googleapis.com/com-roboflow-marketing/supervision/video-examples/jets-1280x720-2.mp4",
        "rfdetr-seg-small",
        "bytetrack",
        0.2,
        30,
        0.3,
        3,
        0.1,
        0.6,
        [],
        True,
        True,
        False,
        False,
        True,
        False,
    ],
    [
        "https://storage.googleapis.com/com-roboflow-marketing/supervision/video-examples/vehicles-1280x720.mp4",
        "rfdetr-small",
        "bytetrack",
        0.2,
        30,
        0.3,
        3,
        0.1,
        0.6,
        [],
        True,
        True,
        True,
        False,
        True,
        False,
    ],
]


def _get_video_info(path: str) -> tuple[float, int]:
    """Return video duration in seconds and frame count using OpenCV."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise gr.Error("Could not open the uploaded video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        raise gr.Error("Could not determine video frame rate.")
    return frame_count / fps, frame_count


def _resolve_class_filter(
    classes: list[str] | None,
    class_names: list[str],
) -> list[int] | None:
    """Resolve class names to integer IDs."""
    if not classes:
        return None

    name_to_id = {name: i for i, name in enumerate(class_names)}
    class_filter: list[int] = []
    for name in classes:
        if name in name_to_id:
            class_filter.append(name_to_id[name])
    return class_filter if class_filter else None


def track(
    video_path: str,
    model_id: str,
    tracker_type: str,
    confidence: float,
    lost_track_buffer: int,
    track_activation_threshold: float,
    minimum_consecutive_frames: int,
    minimum_iou_threshold: float,
    high_conf_det_threshold: float,
    classes: list[str] | None = None,
    show_boxes: bool = True,
    show_ids: bool = True,
    show_labels: bool = False,
    show_confidence: bool = False,
    show_trajectories: bool = False,
    show_masks: bool = False,
    progress=gr.Progress(track_tqdm=True),
) -> str:
    """Run tracking on the uploaded video and return the output path."""
    if video_path is None:
        raise gr.Error("Please upload a video.")

    duration, total_frames = _get_video_info(video_path)
    if duration > MAX_DURATION_SECONDS:
        raise gr.Error(
            f"Video is {duration:.1f}s long. "
            f"Maximum allowed duration is {MAX_DURATION_SECONDS}s."
        )

    detection_model = LOADED_MODELS[model_id]
    class_names = getattr(detection_model, "class_names", [])

    class_filter = _resolve_class_filter(classes, class_names)

    tracker: ByteTrackTracker | SORTTracker
    if tracker_type == "bytetrack":
        tracker = ByteTrackTracker(
            lost_track_buffer=lost_track_buffer,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
            high_conf_det_threshold=high_conf_det_threshold,
        )
    else:
        tracker = SORTTracker(
            lost_track_buffer=lost_track_buffer,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
        )
    tracker.reset()

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

    tmp_dir = tempfile.mkdtemp()
    output_path = str(Path(tmp_dir) / "output.mp4")

    video_info = sv.VideoInfo.from_video_path(video_path)

    frame_gen = frames_from_source(video_path)

    with sv.VideoSink(output_path, video_info=video_info) as sink:
        for frame_idx, frame in tqdm(
            frame_gen, total=total_frames, desc="Processing video..."
        ):
            predictions = detection_model(frame)
            if predictions:
                detections = predictions[0].to_supervision()

                if len(detections) > 0 and detections.confidence is not None:
                    mask = detections.confidence >= confidence
                    detections = detections[mask]

                if class_filter is not None and len(detections) > 0:
                    mask = np.isin(detections.class_id, class_filter)
                    detections = detections[mask]
            else:
                detections = sv.Detections.empty()

            tracked = tracker.update(detections)

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
                annotated = label_annotator.annotate(annotated, labeled, labels=labels)

            sink.write_frame(annotated)

    return output_path


with gr.Blocks(title="Trackers Playground 🔥") as demo:
    gr.Markdown(
        "# Trackers Playground 🔥\n\n"
        "Upload a video, detect COCO objects with "
        "[RF-DETR](https://github.com/roboflow-ai/rf-detr) and track them with "
        "[Trackers](https://github.com/roboflow/trackers)."
    )

    with gr.Row():
        input_video = gr.Video(label="Input Video")
        output_video = gr.Video(label="Tracked Video")

    track_btn = gr.Button(value="Track", variant="primary")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=MODELS,
            value="rfdetr-small",
            label="Detection Model",
        )
        tracker_dropdown = gr.Dropdown(
            choices=TRACKERS,
            value="bytetrack",
            label="Tracker",
        )

    with gr.Accordion("Configuration", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model")
                confidence_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.05,
                    label="Detection Confidence",
                    info="Minimum score for a detection to be kept.",
                )
                class_filter = gr.CheckboxGroup(
                    choices=COCO_CLASSES,
                    value=[],
                    label="Filter Classes",
                    info="Only track selected classes. None selected means all.",
                )

            with gr.Column():
                gr.Markdown("### Tracker")
                lost_track_buffer_slider = gr.Slider(
                    minimum=1,
                    maximum=120,
                    value=30,
                    step=1,
                    label="Lost Track Buffer",
                    info="Frames to keep a lost track before removing it.",
                )
                track_activation_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.05,
                    label="Track Activation Threshold",
                    info="Minimum score for a track to be activated.",
                )
                min_consecutive_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=2,
                    step=1,
                    label="Minimum Consecutive Frames",
                    info="Detections needed before a track is confirmed.",
                )
                min_iou_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.05,
                    label="Minimum IoU Threshold",
                    info="Overlap required to match a detection to a track.",
                )
                high_conf_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
                    step=0.05,
                    label="High Confidence Detection Threshold",
                    info="Detections above this are matched first (ByteTrack only).",
                )

            with gr.Column():
                gr.Markdown("### Visualization")
                show_boxes_checkbox = gr.Checkbox(
                    value=True,
                    label="Show Boxes",
                    info="Draw bounding boxes around detections.",
                )
                show_ids_checkbox = gr.Checkbox(
                    value=True,
                    label="Show IDs",
                    info="Display track ID for each object.",
                )
                show_labels_checkbox = gr.Checkbox(
                    value=False,
                    label="Show Labels",
                    info="Display class name for each detection.",
                )
                show_confidence_checkbox = gr.Checkbox(
                    value=False,
                    label="Show Confidence",
                    info="Display detection confidence score.",
                )
                show_trajectories_checkbox = gr.Checkbox(
                    value=False,
                    label="Show Trajectories",
                    info="Draw motion path for each tracked object.",
                )
                show_masks_checkbox = gr.Checkbox(
                    value=False,
                    label="Show Masks",
                    info="Draw segmentation masks (seg models only).",
                )

    gr.Examples(
        examples=VIDEO_EXAMPLES,
        fn=track,
        cache_examples=True,
        inputs=[
            input_video,
            model_dropdown,
            tracker_dropdown,
            confidence_slider,
            lost_track_buffer_slider,
            track_activation_slider,
            min_consecutive_slider,
            min_iou_slider,
            high_conf_slider,
            class_filter,
            show_boxes_checkbox,
            show_ids_checkbox,
            show_labels_checkbox,
            show_confidence_checkbox,
            show_trajectories_checkbox,
            show_masks_checkbox,
        ],
        outputs=output_video,
    )

    track_btn.click(
        fn=track,
        inputs=[
            input_video,
            model_dropdown,
            tracker_dropdown,
            confidence_slider,
            lost_track_buffer_slider,
            track_activation_slider,
            min_consecutive_slider,
            min_iou_slider,
            high_conf_slider,
            class_filter,
            show_boxes_checkbox,
            show_ids_checkbox,
            show_labels_checkbox,
            show_confidence_checkbox,
            show_trajectories_checkbox,
            show_masks_checkbox,
        ],
        outputs=output_video,
    )

if __name__ == "__main__":
    demo.launch()
