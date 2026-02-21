# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Gradio app for the trackers library — run object tracking on uploaded videos."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import gradio as gr

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


def _get_video_duration(path: str) -> float:
    """Return video duration in seconds using OpenCV."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise gr.Error("Could not open the uploaded video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        raise gr.Error("Could not determine video frame rate.")
    return frame_count / fps


def track(
    video_path: str,
    model: str,
    tracker: str,
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
) -> str:
    """Run tracking on the uploaded video and return the output path."""
    if video_path is None:
        raise gr.Error("Please upload a video.")

    duration = _get_video_duration(video_path)
    if duration > MAX_DURATION_SECONDS:
        raise gr.Error(
            f"Video is {duration:.1f}s long. "
            f"Maximum allowed duration is {MAX_DURATION_SECONDS}s."
        )

    tmp_dir = tempfile.mkdtemp()
    output_path = str(Path(tmp_dir) / "output.mp4")

    cmd = [
        "trackers",
        "track",
        "--source",
        video_path,
        "--output",
        output_path,
        "--overwrite",
        "--model",
        model,
        "--tracker",
        tracker,
        "--model.confidence",
        str(confidence),
        "--tracker.lost_track_buffer",
        str(lost_track_buffer),
        "--tracker.track_activation_threshold",
        str(track_activation_threshold),
        "--tracker.minimum_consecutive_frames",
        str(minimum_consecutive_frames),
        "--tracker.minimum_iou_threshold",
        str(minimum_iou_threshold),
    ]

    # ByteTrack extra param
    if tracker == "bytetrack":
        cmd += ["--tracker.high_conf_det_threshold", str(high_conf_det_threshold)]

    if classes:
        cmd += ["--classes", ",".join(classes)]

    if show_boxes:
        cmd += ["--show-boxes"]
    else:
        cmd += ["--no-boxes"]
    if show_ids:
        cmd += ["--show-ids"]
    if show_labels:
        cmd += ["--show-labels"]
    if show_confidence:
        cmd += ["--show-confidence"]
    if show_trajectories:
        cmd += ["--show-trajectories"]
    if show_masks:
        cmd += ["--show-masks"]

    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise gr.Error(f"Tracking failed:\n{result.stderr[-500:]}")

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
