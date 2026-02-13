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

VIDEO_EXAMPLES = [
    [
        "https://storage.googleapis.com/com-roboflow-marketing/supervision/video-examples/bikes-1280x720-1.mp4",
        "rfdetr-nano",
        "bytetrack",
        0.5,
        30,
        0.7,
        2,
        0.1,
        0.6,
    ],
    [
        "https://storage.googleapis.com/com-roboflow-marketing/supervision/video-examples/bikes-1280x720-2.mp4",
        "rfdetr-small",
        "sort",
        0.4,
        30,
        0.25,
        3,
        0.3,
        0.6,
    ],
    [
        "https://storage.googleapis.com/com-roboflow-marketing/supervision/video-examples/football-1280x720-1.mp4",
        "rfdetr-medium",
        "bytetrack",
        0.3,
        45,
        0.6,
        2,
        0.15,
        0.5,
    ],
    [
        "https://storage.googleapis.com/com-roboflow-marketing/supervision/video-examples/cars-1280x720-1.mp4",
        "rfdetr-nano",
        "bytetrack",
        0.5,
        30,
        0.7,
        2,
        0.1,
        0.6,
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
    raw_output = str(Path(tmp_dir) / "raw_output.mp4")
    final_output = str(Path(tmp_dir) / "output.mp4")

    cmd = [
        "trackers",
        "track",
        "--source",
        video_path,
        "--output",
        raw_output,
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

    if tracker == "bytetrack":
        cmd += ["--tracker.high_conf_det_threshold", str(high_conf_det_threshold)]

    if model.startswith("rfdetr-seg"):
        cmd += ["--show-masks"]

    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        raise gr.Error(f"Tracking failed:\n{result.stderr[-500:]}")

    # Convert mp4v → H.264 for browser playback
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        raw_output,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-y",
        final_output,
    ]
    ff_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)  # noqa: S603
    if ff_result.returncode != 0:
        return raw_output

    return final_output


with gr.Blocks(title="Trackers") as demo:
    gr.Markdown(
        "# Roboflow Trackers\n"
        "Upload a video, pick a detection model and tracker, then download "
        "the tracked result. Videos are limited to 30 seconds.\n\n"
        "Powered by [roboflow/trackers]"
        "(https://github.com/roboflow/trackers)."
    )

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video")
            model_dropdown = gr.Dropdown(
                choices=MODELS,
                value="rfdetr-nano",
                label="Detection Model",
            )
            tracker_dropdown = gr.Dropdown(
                choices=TRACKERS,
                value="bytetrack",
                label="Tracker",
            )

            with gr.Accordion("Advanced Parameters", open=False):
                confidence_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Detection Confidence",
                )
                lost_track_buffer_slider = gr.Slider(
                    minimum=1,
                    maximum=120,
                    value=30,
                    step=1,
                    label="Lost Track Buffer (frames)",
                )
                track_activation_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="Track Activation Threshold",
                )
                min_consecutive_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=2,
                    step=1,
                    label="Minimum Consecutive Frames",
                )
                min_iou_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.05,
                    label="Minimum IoU Threshold",
                )
                high_conf_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
                    step=0.05,
                    label="High Confidence Detection Threshold (ByteTrack only)",
                )

            track_btn = gr.Button("Track", variant="primary")

        with gr.Column():
            output_video = gr.Video(label="Tracked Video")

    gr.Examples(
        fn=track,
        examples=VIDEO_EXAMPLES,
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
        ],
        outputs=output_video,
    )

if __name__ == "__main__":
    demo.launch()
