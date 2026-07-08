#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rfdetr
import supervision as sv
from rfdetr.assets.coco_classes import COCO_CLASSES
from supervision.assets import VideoAssets, download_assets

from trackers import BoTSORTTracker, ByteTrackTracker, CBIoUTracker, OCSORTTracker, SORTTracker, frames_from_source
from trackers.core.base import BaseTracker

SOURCE: Path | None = None
OUTPUT_DIR = Path("outputs/rfdetr-all-trackers")
DETECTIONS_JSON = OUTPUT_DIR / "detections.json"
TRACKING_JSON = OUTPUT_DIR / "tracking.json"
MODEL_CLASS = "RFDETRNano"
CONFIDENCE = 0.2
CLASSES: tuple[str | int, ...] | None = ("person",)
MAX_FRAMES: int | None = None
FRAME_AWARE_TRACKERS = {"botsort"}
TRACKER_CLASSES: dict[str, type[BaseTracker]] = {
    "sort": SORTTracker,
    "bytetrack": ByteTrackTracker,
    "ocsort": OCSORTTracker,
    "botsort": BoTSORTTracker,
    "cbiou": CBIoUTracker,
}
TRACKERS = tuple(TRACKER_CLASSES)
TRACKER_KWARGS: dict[str, dict[str, object]] = {
    "sort": {
        "lost_track_buffer": 90,
        "track_activation_threshold": 0.2,
        "minimum_iou_threshold": 0.1,
        "minimum_consecutive_frames": 2,
    },
    "bytetrack": {
        "lost_track_buffer": 120,
        "track_activation_threshold": 0.2,
        "minimum_iou_threshold": 0.08,
        "high_conf_det_threshold": 0.2,
        "minimum_consecutive_frames": 2,
    },
    "ocsort": {
        "lost_track_buffer": 120,
        "minimum_iou_threshold": 0.08,
        "high_conf_det_threshold": 0.2,
        "minimum_consecutive_frames": 2,
    },
    "botsort": {
        "lost_track_buffer": 120,
        "track_activation_threshold": 0.2,
        "minimum_iou_threshold_first_assoc": 0.08,
        "minimum_iou_threshold_second_assoc": 0.18,
        "minimum_iou_threshold_unconfirmed_assoc": 0.1,
        "high_conf_det_threshold": 0.2,
        "minimum_consecutive_frames": 2,
    },
    "cbiou": {
        "lost_track_buffer": 120,
        "track_activation_threshold": 0.2,
        "minimum_iou_threshold_first_assoc": 0.08,
        "minimum_iou_threshold_second_assoc": 0.18,
        "minimum_iou_threshold_unconfirmed_assoc": 0.1,
        "high_conf_det_threshold": 0.2,
        "minimum_consecutive_frames": 2,
    },
}

COLOR_PALETTE = sv.ColorPalette.from_hex(
    [
        "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff", "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
    ]
)


def _resolve_source(source: Path | None) -> Path:
    if source is not None:
        return source

    return Path(download_assets(VideoAssets.PEOPLE_WALKING)).resolve()


def _init_model(model_class_name: str) -> tuple[Any, list[str]]:
    return getattr(rfdetr, model_class_name)(), _coco_class_names(COCO_CLASSES)


def _coco_class_names(coco_classes: Mapping[int, str] | Sequence[str]) -> list[str]:
    if isinstance(coco_classes, Mapping):
        return [coco_classes.get(class_id, str(class_id)) for class_id in range(max(coco_classes) + 1)]
    return list(coco_classes)


def _video_info_from_path(path: Path) -> sv.VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video source: {path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    return sv.VideoInfo(width=width, height=height, fps=fps, total_frames=total_frames)


def _resolve_class_filter(classes: Iterable[str | int] | None, class_names: Iterable[str]) -> list[int] | None:
    if not classes:
        return None

    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    class_filter: list[int] = []
    for item in classes:
        if isinstance(item, int):
            class_filter.append(item)
        elif item in name_to_id:
            class_filter.append(name_to_id[item])
        else:
            print(f"Warning: class {item!r} not found in model class list, skipping.")

    return class_filter if class_filter else None


def _run_model(model: Any, frame: np.ndarray, *, confidence: float, class_filter: list[int] | None) -> sv.Detections:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = model.predict(frame_rgb, threshold=confidence)
    if len(detections) == 0:
        return detections

    if class_filter is not None and len(detections) > 0 and detections.class_id is not None:
        detections = detections[np.isin(detections.class_id, class_filter)]

    return detections


def _format_labels(detections: sv.Detections, class_names: list[str]) -> list[str]:
    labels: list[str] = []
    for idx in range(len(detections)):
        parts: list[str] = []

        if detections.tracker_id is not None:
            parts.append(f"#{int(detections.tracker_id[idx])}")

        if detections.class_id is not None:
            class_id = int(detections.class_id[idx])
            parts.append(class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id))

        if detections.confidence is not None:
            parts.append(f"{detections.confidence[idx]:.2f}")

        labels.append(" ".join(parts))
    return labels


def _detections_to_json(detections: sv.Detections, class_names: list[str]) -> list[dict[str, object]]:
    rows = []
    for idx in range(len(detections)):
        class_id = int(detections.class_id[idx]) if detections.class_id is not None else None
        row = {
            "xyxy": [float(v) for v in detections.xyxy[idx]],
            "confidence": float(detections.confidence[idx]) if detections.confidence is not None else None,
            "class_id": class_id,
            "class_name": class_names[class_id] if class_id is not None and 0 <= class_id < len(class_names) else None,
        }
        if detections.tracker_id is not None:
            row["tracker_id"] = int(detections.tracker_id[idx])
        rows.append(row)
    return rows


def _annotate_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    *,
    class_names: list[str],
    box_annotator: sv.BoxAnnotator,
    label_annotator: sv.LabelAnnotator,
    trace_annotator: sv.TraceAnnotator,
) -> np.ndarray:
    visible = detections[detections.tracker_id != -1] if detections.tracker_id is not None else detections
    annotated = trace_annotator.annotate(frame.copy(), visible)
    annotated = box_annotator.annotate(annotated, visible)
    return label_annotator.annotate(annotated, visible, labels=_format_labels(visible, class_names))


def main() -> int:
    source = _resolve_source(SOURCE)
    video_info = _video_info_from_path(source)
    model, class_names = _init_model(MODEL_CLASS)
    class_filter = _resolve_class_filter(CLASSES, class_names)

    trackers = {name: TRACKER_CLASSES[name](**TRACKER_KWARGS[name]) for name in TRACKERS}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    box_annotator = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(
        color=COLOR_PALETTE,
        text_color=sv.Color.BLACK,
        text_position=sv.Position.TOP_LEFT,
        color_lookup=sv.ColorLookup.TRACK,
    )
    trace_annotators = {
        name: sv.TraceAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.TRACK) for name in trackers
    }
    detection_frames = []
    tracking_frames = {name: [] for name in TRACKERS}

    sinks = {
        name: sv.VideoSink(str(OUTPUT_DIR / f"{source.stem}-{name}.mp4"), video_info=video_info) for name in trackers
    }

    with ExitStack() as stack:
        open_sinks = {name: stack.enter_context(sink) for name, sink in sinks.items()}
        for frame_idx, frame in frames_from_source(source):
            if MAX_FRAMES is not None and frame_idx > MAX_FRAMES:
                break

            detections = _run_model(model, frame, confidence=CONFIDENCE, class_filter=class_filter)
            detection_frames.append({"frame": frame_idx, "detections": _detections_to_json(detections, class_names)})
            for name, tracker in trackers.items():
                tracked = tracker.update(detections, frame if name in FRAME_AWARE_TRACKERS else None)
                tracking_frames[name].append({"frame": frame_idx, "tracks": _detections_to_json(tracked, class_names)})
                annotated = _annotate_frame(
                    frame,
                    tracked,
                    class_names=class_names,
                    box_annotator=box_annotator,
                    label_annotator=label_annotator,
                    trace_annotator=trace_annotators[name],
                )
                open_sinks[name].write_frame(annotated)

            print(f"\rProcessed frame {frame_idx}", end="", flush=True)

    DETECTIONS_JSON.write_text(
        json.dumps(
            {
                "source": str(source),
                "model_class": MODEL_CLASS,
                "confidence": CONFIDENCE,
                "classes": list(CLASSES) if CLASSES is not None else None,
                "frames": detection_frames,
            },
            indent=2,
        )
    )
    TRACKING_JSON.write_text(
        json.dumps(
            {
                "source": str(source),
                "model_class": MODEL_CLASS,
                "confidence": CONFIDENCE,
                "classes": list(CLASSES) if CLASSES is not None else None,
                "tracker_parameters": TRACKER_KWARGS,
                "trackers": tracking_frames,
            },
            indent=2,
        )
    )
    print(f"\nWrote {len(trackers)} videos, {DETECTIONS_JSON}, and {TRACKING_JSON} to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
