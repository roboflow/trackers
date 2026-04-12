# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from trackers.calibration.types import CalibrationFrame, TrackProjection


def write_manifest(path: str | Path, data: dict[str, object]) -> Path:
    """Write a JSON manifest file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return output_path


def write_calibration_jsonl(
    path: str | Path,
    frames: Iterable[CalibrationFrame],
) -> Path:
    """Write calibration frames as JSON Lines."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(json.dumps(frame.to_dict(), sort_keys=True))
            handle.write("\n")
    return output_path


def write_homography_jsonl(
    path: str | Path,
    frames: Iterable[CalibrationFrame],
) -> Path:
    """Write only per-frame homography data as JSON Lines."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            payload = {
                "frame_idx": frame.frame_idx,
                "timestamp_s": frame.timestamp_s,
                "image_to_pitch": (
                    None
                    if frame.image_to_pitch is None
                    else frame.image_to_pitch.tolist()
                ),
                "pitch_to_image": (
                    None
                    if frame.pitch_to_image is None
                    else frame.pitch_to_image.tolist()
                ),
                "confidence": frame.confidence,
                "provider": frame.provider,
                "pitch_dimensions": frame.pitch_dimensions.to_dict(),
                "diagnostics": frame.diagnostics,
            }
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
    return output_path


def load_calibration_jsonl(path: str | Path) -> list[CalibrationFrame]:
    """Load calibration frames from JSON Lines."""
    input_path = Path(path)
    frames: list[CalibrationFrame] = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            frames.append(CalibrationFrame.from_dict(json.loads(stripped)))
    return frames


def write_calibration_quality_csv(
    path: str | Path,
    frames: Iterable[CalibrationFrame],
) -> Path:
    """Write frame-level calibration diagnostics as CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_idx",
        "timestamp_s",
        "has_homography",
        "confidence",
        "rep_err",
        "mode",
        "use_ransac",
        "calib_plane",
        "held_from_frame_idx",
        "held_frame_gap",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            diagnostics = frame.diagnostics
            writer.writerow(
                {
                    "frame_idx": frame.frame_idx,
                    "timestamp_s": frame.timestamp_s,
                    "has_homography": frame.has_homography,
                    "confidence": frame.confidence,
                    "rep_err": diagnostics.get("rep_err"),
                    "mode": diagnostics.get("mode"),
                    "use_ransac": diagnostics.get("use_ransac"),
                    "calib_plane": diagnostics.get("calib_plane"),
                    "held_from_frame_idx": diagnostics.get("held_from_frame_idx"),
                    "held_frame_gap": diagnostics.get("held_frame_gap"),
                }
            )
    return output_path


def write_track_projections_csv(
    path: str | Path,
    projections: Iterable[TrackProjection],
) -> Path:
    """Write projected track positions as CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TrackProjection.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for projection in projections:
            writer.writerow(projection.to_dict())
    return output_path
