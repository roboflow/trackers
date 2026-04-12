# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _matrix_from_value(value: object) -> np.ndarray | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 homography matrix, got {matrix.shape}")
    return matrix


def _matrix_to_value(matrix: np.ndarray | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return np.asarray(matrix, dtype=np.float64).tolist()


@dataclass(frozen=True)
class PitchDimensions:
    """Physical pitch dimensions in meters."""

    length_m: float = 105.0
    width_m: float = 68.0

    def to_dict(self) -> dict[str, float]:
        return {
            "length_m": self.length_m,
            "width_m": self.width_m,
        }


@dataclass(slots=True)
class CalibrationFrame:
    """Calibration output for a single video frame.

    The canonical convention in this scaffold is that pitch coordinates are in
    metric space using a top-left pitch origin:
    - x grows along the pitch length
    - y grows across the pitch width
    """

    frame_idx: int
    timestamp_s: float
    image_to_pitch: np.ndarray | None = None
    pitch_to_image: np.ndarray | None = None
    confidence: float | None = None
    provider: str | None = None
    pitch_dimensions: PitchDimensions = field(default_factory=PitchDimensions)
    camera_parameters: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image_to_pitch = _matrix_from_value(self.image_to_pitch)
        self.pitch_to_image = _matrix_from_value(self.pitch_to_image)

    @property
    def has_homography(self) -> bool:
        return self.image_to_pitch is not None and self.pitch_to_image is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_s": self.timestamp_s,
            "image_to_pitch": _matrix_to_value(self.image_to_pitch),
            "pitch_to_image": _matrix_to_value(self.pitch_to_image),
            "confidence": self.confidence,
            "provider": self.provider,
            "pitch_dimensions": self.pitch_dimensions.to_dict(),
            "camera_parameters": self.camera_parameters,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationFrame":
        pitch_data = data.get("pitch_dimensions", {})
        return cls(
            frame_idx=int(data["frame_idx"]),
            timestamp_s=float(data["timestamp_s"]),
            image_to_pitch=data.get("image_to_pitch"),
            pitch_to_image=data.get("pitch_to_image"),
            confidence=(
                None
                if data.get("confidence") is None
                else float(data.get("confidence"))
            ),
            provider=data.get("provider"),
            pitch_dimensions=PitchDimensions(
                length_m=float(pitch_data.get("length_m", 105.0)),
                width_m=float(pitch_data.get("width_m", 68.0)),
            ),
            camera_parameters=dict(data.get("camera_parameters", {})),
            diagnostics=dict(data.get("diagnostics", {})),
        )


@dataclass(frozen=True)
class TrackProjection:
    """Pitch-space position for a tracked object on a single frame."""

    frame_idx: int
    track_id: int
    image_x: float
    image_y: float
    pitch_x_m: float
    pitch_y_m: float
    pitch_x_norm: float
    pitch_y_norm: float
    in_pitch_bounds: bool
    timestamp_s: float | None = None
    calibration_confidence: float | None = None
    source_confidence: float | None = None
    provider: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "track_id": self.track_id,
            "image_x": self.image_x,
            "image_y": self.image_y,
            "pitch_x_m": self.pitch_x_m,
            "pitch_y_m": self.pitch_y_m,
            "pitch_x_norm": self.pitch_x_norm,
            "pitch_y_norm": self.pitch_y_norm,
            "in_pitch_bounds": self.in_pitch_bounds,
            "timestamp_s": self.timestamp_s,
            "calibration_confidence": self.calibration_confidence,
            "source_confidence": self.source_confidence,
            "provider": self.provider,
        }
