# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from trackers.calibration.types import CalibrationFrame


class HoldLastCalibration:
    """Carry forward the last trustworthy calibration across short gaps.

    This is intentionally conservative: it does not attempt to optimize the
    homography itself, it only prevents short calibration dropouts from causing
    immediate loss of pitch coordinates. A more advanced backend can replace
    this with model-aware smoothing later.
    """

    def __init__(self, max_gap_frames: int = 15, min_confidence: float = 0.4) -> None:
        self.max_gap_frames = max_gap_frames
        self.min_confidence = min_confidence
        self._last_good_frame: CalibrationFrame | None = None

    def update(self, frame: CalibrationFrame) -> CalibrationFrame:
        confidence = frame.confidence if frame.confidence is not None else 1.0
        is_good = frame.has_homography and confidence >= self.min_confidence

        if is_good:
            self._last_good_frame = frame
            return frame

        if self._last_good_frame is None:
            return frame

        frame_gap = frame.frame_idx - self._last_good_frame.frame_idx
        if frame_gap > self.max_gap_frames:
            return frame

        diagnostics = dict(frame.diagnostics)
        diagnostics.update(
            {
                "held_from_frame_idx": self._last_good_frame.frame_idx,
                "held_frame_gap": frame_gap,
            }
        )
        return replace(
            frame,
            image_to_pitch=self._last_good_frame.image_to_pitch,
            pitch_to_image=self._last_good_frame.pitch_to_image,
            confidence=self._last_good_frame.confidence,
            provider=self._last_good_frame.provider,
            camera_parameters=dict(self._last_good_frame.camera_parameters),
            diagnostics=diagnostics,
        )


def _is_good_frame(frame: CalibrationFrame, min_confidence: float) -> bool:
    confidence = frame.confidence if frame.confidence is not None else 1.0
    return frame.has_homography and confidence >= min_confidence


def _apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    projected = homogeneous @ homography.T
    projected = projected[:, :2] / projected[:, 2:3]
    return projected


def _control_points(frame: CalibrationFrame) -> np.ndarray:
    length_m = frame.pitch_dimensions.length_m
    width_m = frame.pitch_dimensions.width_m
    return np.array(
        [
            [0.0, 0.0],
            [length_m / 2.0, 0.0],
            [length_m, 0.0],
            [0.0, width_m / 2.0],
            [length_m / 2.0, width_m / 2.0],
            [length_m, width_m / 2.0],
            [0.0, width_m],
            [length_m / 2.0, width_m],
            [length_m, width_m],
            [11.0, width_m / 2.0],
            [length_m - 11.0, width_m / 2.0],
        ],
        dtype=np.float64,
    )


def _interpolate_between_frames(
    previous_frame: CalibrationFrame,
    next_frame: CalibrationFrame,
    target_frame: CalibrationFrame,
) -> CalibrationFrame:
    if previous_frame.pitch_to_image is None or next_frame.pitch_to_image is None:
        return target_frame

    total_gap = next_frame.frame_idx - previous_frame.frame_idx
    if total_gap <= 1:
        return target_frame

    alpha = (target_frame.frame_idx - previous_frame.frame_idx) / total_gap
    control_points = _control_points(previous_frame)
    previous_image_points = _apply_homography(
        control_points, previous_frame.pitch_to_image
    )
    next_image_points = _apply_homography(control_points, next_frame.pitch_to_image)
    interpolated_image_points = (
        1.0 - alpha
    ) * previous_image_points + alpha * next_image_points

    pitch_to_image, _ = cv2.findHomography(
        control_points.astype(np.float32),
        interpolated_image_points.astype(np.float32),
        0,
    )
    if pitch_to_image is None:
        return target_frame

    image_to_pitch = np.linalg.inv(pitch_to_image)
    image_to_pitch /= image_to_pitch[-1, -1]
    pitch_to_image /= pitch_to_image[-1, -1]
    previous_confidence = previous_frame.confidence or 0.0
    next_confidence = next_frame.confidence or 0.0
    diagnostics = dict(target_frame.diagnostics)
    diagnostics.update(
        {
            "interpolated_from_frame_idx": previous_frame.frame_idx,
            "interpolated_to_frame_idx": next_frame.frame_idx,
            "interpolation_alpha": alpha,
        }
    )
    return replace(
        target_frame,
        image_to_pitch=image_to_pitch,
        pitch_to_image=pitch_to_image,
        confidence=((1.0 - alpha) * previous_confidence) + (alpha * next_confidence),
        provider=previous_frame.provider,
        camera_parameters={},
        diagnostics=diagnostics,
    )


def interpolate_calibration_gaps(
    frames: list[CalibrationFrame],
    *,
    max_gap_frames: int = 15,
    min_confidence: float = 0.4,
    edge_strategy: str = "hold",
) -> list[CalibrationFrame]:
    """Interpolate homographies between valid neighbors and optionally hold edges."""
    if not frames:
        return []

    interpolated_frames = list(frames)
    valid_indices = [
        index
        for index, frame in enumerate(frames)
        if _is_good_frame(frame, min_confidence=min_confidence)
    ]
    for previous_index, next_index in zip(valid_indices, valid_indices[1:]):
        previous_frame = frames[previous_index]
        next_frame = frames[next_index]
        gap_size = next_frame.frame_idx - previous_frame.frame_idx - 1
        if gap_size <= 0 or gap_size > max_gap_frames:
            continue

        for target_index in range(previous_index + 1, next_index):
            if interpolated_frames[target_index].has_homography:
                continue
            interpolated_frames[target_index] = _interpolate_between_frames(
                previous_frame=previous_frame,
                next_frame=next_frame,
                target_frame=interpolated_frames[target_index],
            )

    if edge_strategy == "hold":
        holder = HoldLastCalibration(
            max_gap_frames=max_gap_frames,
            min_confidence=min_confidence,
        )
        interpolated_frames = [holder.update(frame) for frame in interpolated_frames]

    return interpolated_frames


def fill_calibration_gaps(
    frames: list[CalibrationFrame],
    *,
    max_gap_frames: int = 15,
    min_confidence: float = 0.4,
    mode: str = "hold",
    edge_strategy: str = "hold",
) -> list[CalibrationFrame]:
    """Apply the configured gap-filling strategy to a calibration sequence."""
    if mode == "interpolate":
        return interpolate_calibration_gaps(
            frames,
            max_gap_frames=max_gap_frames,
            min_confidence=min_confidence,
            edge_strategy=edge_strategy,
        )
    if mode != "hold":
        raise ValueError(f"Unsupported calibration smoothing mode: {mode}")
    smoother = HoldLastCalibration(
        max_gap_frames=max_gap_frames,
        min_confidence=min_confidence,
    )
    return [smoother.update(frame) for frame in frames]
