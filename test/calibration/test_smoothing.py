# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.calibration.smoothing import fill_calibration_gaps
from trackers.calibration.types import CalibrationFrame, PitchDimensions


def test_fill_calibration_gaps_interpolates_between_neighbors() -> None:
    pitch = PitchDimensions()
    first = CalibrationFrame(
        frame_idx=1,
        timestamp_s=0.0,
        image_to_pitch=np.eye(3),
        pitch_to_image=np.eye(3),
        confidence=1.0,
        provider="test",
        pitch_dimensions=pitch,
    )
    gap = CalibrationFrame(
        frame_idx=2,
        timestamp_s=1 / 30.0,
        confidence=0.0,
        provider="test",
        pitch_dimensions=pitch,
        diagnostics={"sampled": False},
    )
    translated = np.array(
        [
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    last = CalibrationFrame(
        frame_idx=3,
        timestamp_s=2 / 30.0,
        image_to_pitch=np.linalg.inv(translated),
        pitch_to_image=translated,
        confidence=1.0,
        provider="test",
        pitch_dimensions=pitch,
    )

    frames = fill_calibration_gaps(
        [first, gap, last],
        max_gap_frames=5,
        min_confidence=0.4,
        mode="interpolate",
        edge_strategy="hold",
    )

    assert frames[1].has_homography
    assert frames[1].diagnostics["interpolated_from_frame_idx"] == 1
    assert frames[1].diagnostics["interpolated_to_frame_idx"] == 3
    np.testing.assert_allclose(
        frames[1].pitch_to_image,
        np.array(
            [
                [1.0, 0.0, 5.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        atol=1e-5,
    )
