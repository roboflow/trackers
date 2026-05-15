# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.core.mcbyte.mask_manager import MaskManager
from trackers.core.mcbyte.masks import (
    DummyBoxMaskGenerator,
    DummyIdentityMaskPropagator,
    TrackletSnapshot,
)


def _make_frame(h: int = 100, w: int = 120) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_dummy_box_mask_generator_returns_expected_shape() -> None:
    generator = DummyBoxMaskGenerator()
    frame = _make_frame()

    output = generator.generate(
        frame=frame,
        tracklets=[
            TrackletSnapshot(
                tracker_id=7,
                xyxy=np.array([10, 20, 30, 40], dtype=np.float32),
            )
        ],
    )

    assert output.masks is not None
    assert output.masks.shape == (1, 100, 120)
    assert output.tracklet_mask_dict == {7: 0}


def test_dummy_box_mask_generator_fills_detection_box() -> None:
    generator = DummyBoxMaskGenerator()
    frame = _make_frame()

    output = generator.generate(
        frame=frame,
        tracklets=[
            TrackletSnapshot(
                tracker_id=7,
                xyxy=np.array([10, 20, 30, 40], dtype=np.float32),
            )
        ],
    )

    assert output.masks is not None
    assert output.masks[0, 20:40, 10:30].all()
    assert not output.masks[0, :10, :10].any()
    assert output.masks.sum() == 20 * 20


def test_mask_manager_returns_none_without_previous_inputs() -> None:
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
    )

    output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=None,
        previous_tracklets=[],
    )

    assert output is None


def test_mask_manager_generates_masks_from_previous_frame_inputs() -> None:
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=None,
    )

    output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            TrackletSnapshot(
                tracker_id=3,
                xyxy=np.array([5, 6, 25, 30], dtype=np.float32),
            )
        ],
    )

    assert output is not None
    assert output.masks is not None
    assert output.tracklet_mask_dict == {3: 0}
    assert output.masks.shape == (1, 100, 120)
    assert output.masks.sum() == 20 * 24


def test_mask_manager_uses_propagator_after_initialization() -> None:
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
    )

    previous_tracklets = [
        TrackletSnapshot(
            tracker_id=3,
            xyxy=np.array([5, 6, 25, 30], dtype=np.float32),
        )
    ]

    first_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=previous_tracklets,
    )

    second_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            TrackletSnapshot(
                tracker_id=99,
                xyxy=np.array([50, 50, 70, 70], dtype=np.float32),
            )
        ],
    )

    assert first_output is not None
    assert second_output is not None
    assert second_output.tracklet_mask_dict == first_output.tracklet_mask_dict


def test_mask_manager_reset_clears_state() -> None:
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
    )

    output_before_reset = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            TrackletSnapshot(
                tracker_id=3,
                xyxy=np.array([5, 6, 25, 30], dtype=np.float32),
            )
        ],
    )

    manager.reset()

    output_after_reset = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            TrackletSnapshot(
                tracker_id=9,
                xyxy=np.array([40, 40, 60, 60], dtype=np.float32),
            )
        ],
    )

    assert output_before_reset is not None
    assert output_after_reset is not None
    assert output_after_reset.tracklet_mask_dict == {9: 0}
