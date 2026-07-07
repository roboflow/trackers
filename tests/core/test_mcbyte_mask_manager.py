# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.core.mcbyte.mask_manager import MaskManager
from trackers.core.mcbyte.masks import TrackletSnapshot
from trackers.core.mcbyte.masks.dummy import DummyBoxMaskGenerator, DummyIdentityMaskPropagator


def _make_frame(h: int = 100, w: int = 120) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _tracklet(
    tracker_id: int,
    xyxy: tuple[float, float, float, float],
) -> TrackletSnapshot:
    return TrackletSnapshot(
        tracker_id=tracker_id,
        xyxy=np.array(xyxy, dtype=np.float32),
    )


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


def test_mask_manager_returns_none_without_previous_frame_or_tracklets() -> None:
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


def test_mask_manager_returns_none_without_propagator() -> None:
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

    assert output is None


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


def test_mask_manager_propagates_after_initialization_without_visible_tracklets() -> None:
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
    )

    first_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            TrackletSnapshot(3, np.array([5, 6, 25, 30], dtype=np.float32)),
        ],
    )

    second_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[],
    )

    assert first_output is not None
    assert second_output is not None
    assert second_output.tracklet_mask_dict == {3: 0}


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


def test_mask_manager_adds_new_tracklets_after_initialization() -> None:
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
    )

    first_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            TrackletSnapshot(3, np.array([5, 6, 25, 30], dtype=np.float32)),
        ],
    )

    second_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            TrackletSnapshot(3, np.array([5, 6, 25, 30], dtype=np.float32)),
        ],
        new_tracklets=[
            TrackletSnapshot(9, np.array([40, 40, 60, 60], dtype=np.float32)),
        ],
    )

    assert first_output is not None
    assert second_output is not None
    assert second_output.masks is not None
    assert second_output.masks.shape == (2, 100, 120)
    assert second_output.tracklet_mask_dict == {3: 0, 9: 1}


def test_mask_manager_removes_tracklets_after_initialization() -> None:
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
    )

    manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            TrackletSnapshot(3, np.array([5, 6, 25, 30], dtype=np.float32)),
            TrackletSnapshot(9, np.array([40, 40, 60, 60], dtype=np.float32)),
        ],
    )

    output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            TrackletSnapshot(3, np.array([5, 6, 25, 30], dtype=np.float32)),
            TrackletSnapshot(9, np.array([40, 40, 60, 60], dtype=np.float32)),
        ],
        removed_tracklet_ids=[3],
    )

    assert output is not None
    assert output.masks is not None
    assert output.masks.shape == (1, 100, 120)
    assert output.tracklet_mask_dict == {9: 0}


def test_mask_manager_initializes_clean_tracklets_and_delays_occluded_ones() -> None:
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
        mask_creation_bbox_overlap_threshold=0.6,
    )

    output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            _tracklet(1, (10, 10, 30, 50)),  # upper / occluded
            _tracklet(2, (10, 20, 30, 60)),  # lower-bottom occluder
        ],
    )

    assert output is not None
    assert output.tracklet_mask_dict == {2: 0}
    assert manager._pending_tracklet_ids == {1}


def test_mask_manager_late_initializes_when_first_valid_tracklets_appear_later() -> None:
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
    )

    first_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=None,
        previous_tracklets=[],
    )

    second_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[],
    )

    third_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            _tracklet(7, (10, 20, 30, 40)),
        ],
    )

    assert first_output is None
    assert second_output is None
    assert third_output is not None
    assert third_output.tracklet_mask_dict == {7: 0}


def test_mask_manager_initialization_retries_pending_tracklet_with_latest_box() -> None:
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
        mask_creation_bbox_overlap_threshold=0.6,
    )

    first_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            _tracklet(1, (10, 10, 30, 50)),  # occluded
            _tracklet(2, (10, 20, 30, 60)),
        ],
    )

    assert manager._pending_tracklet_ids == {1}
    assert first_output is not None
    assert first_output.tracklet_mask_dict == {2: 0}

    second_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            _tracklet(1, (60, 10, 80, 50)),  # now clean
            _tracklet(2, (10, 20, 30, 60)),
        ],
    )

    assert manager._pending_tracklet_ids == set()
    assert second_output is not None
    assert second_output.tracklet_mask_dict == {2: 0, 1: 1}


def test_mask_manager_delays_and_retries_new_tracklet_after_initialization() -> None:
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
        mask_creation_bbox_overlap_threshold=0.6,
    )

    manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            _tracklet(2, (10, 20, 30, 60)),
        ],
    )

    delayed_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            _tracklet(1, (10, 10, 30, 50)),
            _tracklet(2, (10, 20, 30, 60)),
        ],
        new_tracklets=[
            _tracklet(1, (10, 10, 30, 50)),
        ],
    )

    assert delayed_output is not None
    assert delayed_output.tracklet_mask_dict == {2: 0}
    assert manager._pending_tracklet_ids == {1}

    retried_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            _tracklet(1, (60, 10, 80, 50)),
            _tracklet(2, (10, 20, 30, 60)),
        ],
    )

    assert retried_output is not None
    assert retried_output.tracklet_mask_dict == {2: 0, 1: 1}
    assert manager._pending_tracklet_ids == set()


def test_mask_manager_removes_terminated_tracklet_from_pending_pool() -> None:
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
        mask_creation_bbox_overlap_threshold=0.6,
    )

    manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            _tracklet(2, (10, 20, 30, 60)),
        ],
    )

    manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            _tracklet(1, (10, 10, 30, 50)),
            _tracklet(2, (10, 20, 30, 60)),
        ],
        new_tracklets=[
            _tracklet(1, (10, 10, 30, 50)),
        ],
    )

    assert manager._pending_tracklet_ids == {1}

    output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            _tracklet(2, (10, 20, 30, 60)),
        ],
        removed_tracklet_ids=[1],
    )

    assert output is not None
    assert output.tracklet_mask_dict == {2: 0}
    assert manager._pending_tracklet_ids == set()
