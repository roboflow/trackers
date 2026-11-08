# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from trackers.core.masks import TrackletSnapshot
from trackers.core.masks.base import MaskOutput
from trackers.core.masks.dummy import DummyBoxMaskGenerator, DummyIdentityMaskPropagator
from trackers.core.masks.manager import MaskManager


class _FlakyMaskPropagator(DummyIdentityMaskPropagator):
    """Propagator that succeeds a bounded number of times, then fails.

    Unlike ``DummyIdentityMaskPropagator``, which never returns ``None`` once initialized, this test double reproduces a
    propagation-runtime failure (e.g. an external backend such as Cutie losing its internal memory) occurring mid-
    sequence, after the manager has already been initialized.
    """

    def __init__(self, succeed_calls: int = 1) -> None:
        super().__init__()
        self._succeed_calls = succeed_calls
        self._propagate_calls = 0

    def propagate(self, frame: np.ndarray) -> MaskOutput | None:
        self._propagate_calls += 1
        if self._propagate_calls > self._succeed_calls:
            return None
        return super().propagate(frame)


class _WrongResolutionMaskPropagator(DummyIdentityMaskPropagator):
    """Propagator that returns masks at a resolution different from the frame.

    Reproduces a backend that internally resizes or pads without upsampling back to the input frame grid, which would
    make mask-conditioning gate associations in the wrong coordinate space.
    """

    def propagate(self, frame: np.ndarray) -> MaskOutput | None:
        output = super().propagate(frame)
        if output is None or output.masks is None:
            return output
        num_masks = output.masks.shape[0]
        resized = np.zeros((num_masks, frame.shape[0] + 5, frame.shape[1]), dtype=output.masks.dtype)
        return MaskOutput(
            masks=resized,
            tracklet_mask_dict=output.tracklet_mask_dict,
            mask_avg_prob_dict=output.mask_avg_prob_dict,
        )


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


def test_mask_manager_rejects_propagated_masks_at_wrong_resolution() -> None:
    """A propagator returning masks whose (H, W) differ from the frame raises ValueError at the propagate boundary."""
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=_WrongResolutionMaskPropagator(),
    )

    with pytest.raises(ValueError, match="match the frame resolution"):
        manager.get_updated_masks(
            frame=_make_frame(),
            previous_frame=_make_frame(),
            previous_tracklets=[
                TrackletSnapshot(3, np.array([5, 6, 25, 30], dtype=np.float32)),
            ],
        )


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


def test_mask_manager_removes_terminated_tracklet_from_pending_pool_before_initialization() -> None:
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
        mask_creation_bbox_overlap_threshold=0.6,
    )

    manager._pending_tracklet_ids = {1}

    output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[],
        removed_tracklet_ids=[1],
    )

    assert output is None
    assert manager._pending_tracklet_ids == set()


def test_mask_manager_resets_initialized_flag_when_propagation_fails_after_init() -> None:
    """Scenario: propagate() returns None after a prior successful init.

    The manager must reset ``_initialized`` to False so that a later call re-attempts mask creation from scratch instead
    of assuming a still-valid propagator state.
    """
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=_FlakyMaskPropagator(succeed_calls=1),
    )

    first_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            TrackletSnapshot(3, np.array([5, 6, 25, 30], dtype=np.float32)),
        ],
    )

    assert first_output is not None
    assert manager._initialized is True

    second_output = manager.get_updated_masks(
        frame=_make_frame(),
        previous_frame=_make_frame(),
        previous_tracklets=[
            TrackletSnapshot(3, np.array([5, 6, 25, 30], dtype=np.float32)),
        ],
    )

    assert second_output is None
    assert manager._initialized is False


def test_mask_manager_init_raises_value_error_for_out_of_range_threshold() -> None:
    """Scenario: constructing MaskManager with an overlap threshold outside [0, 1] must raise ValueError instead of
    silently accepting it."""
    with pytest.raises(ValueError, match="mask_creation_bbox_overlap_threshold"):
        MaskManager(
            mask_generator=DummyBoxMaskGenerator(),
            mask_propagator=DummyIdentityMaskPropagator(),
            mask_creation_bbox_overlap_threshold=1.5,
        )


def test_mask_manager_pending_tracklet_ids_property_snapshots_the_pool() -> None:
    """Scenario: the public property mirrors the deferral pool and hands back a read-only copy, so a caller cannot
    mutate the manager through it."""
    manager = MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
    )
    manager._pending_tracklet_ids = {1, 2}

    pending = manager.pending_tracklet_ids

    assert pending == frozenset({1, 2})
    assert isinstance(pending, frozenset)
    assert manager._pending_tracklet_ids == {1, 2}
