# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trackers.core.mcbyte.masks.base import MaskOutput  # noqa: E402
from trackers.core.mcbyte.masks.cutie import (  # noqa: E402
    CutieMaskPropagator,
    _binary_masks_to_non_overlapping_torch,
    _build_tracklet_object_dict,
    _compute_mask_avg_prob_dict,
    _image_to_torch,
    _indexed_mask_to_binary_masks,
    _torch_prob_to_numpy_mask,
)


def test_image_to_torch_converts_rgb_numpy_frame() -> None:
    frame = np.array(
        [
            [[0, 128, 255], [255, 0, 0]],
            [[0, 255, 0], [0, 0, 255]],
        ],
        dtype=np.uint8,
    )

    frame_torch = _image_to_torch(frame, device=torch.device("cpu"))

    assert frame_torch.shape == (3, 2, 2)
    assert frame_torch.dtype == torch.float32
    assert torch.isclose(frame_torch[0, 0, 0], torch.tensor(0.0))
    assert torch.isclose(frame_torch[1, 0, 0], torch.tensor(128 / 255))
    assert torch.isclose(frame_torch[2, 0, 0], torch.tensor(1.0))


def test_torch_prob_to_numpy_mask_returns_argmax_index_mask() -> None:
    prob = torch.tensor(
        [
            [[0.9, 0.1], [0.2, 0.1]],
            [[0.1, 0.8], [0.7, 0.2]],
            [[0.0, 0.1], [0.1, 0.7]],
        ],
        dtype=torch.float32,
    )

    indexed_mask = _torch_prob_to_numpy_mask(prob)

    expected = np.array(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(indexed_mask, expected)


def test_binary_masks_to_non_overlapping_torch_resolves_overlaps_with_first_mask_priority() -> None:
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 2:4, 2:4] = True

    masks_torch = _binary_masks_to_non_overlapping_torch(
        masks=masks,
        device=torch.device("cpu"),
    )

    assert masks_torch.shape == (2, 4, 4)
    assert masks_torch.dtype == torch.float32

    # First mask keeps its whole region.
    assert masks_torch[0, 1:3, 1:3].bool().all()
    assert masks_torch[0].sum().item() == 4

    # Second mask loses the overlapping pixel at (2, 2).
    assert masks_torch[1, 2, 2].item() == 0.0
    assert masks_torch[1, 2, 3].item() == 1.0
    assert masks_torch[1, 3, 2].item() == 1.0
    assert masks_torch[1, 3, 3].item() == 1.0
    assert masks_torch[1].sum().item() == 3


def test_indexed_mask_to_binary_masks_returns_one_channel_per_object_id() -> None:
    indexed_mask = np.array(
        [
            [0, 1, 1],
            [0, 2, 2],
        ],
        dtype=np.uint8,
    )

    masks = _indexed_mask_to_binary_masks(
        indexed_mask=indexed_mask,
        object_ids=[1, 2],
    )

    assert masks.shape == (2, 2, 3)
    assert masks.dtype == bool

    np.testing.assert_array_equal(
        masks[0],
        np.array(
            [
                [False, True, True],
                [False, False, False],
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(
        masks[1],
        np.array(
            [
                [False, False, False],
                [False, True, True],
            ],
            dtype=bool,
        ),
    )


def test_build_tracklet_object_dict_converts_local_indices_to_cutie_object_ids() -> None:
    tracklet_object_dict = _build_tracklet_object_dict(
        {
            10: 0,
            20: 1,
            30: 2,
        }
    )

    assert tracklet_object_dict == {
        10: 1,
        20: 2,
        30: 3,
    }


def test_compute_mask_avg_prob_dict_matches_original_mcbyte_logic() -> None:
    prob = torch.tensor(
        [
            [[0.8, 0.1, 0.1], [0.6, 0.2, 0.1]],
            [[0.1, 0.7, 0.8], [0.3, 0.6, 0.2]],
            [[0.1, 0.2, 0.1], [0.1, 0.2, 0.7]],
        ],
        dtype=torch.float32,
    )

    mask_avg_prob_dict = _compute_mask_avg_prob_dict(
        prob=prob,
        object_ids=[1, 2],
    )

    assert np.isclose(mask_avg_prob_dict[1], np.mean([0.7, 0.8, 0.6]))
    assert np.isclose(mask_avg_prob_dict[2], 0.7)


def test_propagate_returns_none_when_not_initialized() -> None:
    propagator = object.__new__(CutieMaskPropagator)
    propagator._initialized = False

    output = propagator.propagate(frame=np.zeros((4, 4, 3), dtype=np.uint8))

    assert output is None


def test_reset_clears_internal_state() -> None:
    propagator = object.__new__(CutieMaskPropagator)

    class DummyProcessor:
        def __init__(self) -> None:
            self.clear_memory_called = False

        def clear_memory(self) -> None:
            self.clear_memory_called = True

    processor = DummyProcessor()

    propagator.processor = processor
    propagator._tracklet_object_dict = {10: 1}
    propagator._object_ids = [1]
    propagator._initialized = True

    propagator.reset()

    assert processor.clear_memory_called
    assert propagator._tracklet_object_dict == {}
    assert propagator._object_ids == []
    assert not propagator._initialized


def test_initialize_validates_mask_shape() -> None:
    propagator = object.__new__(CutieMaskPropagator)

    mask_output = MaskOutput(
        masks=np.zeros((4, 4), dtype=bool),
        tracklet_mask_dict={1: 0},
        mask_avg_prob_dict=None,
    )

    with pytest.raises(ValueError, match="expects masks with shape"):
        propagator.initialize(
            frame=np.zeros((4, 4, 3), dtype=np.uint8),
            mask_output=mask_output,
        )


def test_initialize_validates_mapping_length() -> None:
    propagator = object.__new__(CutieMaskPropagator)

    mask_output = MaskOutput(
        masks=np.zeros((2, 4, 4), dtype=bool),
        tracklet_mask_dict={1: 0},
        mask_avg_prob_dict=None,
    )

    with pytest.raises(ValueError, match="must match number of masks"):
        propagator.initialize(
            frame=np.zeros((4, 4, 3), dtype=np.uint8),
            mask_output=mask_output,
        )


def test_initialize_validates_local_mask_indices() -> None:
    propagator = object.__new__(CutieMaskPropagator)

    mask_output = MaskOutput(
        masks=np.zeros((2, 4, 4), dtype=bool),
        tracklet_mask_dict={1: 1, 2: 2},
        mask_avg_prob_dict=None,
    )

    with pytest.raises(ValueError, match="local mask indices"):
        propagator.initialize(
            frame=np.zeros((4, 4, 3), dtype=np.uint8),
            mask_output=mask_output,
        )
