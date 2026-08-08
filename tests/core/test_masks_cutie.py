# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from contextlib import nullcontext
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trackers.core.masks.base import MaskOutput  # noqa: E402
from trackers.core.masks.cutie import (  # noqa: E402
    CutieMaskPropagator,
    _apply_backend_perf_options,
    _autocast_context,
    _binary_masks_to_indexed_mask,
    _binary_masks_to_non_overlapping_torch,
    _build_tracklet_object_dict,
    _compute_mask_avg_prob_dict,
    _get_object_id_to_tmp_id,
    _image_to_torch,
    _indexed_mask_to_binary_masks,
    _output_prob_to_object_indexed_mask,
)


class _FakeObject:
    """Cutie object stub exposing the immutable ``id`` attribute."""

    def __init__(self, object_id: int) -> None:
        self.id = object_id


class _FakeObjectManager:
    """ObjectManager stub exposing the temporary-ID -> object mapping."""

    def __init__(self, tmp_id_to_obj: dict[int, _FakeObject]) -> None:
        self.tmp_id_to_obj = tmp_id_to_obj


class _FakeFeatureStore:
    """ImageFeatureStore stub modeling encode-on-miss plus explicit delete.

    ``encode_count`` records how many distinct time indices were encoded so a
    test can assert an add_masks reused a prior propagate's features (count
    unchanged) instead of re-encoding the identical frame.
    """

    def __init__(self) -> None:
        self._store: dict[int, tuple[str, int]] = {}
        self.encode_count = 0
        self.no_warning = False

    def get_features(self, index: int, image: Any) -> tuple[str, int]:
        if index not in self._store:
            self.encode_count += 1
            self._store[index] = ("features", index)
        return self._store[index]

    def delete(self, index: int) -> None:
        self._store.pop(index, None)


class _FakeInferenceCore:
    """InferenceCore stub mirroring curr_ti bookkeeping and delete_buffer.

    ``step`` increments ``curr_ti`` before its feature lookup (as the real core
    does), encodes on a store miss, and drops the entry when ``delete_buffer`` is
    True. It returns a fixed probability tensor so the wrapper's downstream mask
    conversion runs unchanged.
    """

    def __init__(self, prob: Any, tmp_id_to_obj: dict[int, _FakeObject]) -> None:
        self.image_feature_store = _FakeFeatureStore()
        self.object_manager = _FakeObjectManager(tmp_id_to_obj)
        self.curr_ti = -1
        self._prob = prob

    def step(
        self,
        image: Any,
        mask: Any | None = None,
        objects: list[int] | None = None,
        *,
        idx_mask: bool = True,
        delete_buffer: bool = True,
    ) -> Any:
        self.curr_ti += 1
        self.image_feature_store.get_features(self.curr_ti, image)
        if delete_buffer:
            self.image_feature_store.delete(self.curr_ti)
        return self._prob

    def clear_memory(self) -> None:
        self.curr_ti = -1


def _make_reuse_propagator(prob: Any, tmp_id_to_obj: dict[int, _FakeObject]) -> CutieMaskPropagator:
    """Build an initialized propagator wired to a feature-store-aware fake core."""
    propagator = object.__new__(CutieMaskPropagator)
    propagator.device = torch.device("cpu")
    propagator.use_amp = False
    propagator.processor = _FakeInferenceCore(prob, tmp_id_to_obj)
    propagator._initialized = True
    propagator._tracklet_object_dict = {10: 1}
    propagator._object_ids = [1]
    propagator._last_object_id = 1
    propagator._last_indexed_mask = np.zeros((2, 2), dtype=np.int32)
    propagator._cached_feature_ti = None
    propagator._cached_feature_frame = None
    return propagator


def test_drop_cached_features_deletes_store_entry_and_clears_fields() -> None:
    """_drop_cached_features removes the retained store entry and resets the cache pointers."""
    prob = torch.zeros((3, 2, 2), dtype=torch.float32)
    prob[0] = 1.0
    propagator = _make_reuse_propagator(prob, {1: _FakeObject(1), 2: _FakeObject(2)})
    propagator.processor.image_feature_store._store[7] = ("features", 7)
    propagator._cached_feature_ti = 7
    propagator._cached_feature_frame = np.zeros((2, 2, 3), dtype=np.uint8)

    propagator._drop_cached_features()

    assert 7 not in propagator.processor.image_feature_store._store
    assert propagator._cached_feature_ti is None
    assert propagator._cached_feature_frame is None


def test_add_masks_reuses_cached_features_for_the_same_frame() -> None:
    """add_masks on the frame propagate last encoded reuses those features, skipping a re-encode."""
    prob = torch.zeros((3, 2, 2), dtype=torch.float32)
    prob[0] = 1.0
    propagator = _make_reuse_propagator(prob, {1: _FakeObject(1), 2: _FakeObject(2)})
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    # Simulate the preceding propagate: features kept at ti=0, cache recorded.
    store = propagator.processor.image_feature_store
    propagator.processor.curr_ti = 0
    store._store[0] = ("features", 0)
    store.encode_count = 0
    propagator._cached_feature_ti = 0
    propagator._cached_feature_frame = frame

    propagator.add_masks(
        frame=frame,
        mask_output=MaskOutput(
            masks=np.ones((1, 2, 2), dtype=bool),
            tracklet_mask_dict={20: 0},
            mask_avg_prob_dict=None,
        ),
    )

    assert store.encode_count == 0
    # Both the original entry and the pre-seeded copy are dropped: no leak.
    assert store._store == {}
    assert propagator._cached_feature_ti is None


def test_add_masks_reencodes_when_frame_differs_from_cache() -> None:
    """The identity guard falls back to a normal encode when add_masks gets a different frame."""
    prob = torch.zeros((3, 2, 2), dtype=torch.float32)
    prob[0] = 1.0
    propagator = _make_reuse_propagator(prob, {1: _FakeObject(1), 2: _FakeObject(2)})

    store = propagator.processor.image_feature_store
    propagator.processor.curr_ti = 0
    store._store[0] = ("features", 0)
    store.encode_count = 0
    propagator._cached_feature_ti = 0
    propagator._cached_feature_frame = np.zeros((2, 2, 3), dtype=np.uint8)

    propagator.add_masks(
        frame=np.full((2, 2, 3), 255, dtype=np.uint8),
        mask_output=MaskOutput(
            masks=np.ones((1, 2, 2), dtype=bool),
            tracklet_mask_dict={20: 0},
            mask_avg_prob_dict=None,
        ),
    )

    assert store.encode_count == 1
    assert store._store == {}
    assert propagator._cached_feature_ti is None


def test_propagate_retains_single_feature_entry_and_reset_clears_it() -> None:
    """propagate keeps exactly one feature entry alive for reuse; reset drops it to avoid ti collisions."""
    prob = torch.zeros((2, 2, 2), dtype=torch.float32)
    prob[0] = 1.0
    propagator = _make_reuse_propagator(prob, {1: _FakeObject(1)})
    store = propagator.processor.image_feature_store

    propagator.propagate(frame=np.zeros((2, 2, 3), dtype=np.uint8))
    propagator.propagate(frame=np.zeros((2, 2, 3), dtype=np.uint8))

    # Each propagate drops the previous kept entry before keeping its own.
    assert len(store._store) == 1
    assert propagator._cached_feature_ti == propagator.processor.curr_ti

    propagator.reset()

    assert store._store == {}
    assert propagator._cached_feature_ti is None


@pytest.fixture
def initialized_cutie_propagator() -> CutieMaskPropagator:
    propagator = object.__new__(CutieMaskPropagator)
    propagator.device = torch.device("cpu")
    propagator.use_amp = False
    propagator._initialized = True
    propagator._tracklet_object_dict = {}
    propagator._object_ids = []
    propagator._last_object_id = 0
    propagator._last_indexed_mask = None
    propagator._cached_feature_ti = None
    propagator._cached_feature_frame = None
    return propagator


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


def test_image_to_torch_normalizes_near_black_uint8_frame_by_dtype() -> None:
    """A near-black uint8 frame is divided by 255 based on dtype, not skipped by a max>1 heuristic."""
    frame = np.ones((2, 2, 3), dtype=np.uint8)

    frame_torch = _image_to_torch(frame, device=torch.device("cpu"))

    assert torch.allclose(frame_torch, torch.full((3, 2, 2), 1 / 255))


def test_image_to_torch_leaves_normalized_float_frame_unscaled() -> None:
    """A float frame already in [0, 1] is passed through without a second division."""
    frame = np.full((2, 2, 3), 0.5, dtype=np.float32)

    frame_torch = _image_to_torch(frame, device=torch.device("cpu"))

    assert torch.allclose(frame_torch, torch.full((3, 2, 2), 0.5))


def test_image_to_torch_normalizes_uint16_frame_by_dtype_max() -> None:
    """A uint16 frame is divided by 65535, not 255, so wide integer types are not mis-scaled."""
    frame = np.full((2, 2, 3), 65535, dtype=np.uint16)

    frame_torch = _image_to_torch(frame, device=torch.device("cpu"))

    assert torch.allclose(frame_torch, torch.ones((3, 2, 2)))


def test_image_to_torch_rejects_signed_integer_frame() -> None:
    """A signed-integer frame has an ambiguous range and is rejected rather than mis-scaled."""
    frame = np.zeros((2, 2, 3), dtype=np.int16)

    with pytest.raises(ValueError, match="Signed-integer frames are not supported"):
        _image_to_torch(frame, device=torch.device("cpu"))


def test_binary_masks_to_indexed_mask_accepts_non_bool_masks() -> None:
    """Float masks are coerced to bool so the bitwise overlap resolution stays correct."""
    masks = np.zeros((2, 4, 4), dtype=np.float32)
    masks[0, 1:3, 1:3] = 1.0
    masks[1, 2:4, 2:4] = 1.0

    indexed_mask = _binary_masks_to_indexed_mask(masks=masks, object_ids=[1, 2])

    # First mask keeps the shared pixel (2, 2); second mask yields to it.
    assert indexed_mask[2, 2] == 1
    assert indexed_mask[3, 3] == 2
    assert indexed_mask.dtype == np.int32


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
        dtype=np.int32,
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


def test_binary_masks_to_indexed_mask_uses_object_ids_and_first_mask_priority() -> None:
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 2:4, 2:4] = True

    indexed_mask = _binary_masks_to_indexed_mask(
        masks=masks,
        object_ids=[10, 20],
    )

    expected = np.zeros((4, 4), dtype=np.int32)
    expected[1:3, 1:3] = 10
    expected[2, 3] = 20
    expected[3, 2] = 20
    expected[3, 3] = 20

    np.testing.assert_array_equal(indexed_mask, expected)


def test_binary_masks_to_indexed_mask_validates_number_of_object_ids() -> None:
    masks = np.zeros((2, 4, 4), dtype=bool)

    with pytest.raises(ValueError, match="Number of masks must match"):
        _binary_masks_to_indexed_mask(
            masks=masks,
            object_ids=[10],
        )


def test_get_object_id_to_tmp_id_reads_cutie_object_manager_mapping() -> None:
    class DummyObject:
        def __init__(self, object_id: int) -> None:
            self.id = object_id

    class DummyObjectManager:
        def __init__(self) -> None:
            self.tmp_id_to_obj = {
                1: DummyObject(10),
                2: DummyObject(30),
            }

    class DummyProcessor:
        def __init__(self) -> None:
            self.object_manager = DummyObjectManager()

    assert _get_object_id_to_tmp_id(DummyProcessor()) == {
        10: 1,
        30: 2,
    }


def test_build_tracklet_object_dict_converts_local_indices_to_cutie_object_ids() -> None:
    tracklet_object_dict = _build_tracklet_object_dict({10: 0, 20: 1, 30: 2})

    assert tracklet_object_dict == {10: 1, 20: 2, 30: 3}


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
        object_ids=[10, 20],
        object_id_to_tmp_id={10: 1, 20: 2},
    )

    assert np.isclose(mask_avg_prob_dict[10], np.mean([0.7, 0.8, 0.6]))
    assert np.isclose(mask_avg_prob_dict[20], 0.7)


def test_compute_mask_avg_prob_dict_accepts_precomputed_max_result() -> None:
    """A precomputed torch.max(prob, dim=0) yields the same averages as the internal reduction."""
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
        object_ids=[10, 20],
        object_id_to_tmp_id={10: 1, 20: 2},
        max_result=torch.max(prob, dim=0),
    )

    assert np.isclose(mask_avg_prob_dict[10], np.mean([0.7, 0.8, 0.6]))
    assert np.isclose(mask_avg_prob_dict[20], 0.7)


def test_output_prob_to_object_indexed_mask_matches_original_cutie_remap() -> None:
    """The lookup-table conversion reproduces Cutie's argmax + per-object remap, including tmp-ID gaps and ties."""

    class DummyObject:
        def __init__(self, object_id: int) -> None:
            self.id = object_id

    class DummyObjectManager:
        def __init__(self) -> None:
            # Gap after object removal: tmp IDs are compacted while object IDs stay immutable.
            self.tmp_id_to_obj = {
                1: DummyObject(10),
                2: DummyObject(30),
            }

    class DummyProcessor:
        def __init__(self) -> None:
            self.object_manager = DummyObjectManager()

    prob = torch.tensor(
        [
            [[0.8, 0.5, 0.1], [0.5, 0.2, 0.1]],
            [[0.1, 0.5, 0.8], [0.5, 0.6, 0.2]],
            [[0.1, 0.2, 0.1], [0.5, 0.2, 0.7]],
        ],
        dtype=torch.float32,
    )
    processor = DummyProcessor()

    indexed_mask = _output_prob_to_object_indexed_mask(processor=processor, prob=prob)

    # Reference: original Cutie output_prob_to_mask remap applied to argmax.
    reference = torch.argmax(prob, dim=0)
    remapped = torch.zeros_like(reference)
    for tmp_id, obj in processor.object_manager.tmp_id_to_obj.items():
        remapped[reference == tmp_id] = obj.id

    assert indexed_mask.dtype == np.int32
    np.testing.assert_array_equal(indexed_mask, remapped.numpy())


def test_output_prob_to_object_indexed_mask_accepts_precomputed_indices() -> None:
    """Passing precomputed torch.max indices yields the same indexed mask as the internal reduction."""

    class DummyObject:
        def __init__(self, object_id: int) -> None:
            self.id = object_id

    class DummyObjectManager:
        def __init__(self) -> None:
            self.tmp_id_to_obj = {1: DummyObject(7)}

    class DummyProcessor:
        def __init__(self) -> None:
            self.object_manager = DummyObjectManager()

    prob = torch.tensor(
        [
            [[0.9, 0.1], [0.4, 0.6]],
            [[0.1, 0.9], [0.6, 0.4]],
        ],
        dtype=torch.float32,
    )
    processor = DummyProcessor()

    indexed_mask = _output_prob_to_object_indexed_mask(
        processor=processor,
        prob=prob,
        max_indices=torch.max(prob, dim=0).indices,
    )

    np.testing.assert_array_equal(
        indexed_mask,
        _output_prob_to_object_indexed_mask(processor=processor, prob=prob),
    )


def test_propagate_returns_none_when_not_initialized() -> None:
    propagator = object.__new__(CutieMaskPropagator)
    propagator._initialized = False

    output = propagator.propagate(frame=np.zeros((4, 4, 3), dtype=np.uint8))

    assert output is None


def test_reset_clears_internal_state(
    initialized_cutie_propagator: CutieMaskPropagator,
) -> None:
    class DummyProcessor:
        def __init__(self) -> None:
            self.clear_memory_called = False

        def clear_memory(self) -> None:
            self.clear_memory_called = True

    processor = DummyProcessor()

    propagator = initialized_cutie_propagator
    propagator.processor = processor
    propagator._tracklet_object_dict = {10: 1}
    propagator._object_ids = [1]
    propagator._last_object_id = 5
    propagator._last_indexed_mask = np.ones((4, 4), dtype=np.int32)

    propagator.reset()

    assert processor.clear_memory_called
    assert propagator._tracklet_object_dict == {}
    assert propagator._object_ids == []
    assert not propagator._initialized
    assert propagator._last_object_id == 0
    assert propagator._last_indexed_mask is None


def test_initialize_success_path_builds_state_and_calls_processor_step() -> None:
    """Scenario: initialize() with valid non-overlapping masks builds tracklet/object state and feeds Cutie memory."""

    class DummyProcessor:
        def __init__(self) -> None:
            self.step_calls: list[dict[str, Any]] = []
            self.clear_memory_called = False

        def clear_memory(self) -> None:
            self.clear_memory_called = True

        def step(
            self,
            frame: Any,
            mask: Any | None = None,
            objects: list[int] | None = None,
            *,
            idx_mask: bool = True,
        ) -> Any:
            self.step_calls.append(
                {
                    "frame": frame,
                    "mask": mask,
                    "objects": objects,
                    "idx_mask": idx_mask,
                }
            )
            return torch.zeros((3, 4, 4), dtype=torch.float32)

    processor = DummyProcessor()

    propagator = object.__new__(CutieMaskPropagator)
    propagator.device = torch.device("cpu")
    propagator.use_amp = False
    propagator.processor = processor
    propagator._tracklet_object_dict = {}
    propagator._object_ids = []
    propagator._initialized = False
    propagator._last_object_id = 0
    propagator._last_indexed_mask = None
    propagator._cached_feature_ti = None
    propagator._cached_feature_frame = None

    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True

    mask_output = MaskOutput(
        masks=masks,
        tracklet_mask_dict={10: 0, 20: 1},
        mask_avg_prob_dict=None,
    )

    propagator.initialize(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        mask_output=mask_output,
    )

    assert propagator._tracklet_object_dict == {10: 1, 20: 2}
    assert propagator._object_ids == [1, 2]
    assert propagator._last_object_id == 2
    assert propagator._initialized is True
    # Cutie temporal memory is cleared before the masks-present step so a
    # mid-sequence re-init never reuses stale memory.
    assert processor.clear_memory_called is True

    expected_indexed_mask = np.zeros((4, 4), dtype=np.int32)
    expected_indexed_mask[0:2, 0:2] = 1
    expected_indexed_mask[2:4, 2:4] = 2
    np.testing.assert_array_equal(propagator._last_indexed_mask, expected_indexed_mask)

    assert len(processor.step_calls) == 1
    step_call = processor.step_calls[0]
    assert step_call["objects"] == [1, 2]
    assert step_call["idx_mask"] is False
    assert step_call["frame"].shape == (3, 4, 4)
    np.testing.assert_array_equal(
        step_call["mask"].cpu().numpy(),
        masks.astype(np.float32),
    )


def test_initialize_clears_stale_memory_on_mid_sequence_reinitialization() -> None:
    """Scenario: a second initialize() clears Cutie memory and rebuilds fresh object state, no prior-segment leak."""

    class DummyProcessor:
        def __init__(self) -> None:
            self.clear_memory_calls = 0
            self.step_calls = 0

        def clear_memory(self) -> None:
            self.clear_memory_calls += 1

        def step(
            self,
            frame: Any,
            mask: Any | None = None,
            objects: list[int] | None = None,
            *,
            idx_mask: bool = True,
        ) -> Any:
            self.step_calls += 1
            return torch.zeros((3, 4, 4), dtype=torch.float32)

    processor = DummyProcessor()

    propagator = object.__new__(CutieMaskPropagator)
    propagator.device = torch.device("cpu")
    propagator.use_amp = False
    propagator.processor = processor
    propagator._tracklet_object_dict = {}
    propagator._object_ids = []
    propagator._initialized = False
    propagator._last_object_id = 0
    propagator._last_indexed_mask = None
    propagator._cached_feature_ti = None
    propagator._cached_feature_frame = None

    first_masks = np.zeros((2, 4, 4), dtype=bool)
    first_masks[0, 0:2, 0:2] = True
    first_masks[1, 2:4, 2:4] = True
    propagator.initialize(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        mask_output=MaskOutput(masks=first_masks, tracklet_mask_dict={10: 0, 20: 1}),
    )

    # Simulate the prior segment fully terminating, then a new set of tracklets
    # entering — MaskManager re-enters the not-initialized branch and calls
    # initialize() again on the same propagator.
    second_masks = np.zeros((1, 4, 4), dtype=bool)
    second_masks[0, 0:3, 0:3] = True
    propagator.initialize(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        mask_output=MaskOutput(masks=second_masks, tracklet_mask_dict={99: 0}),
    )

    assert processor.clear_memory_calls == 2
    assert processor.step_calls == 2
    # State reflects only the second segment; no leakage from ids 10/20.
    assert propagator._tracklet_object_dict == {99: 1}
    assert propagator._object_ids == [1]
    assert propagator._last_object_id == 1


@pytest.mark.parametrize(
    "mask_output",
    [
        pytest.param(
            MaskOutput(masks=None, tracklet_mask_dict={}, mask_avg_prob_dict=None),
            id="masks-is-none",
        ),
        pytest.param(
            MaskOutput(
                masks=np.zeros((0, 4, 4), dtype=bool),
                tracklet_mask_dict={},
                mask_avg_prob_dict=None,
            ),
            id="zero-masks",
        ),
    ],
)
def test_initialize_resets_cutie_memory_when_no_masks_to_initialize(
    mask_output: MaskOutput,
) -> None:
    """Scenario: initialize() with no masks (masks=None or an empty mask array) clears Cutie memory via reset()."""

    class DummyProcessor:
        def __init__(self) -> None:
            self.clear_memory_called = False

        def clear_memory(self) -> None:
            self.clear_memory_called = True

    processor = DummyProcessor()

    propagator = object.__new__(CutieMaskPropagator)
    propagator.processor = processor
    propagator._tracklet_object_dict = {10: 1}
    propagator._object_ids = [1]
    propagator._last_object_id = 1
    propagator._last_indexed_mask = np.ones((4, 4), dtype=np.int32)
    propagator._initialized = True
    propagator._cached_feature_ti = None
    propagator._cached_feature_frame = None

    propagator.initialize(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        mask_output=mask_output,
    )

    assert processor.clear_memory_called
    assert propagator._tracklet_object_dict == {}
    assert propagator._object_ids == []
    assert propagator._last_object_id == 0
    assert propagator._last_indexed_mask is None
    assert not propagator._initialized


@pytest.mark.parametrize(
    ("masks", "tracklet_mask_dict", "error_match"),
    [
        (
            np.zeros((4, 4), dtype=bool),
            {1: 0},
            "expects masks with shape",
        ),
        (
            np.zeros((2, 4, 4), dtype=bool),
            {1: 0},
            "must match number of masks",
        ),
        (
            np.zeros((2, 4, 4), dtype=bool),
            {1: 1, 2: 2},
            "local mask indices",
        ),
    ],
)
def test_initialize_validates_mask_output(
    masks: np.ndarray,
    tracklet_mask_dict: dict[int, int],
    error_match: str,
) -> None:
    propagator = object.__new__(CutieMaskPropagator)

    mask_output = MaskOutput(
        masks=masks,
        tracklet_mask_dict=tracklet_mask_dict,
        mask_avg_prob_dict=None,
    )

    with pytest.raises(ValueError, match=error_match):
        propagator.initialize(
            frame=np.zeros((4, 4, 3), dtype=np.uint8),
            mask_output=mask_output,
        )


def test_add_masks_requires_initialization() -> None:
    propagator = object.__new__(CutieMaskPropagator)
    propagator._initialized = False

    mask_output = MaskOutput(
        masks=np.zeros((1, 4, 4), dtype=bool),
        tracklet_mask_dict={10: 0},
        mask_avg_prob_dict=None,
    )

    with pytest.raises(RuntimeError, match="initialized"):
        propagator.add_masks(
            frame=np.zeros((4, 4, 3), dtype=np.uint8),
            mask_output=mask_output,
        )


def test_add_masks_assigns_new_object_ids_and_updates_state(
    initialized_cutie_propagator: CutieMaskPropagator,
) -> None:
    class DummyObject:
        def __init__(self, object_id: int) -> None:
            self.id = object_id

    class DummyObjectManager:
        def __init__(self) -> None:
            self.tmp_id_to_obj = {tmp_id: DummyObject(tmp_id) for tmp_id in (1, 2, 3, 4)}

    class DummyProcessor:
        def __init__(self) -> None:
            self.step_calls: list[dict[str, Any]] = []
            self.object_manager = DummyObjectManager()

        def step(
            self,
            frame: Any,
            mask: Any | None = None,
            objects: list[int] | None = None,
            *,
            idx_mask: bool = True,
        ) -> Any:
            self.step_calls.append(
                {
                    "frame": frame,
                    "mask": mask,
                    "objects": objects,
                    "idx_mask": idx_mask,
                }
            )
            # One-hot probability whose channel argmax reproduces the
            # temporary-ID mask [[0,3,3,0],[0,0,4,4],[0,0,0,4],[0,0,0,0]].
            tmp_id_mask = torch.tensor(
                [
                    [0, 3, 3, 0],
                    [0, 0, 4, 4],
                    [0, 0, 0, 4],
                    [0, 0, 0, 0],
                ],
                dtype=torch.int64,
            )
            prob = torch.zeros((5, 4, 4), dtype=torch.float32)
            prob.scatter_(0, tmp_id_mask.unsqueeze(0), 1.0)
            return prob

    processor = DummyProcessor()

    propagator = initialized_cutie_propagator
    propagator.processor = processor
    propagator._tracklet_object_dict = {10: 1, 20: 2}
    propagator._object_ids = [1, 2]
    propagator._last_object_id = 2
    propagator._last_indexed_mask = np.zeros((4, 4), dtype=np.int32)

    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 1:3] = True
    masks[1, 1:3, 2:4] = True

    mask_output = MaskOutput(
        masks=masks,
        tracklet_mask_dict={30: 0, 40: 1},
        mask_avg_prob_dict=None,
    )

    propagator.add_masks(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        mask_output=mask_output,
    )

    assert propagator._last_object_id == 4
    assert propagator._tracklet_object_dict == {
        10: 1,
        20: 2,
        30: 3,
        40: 4,
    }
    assert propagator._object_ids == [1, 2, 3, 4]

    assert len(processor.step_calls) == 1
    step_call = processor.step_calls[0]
    assert step_call["objects"] == [3, 4]
    assert step_call["idx_mask"] is True

    expected_indexed_mask = np.zeros((4, 4), dtype=np.int64)
    expected_indexed_mask[0:2, 1:3] = 3
    expected_indexed_mask[1, 3] = 4
    expected_indexed_mask[2, 2:4] = 4

    np.testing.assert_array_equal(
        step_call["mask"].cpu().numpy(),
        expected_indexed_mask,
    )

    np.testing.assert_array_equal(
        propagator._last_indexed_mask,
        np.array(
            [
                [0, 3, 3, 0],
                [0, 0, 4, 4],
                [0, 0, 0, 4],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )


def test_add_masks_rejects_duplicate_tracklet_ids() -> None:
    propagator = object.__new__(CutieMaskPropagator)
    propagator._initialized = True
    propagator._tracklet_object_dict = {10: 1}

    mask_output = MaskOutput(
        masks=np.zeros((1, 4, 4), dtype=bool),
        tracklet_mask_dict={10: 0},
        mask_avg_prob_dict=None,
    )

    with pytest.raises(ValueError, match="already have Cutie objects"):
        propagator.add_masks(
            frame=np.zeros((4, 4, 3), dtype=np.uint8),
            mask_output=mask_output,
        )


def test_remove_masks_requires_initialization() -> None:
    propagator = object.__new__(CutieMaskPropagator)
    propagator._initialized = False

    with pytest.raises(RuntimeError, match="initialized"):
        propagator.remove_masks([10])


def test_remove_masks_deletes_objects_and_updates_state(
    initialized_cutie_propagator: CutieMaskPropagator,
) -> None:
    class DummyProcessor:
        def __init__(self) -> None:
            self.deleted_objects: list[int] | None = None

        def delete_objects(self, object_ids: list[int]) -> None:
            self.deleted_objects = object_ids

    processor = DummyProcessor()

    propagator = initialized_cutie_propagator
    propagator.processor = processor
    propagator._tracklet_object_dict = {
        10: 1,
        20: 2,
        30: 3,
    }
    propagator._object_ids = [1, 2, 3]
    propagator._last_object_id = 3
    propagator._last_indexed_mask = np.array(
        [
            [0, 1, 2],
            [2, 3, 3],
        ],
        dtype=np.int32,
    )

    propagator.remove_masks([20])

    assert processor.deleted_objects == [2]
    assert propagator._tracklet_object_dict == {
        10: 1,
        30: 3,
    }
    assert propagator._object_ids == [1, 3]
    assert propagator._last_object_id == 3
    assert propagator._initialized

    np.testing.assert_array_equal(
        propagator._last_indexed_mask,
        np.array(
            [
                [0, 1, 0],
                [0, 3, 3],
            ],
            dtype=np.int32,
        ),
    )


def test_remove_masks_marks_uninitialized_when_no_objects_remain(
    initialized_cutie_propagator: CutieMaskPropagator,
) -> None:
    class DummyProcessor:
        def __init__(self) -> None:
            self.deleted_objects: list[int] | None = None
            self.clear_memory_called = False

        def delete_objects(self, object_ids: list[int]) -> None:
            self.deleted_objects = object_ids

        def clear_memory(self) -> None:
            self.clear_memory_called = True

    processor = DummyProcessor()

    propagator = initialized_cutie_propagator
    propagator.processor = processor
    propagator._tracklet_object_dict = {10: 1}
    propagator._object_ids = [1]
    propagator._last_object_id = 1
    propagator._last_indexed_mask = np.ones((2, 2), dtype=np.int32)

    propagator.remove_masks([10])

    assert processor.deleted_objects == [1]
    assert propagator._tracklet_object_dict == {}
    assert propagator._object_ids == []
    assert propagator._last_object_id == 1
    assert propagator._last_indexed_mask is None
    assert not propagator._initialized
    # Removing the last object clears Cutie temporal memory, not just mappings.
    assert processor.clear_memory_called is True


def test_propagate_returns_mask_output_contract_after_temporary_id_shift(
    initialized_cutie_propagator: CutieMaskPropagator,
) -> None:
    class DummyObject:
        def __init__(self, object_id: int) -> None:
            self.id = object_id

    class DummyObjectManager:
        def __init__(self) -> None:
            self.tmp_id_to_obj = {
                1: DummyObject(1),
                2: DummyObject(3),
            }

    class DummyProcessor:
        def __init__(self) -> None:
            self.object_manager = DummyObjectManager()
            self.curr_ti = 0
            self.step_delete_buffer: bool | None = None

        def step(self, frame: Any, *, delete_buffer: bool = True) -> Any:
            # propagate keeps features alive for a possible next add_masks.
            self.step_delete_buffer = delete_buffer
            # Channels:
            # 0: background, 1: tmp 1 -> object 1, 2: tmp 2 -> object 3.
            return torch.tensor(
                [
                    [[0.9, 0.1], [0.1, 0.1]],
                    [[0.1, 0.8], [0.2, 0.1]],
                    [[0.0, 0.1], [0.7, 0.8]],
                ],
                dtype=torch.float32,
            )

        def output_prob_to_mask(self, prob: Any) -> Any:
            # This simulates Cutie's tmp-id to object-id remapping:
            # tmp 2 is remapped to immutable object ID 3.
            return torch.tensor(
                [
                    [0, 1],
                    [3, 3],
                ],
                dtype=torch.int64,
            )

    propagator = initialized_cutie_propagator
    propagator.processor = DummyProcessor()
    propagator._tracklet_object_dict = {
        10: 1,
        30: 3,
    }
    propagator._object_ids = [1, 3]
    propagator._last_object_id = 3
    propagator._last_indexed_mask = None

    output = propagator.propagate(
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
    )

    assert output is not None
    assert output.tracklet_mask_dict == {
        10: 0,
        30: 1,
    }

    assert output.masks is not None
    assert output.masks.shape == (2, 2, 2)

    # Object 1
    np.testing.assert_array_equal(
        output.masks[0],
        np.array(
            [
                [False, True],
                [False, False],
            ],
            dtype=bool,
        ),
    )
    # Object 3
    np.testing.assert_array_equal(
        output.masks[1],
        np.array(
            [
                [False, False],
                [True, True],
            ],
            dtype=bool,
        ),
    )

    assert output.mask_avg_prob_dict is not None
    assert np.isclose(output.mask_avg_prob_dict[10], 0.8)
    assert np.isclose(output.mask_avg_prob_dict[30], np.mean([0.7, 0.8]))

    np.testing.assert_array_equal(
        propagator._last_indexed_mask,
        np.array(
            [
                [0, 1],
                [3, 3],
            ],
            dtype=np.int32,
        ),
    )

    # propagate keeps its image features alive (delete_buffer=False) and records
    # the store index and frame so a following add_masks can reuse them.
    assert propagator.processor.step_delete_buffer is False
    assert propagator._cached_feature_ti == 0
    assert propagator._cached_feature_frame is not None


@patch("trackers.core.masks.cutie.torch.amp.autocast")
def test_autocast_context_disabled_returns_nullcontext(autocast_mock: MagicMock) -> None:
    """With AMP off the helper returns a no-op context and never constructs an autocast."""
    context = _autocast_context(use_amp=False, device=torch.device("cuda"))

    assert isinstance(context, nullcontext)
    autocast_mock.assert_not_called()


@pytest.mark.parametrize("device_type", ["cuda", "cpu"])
@patch("trackers.core.masks.cutie.torch.amp.autocast")
def test_autocast_context_enabled_follows_device_type(autocast_mock: MagicMock, device_type: str) -> None:
    """With AMP on the autocast backend follows the device type, not a hardcoded 'cuda'."""
    context = _autocast_context(use_amp=True, device=torch.device(device_type))

    autocast_mock.assert_called_once_with(device_type, enabled=True)
    assert context is autocast_mock.return_value


@patch("trackers.core.masks.cutie.torch.compile")
def test_apply_backend_perf_options_disabled_invokes_no_transform(compile_mock: MagicMock) -> None:
    """Both transforms off: neither model.to nor torch.compile runs and the model is returned as-is."""
    model = MagicMock()

    result = _apply_backend_perf_options(model, channels_last=False, compile_model=False)

    assert result is model
    model.to.assert_not_called()
    compile_mock.assert_not_called()


@patch("trackers.core.masks.cutie.torch.compile")
def test_apply_backend_perf_options_channels_last_converts_model_once(compile_mock: MagicMock) -> None:
    """channels_last=True calls model.to exactly once with the channels_last format and compiles nothing."""
    model = MagicMock()

    result = _apply_backend_perf_options(model, channels_last=True, compile_model=False)

    model.to.assert_called_once_with(memory_format=torch.channels_last)
    assert result is model.to.return_value
    compile_mock.assert_not_called()


@patch("trackers.core.masks.cutie.torch.compile")
def test_apply_backend_perf_options_compiles_only_shape_stable_encoder_methods(compile_mock: MagicMock) -> None:
    """compile_model=True compiles exactly encode_image + transform_key, leaving the variable-shape methods eager."""
    model = MagicMock()
    original_encode_image = model.encode_image
    original_transform_key = model.transform_key
    original_segment = model.segment
    original_encode_mask = model.encode_mask

    result = _apply_backend_perf_options(model, channels_last=False, compile_model=True)

    assert result is model
    model.to.assert_not_called()
    # Exactly the two shape-stable encoder methods are compiled.
    assert compile_mock.call_count == 2
    compile_mock.assert_any_call(original_encode_image)
    compile_mock.assert_any_call(original_transform_key)
    # Compiled results are assigned back to those two encoder entry points.
    assert model.encode_image is compile_mock.return_value
    assert model.transform_key is compile_mock.return_value
    # Variable object-count methods stay eager to avoid recompilation churn.
    assert model.segment is original_segment
    assert model.encode_mask is original_encode_mask
