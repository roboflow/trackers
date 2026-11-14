# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""EdgeTAM lifecycle contracts without weights or Transformers installed."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from trackers.core.masks.base import MaskOutput, TrackletSnapshot
from trackers.core.masks.edgetam import DEFAULT_EDGETAM_MODEL, EdgeTAMMaskBackend
from trackers.core.masks.manager import MaskManager


@dataclass
class FakeSession:
    """Retain frames and emulate the processor's replaceable prompt batch."""

    frames: list[np.ndarray]
    masks: dict[int, np.ndarray] = field(default_factory=dict)
    pending: dict[int, np.ndarray] = field(default_factory=dict)

    def reset_tracking_data(self) -> None:
        """Clear objects while deliberately retaining frames, like Transformers."""
        self.masks.clear()
        self.pending.clear()


class FakeRuntime:
    """Make lost prompts, wrong frame indices and leaked sessions observable."""

    device = "cpu"
    model_revision = "fake-revision"

    def __init__(self, **kwargs: Any) -> None:
        """Collect weak session references so reset can be checked for release."""
        self.sessions: list[weakref.ReferenceType[FakeSession]] = []
        self.largest_session = 0

    def new_session(self, frame: np.ndarray) -> FakeSession:
        """Seed a separate session with exactly one stored frame."""
        session = FakeSession([frame.copy()])
        self.sessions.append(weakref.ref(session))
        self.largest_session = max(self.largest_session, 1)
        return session

    def prompt_boxes(self, session: FakeSession, tracklets: list[TrackletSnapshot]) -> None:
        """Rasterize boxes independently, preserving overlaps."""
        masks = {}
        for tracklet in tracklets:
            mask = np.zeros(session.frames[0].shape[:2], dtype=bool)
            x1, y1, x2, y2 = tracklet.xyxy.astype(int)
            mask[y1:y2, x1:x2] = True
            masks[tracklet.tracker_id] = mask
        self.prompt_masks(session, 0, masks)

    def prompt_masks(self, session: FakeSession, frame_idx: int, masks: dict[int, np.ndarray]) -> None:
        """Require callers to consume one batch before submitting another."""
        assert not session.pending, "An earlier prompt batch was overwritten."
        assert 0 <= frame_idx < len(session.frames)
        session.pending = {key: value.copy() for key, value in masks.items()}

    def infer(self, session: FakeSession, frame_idx: int, frame: np.ndarray | None = None) -> dict[int, np.ndarray]:
        """Propagate identity masks and return deliberately reversed object order."""
        if frame is not None:
            assert frame_idx == len(session.frames)
            session.frames.append(frame.copy())
        assert frame_idx < len(session.frames)
        session.masks.update(session.pending)
        session.pending.clear()
        self.largest_session = max(self.largest_session, len(session.frames))
        return {key: value.copy() for key, value in reversed(list(session.masks.items()))}


@pytest.fixture
def backend_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace only the model boundary, keeping the public lifecycle real."""
    monkeypatch.setattr("trackers.core.masks.edgetam._EdgeTAMRuntime", FakeRuntime)
    return EdgeTAMMaskBackend


def frame(value: int = 0) -> np.ndarray:
    """Create small distinct RGB images so prompt/frame alignment is testable."""
    return np.full((8, 12, 3), value, dtype=np.uint8)


def tracklet(tracker_id: int) -> TrackletSnapshot:
    """Use a standard box for single-object and input-validation checks."""
    return TrackletSnapshot(tracker_id, np.array([1, 1, 6, 7], dtype=np.float32))


def assert_owned_masks(output: MaskOutput | None, expected: dict[int, np.ndarray]) -> None:
    """Check each identity's pixels, independently of the returned row ordering."""
    assert output is not None
    assert output.masks is not None
    assert set(output.tracklet_mask_dict) == set(expected)
    for tracker_id, mask in expected.items():
        np.testing.assert_array_equal(
            output.masks[output.tracklet_mask_dict[tracker_id]], mask, err_msg=f"Wrong mask for tracklet {tracker_id}"
        )


def test_runtime_metadata_is_read_only(backend_factory: Any) -> None:
    """Metadata assignments cannot pretend to move the model or change its weights."""
    backend = backend_factory()
    for name, expected in (("device", "cpu"), ("model_revision", "fake-revision")):
        assert getattr(backend, name) == expected
        with pytest.raises(AttributeError):
            setattr(backend, name, "replacement")
        assert getattr(backend, name) == expected


@pytest.mark.parametrize(
    ("model_id", "revision", "expected_revision"),
    [
        (DEFAULT_EDGETAM_MODEL, None, "c266ce53b3fc00f0f495b583f6a116c4e57f53bb"),
        (DEFAULT_EDGETAM_MODEL, "main", "main"),
        (DEFAULT_EDGETAM_MODEL, "another-commit", "another-commit"),
        ("user/custom-edgetam", None, None),
        ("user/custom-edgetam", "custom-commit", "custom-commit"),
        ("/local/edgetam", None, None),
    ],
)
def test_default_checkpoint_is_pinned_without_overriding_custom_models(
    monkeypatch: pytest.MonkeyPatch, model_id: str, revision: str | None, expected_revision: str | None
) -> None:
    """A stable default must not leak a repository-specific commit into another model."""
    constructor = Mock(side_effect=FakeRuntime)
    monkeypatch.setattr("trackers.core.masks.edgetam._EdgeTAMRuntime", constructor)
    EdgeTAMMaskBackend(model_id=model_id, revision=revision)
    assert constructor.call_args.kwargs["model_id"] == model_id
    assert constructor.call_args.kwargs["revision"] == expected_revision


def test_backend_lifecycle_preserves_mask_ownership(backend_factory: Any) -> None:
    """Keep distinct overlapping masks with their owners through the full lifecycle."""
    backend = backend_factory(max_session_frames=4)
    snapshots = [tracklet(0), TrackletSnapshot(20, np.array([4, 3, 9, 8]))]
    expected = {key: np.zeros((8, 12), dtype=bool) for key in (0, 20)}
    expected[0][1:7, 1:6] = True
    expected[20][3:8, 4:9] = True
    initial = backend.generate(frame(), snapshots)
    assert initial.mask_avg_prob_dict is None
    assert initial.masks.shape == (2, 8, 12)
    assert np.any(initial.masks[0] & initial.masks[1])
    assert_owned_masks(initial, expected)
    backend.initialize(frame(), initial)
    assert_owned_masks(backend.propagate(frame(1)), expected)

    # Keep an ID above 255 active through both survivor reseeding and rollover.
    addition = backend.generate(frame(1), [TrackletSnapshot(300, np.array([2, 0, 8, 4]))])
    expected[300] = np.zeros((8, 12), dtype=bool)
    expected[300][0:4, 2:8] = True
    assert_owned_masks(addition, {300: expected[300]})
    backend.add_masks(frame(1), addition)
    assert_owned_masks(backend.propagate(frame(2)), expected)
    backend.remove_masks([20])
    del expected[20]
    assert_owned_masks(backend.propagate(frame(3)), expected)
    assert_owned_masks(backend.propagate(frame(4)), expected)
    assert backend.session_index == 1
    assert backend.session_frame_count == 2
    assert backend._runtime.largest_session <= 4
    backend.remove_masks([0, 300])
    assert backend.propagate(frame(5)) is None
    assert all(reference() is None for reference in backend._runtime.sessions)
    backend.add_masks(frame(5), backend.generate(frame(5), [tracklet(50)]))
    assert_owned_masks(backend.propagate(frame(6)), {50: expected[0]})
    runtime = backend._runtime
    backend.reset()
    assert backend._runtime is runtime
    assert backend.session_frame_count == 0
    assert all(reference() is None for reference in runtime.sessions)


@pytest.mark.parametrize("cap", [2, 3, None])
def test_session_bound_includes_seed(backend_factory: Any, cap: int | None) -> None:
    """Progress indefinitely within the cap, including the transferred seed."""
    backend = backend_factory(max_session_frames=cap)
    backend.initialize(frame(), backend.generate(frame(), [tracklet(0)]))
    for index in range(1, 20):
        output = backend.propagate(frame(index))
        assert output.tracklet_mask_dict == {0: 0}
        assert backend.session_frame_count <= (cap or 20)
    assert backend._runtime.largest_session == (cap or 20)


def test_generation_and_returned_arrays_cannot_mutate_temporal_state(backend_factory: Any) -> None:
    """A temporary box session and caller-owned outputs cannot corrupt rollover."""
    backend = backend_factory(max_session_frames=2)
    initial = backend.generate(frame(), [tracklet(0)])
    backend.initialize(frame(), initial)
    initial.masks[:] = False
    backend.generate(frame(90), [tracklet(90)])
    output = backend.propagate(frame(1))
    assert output.masks.any()
    output.masks[:] = False
    assert backend.propagate(frame(2)).masks.any()
    assert backend.session_index == 1


@pytest.mark.parametrize("operation", ["initialize", "propagate", "rollover", "add_masks", "remove_masks"])
@pytest.mark.parametrize(
    ("failure", "message"),
    [
        pytest.param("exception", "Simulated model failure", id="inference-exception"),
        pytest.param("missing_id", "returned object IDs", id="missing-object"),
        pytest.param("wrong_resolution", "wrong frame resolution", id="wrong-resolution"),
    ],
)
def test_failed_temporal_inference_discards_state_and_allows_reinitialization(
    backend_factory: Any, monkeypatch: pytest.MonkeyPatch, operation: str, failure: str, message: str
) -> None:
    """Failed temporal updates must release their history and never return stale masks."""
    backend = backend_factory(max_session_frames=3)
    initial = backend.generate(frame(), [tracklet(0), tracklet(20)])
    backend.initialize(frame(), initial)
    for index in range(1, 4):
        backend.propagate(frame(index))
    # Begin with a nonzero session index so failure must also clear the counter.
    assert backend.session_index == 1
    assert backend.session_frame_count == 2
    if operation == "rollover":
        backend.propagate(frame(4))
    addition = MaskOutput(np.ones((1, 8, 12), dtype=bool), {300: 0})
    original_infer = backend._runtime.infer

    def fail_inference(session: FakeSession, frame_idx: int, frame: np.ndarray | None = None) -> dict[int, np.ndarray]:
        """Fail after touching model state, as a partial inference can do."""
        predictions = original_infer(session, frame_idx, frame)
        if failure == "exception":
            raise RuntimeError("Simulated model failure")
        first_id = next(iter(predictions))
        if failure == "missing_id":
            del predictions[first_id]
        else:
            predictions[first_id] = np.zeros((1, 1), dtype=bool)
        return predictions

    with monkeypatch.context() as patch:
        patch.setattr(backend._runtime, "infer", fail_inference)
        with pytest.raises(RuntimeError, match=message):
            if operation == "initialize":
                backend.initialize(frame(3), initial)
            elif operation == "propagate":
                backend.propagate(frame(4))
            elif operation == "rollover":
                backend.propagate(frame(5))
            elif operation == "add_masks":
                backend.add_masks(frame(3), addition)
            else:
                backend.remove_masks([20])

    assert backend.session_frame_count == 0
    assert backend.session_index == 0
    assert backend.propagate(frame(6)) is None
    assert all(reference() is None for reference in backend._runtime.sessions)
    backend.initialize(frame(7), addition)
    assert_owned_masks(backend.propagate(frame(8)), {300: np.ones((8, 12), dtype=bool)})
    assert backend.session_frame_count == 2
    assert backend.session_index == 0


def test_failed_temporary_generation_preserves_active_session(
    backend_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed box prompt must leave ongoing propagation and its counters intact."""
    backend = backend_factory(max_session_frames=3)
    initial = backend.generate(frame(), [tracklet(0), tracklet(20)])
    backend.initialize(frame(), initial)
    for index in range(1, 4):
        backend.propagate(frame(index))
    original_infer = backend._runtime.infer

    def fail_inference(session: FakeSession, frame_idx: int, frame: np.ndarray | None = None) -> dict[int, np.ndarray]:
        original_infer(session, frame_idx, frame)
        raise RuntimeError("Simulated temporary generation failure")

    with monkeypatch.context() as patch:
        patch.setattr(backend._runtime, "infer", fail_inference)
        with pytest.raises(RuntimeError, match="temporary generation failure"):
            backend.generate(frame(90), [tracklet(300)])

    assert backend.session_index == 1
    assert backend.session_frame_count == 2
    expected = {key: initial.masks[row] for key, row in initial.tracklet_mask_dict.items()}
    assert_owned_masks(backend.propagate(frame(4)), expected)
    assert backend.session_frame_count == 3
    regenerated = backend.generate(frame(90), [tracklet(300)])
    assert_owned_masks(regenerated, {300: expected[0]})


def test_mask_manager_uses_previous_image_for_lifecycle(backend_factory: Any) -> None:
    """Exercise the unchanged McByte consumer, including simultaneous add/remove."""
    backend = backend_factory(max_session_frames=2)
    manager = MaskManager(backend, backend)
    assert manager.get_updated_masks(frame(), None, []) is None
    first = manager.get_updated_masks(frame(1), frame(), [tracklet(0)])
    assert first is not None
    assert first.tracklet_mask_dict == {0: 0}
    next_output = manager.get_updated_masks(
        frame(2), frame(1), [tracklet(30)], new_tracklets=[tracklet(30)], removed_tracklet_ids=[0]
    )
    assert next_output is not None
    assert next_output.tracklet_mask_dict == {30: 0}
    assert backend._runtime.largest_session <= 2
    manager.reset()
    assert backend.propagate(frame(3)) is None


@pytest.mark.parametrize("cap", [0, 1, -1, True, 2.5])
def test_invalid_cap_rejected_before_loading(monkeypatch: pytest.MonkeyPatch, cap: Any) -> None:
    """Invalid bounds must not trigger a model download."""

    def fail_load(**kwargs: Any) -> None:
        raise AssertionError("Model loading should not be reached.")

    monkeypatch.setattr("trackers.core.masks.edgetam._EdgeTAMRuntime", fail_load)
    with pytest.raises(ValueError, match="max_session_frames"):
        EdgeTAMMaskBackend(max_session_frames=cap)


def test_addition_validation_does_not_damage_session(backend_factory: Any) -> None:
    """Reject duplicate IDs, wrong frames and malformed mask mappings before mutation."""
    backend = backend_factory()
    initial = backend.generate(frame(), [tracklet(0)])
    backend.initialize(frame(), initial)
    with pytest.raises(ValueError, match="already"):
        backend.add_masks(frame(), initial)
    addition = backend.generate(frame(1), [tracklet(2)])
    with pytest.raises(ValueError, match="frame"):
        backend.add_masks(frame(1), addition)
    invalid = MaskOutput(np.ones((2, 8, 12), dtype=bool), {2: 0, 3: 0})
    with pytest.raises(ValueError, match="mapping"):
        backend.add_masks(frame(), invalid)
    assert backend.propagate(frame(2)).tracklet_mask_dict == {0: 0}


def test_empty_masks_survive_transfer_and_unknown_removal(backend_factory: Any) -> None:
    """Empty masks remain owned until explicit removal, without synthetic recovery."""
    backend = backend_factory(max_session_frames=2)
    backend.initialize(frame(), MaskOutput(np.zeros((1, 8, 12), dtype=bool), {20: 0}))
    backend.propagate(frame(1))
    backend.remove_masks([999])
    output = backend.propagate(frame(2))
    assert output.tracklet_mask_dict == {20: 0}
    assert not output.masks.any()


def test_resolution_change_requires_reset(backend_factory: Any) -> None:
    """Reject a different pixel grid before entering model state."""
    backend = backend_factory()
    backend.initialize(frame(), backend.generate(frame(), [tracklet(0)]))
    with pytest.raises(ValueError, match="resolution"):
        backend.propagate(np.zeros((10, 12, 3), dtype=np.uint8))
    assert backend.session_frame_count == 1


@pytest.mark.parametrize("empty_masks", [None, np.zeros((0, 8, 12), dtype=bool)])
def test_empty_inputs_match_shared_contract(backend_factory: Any, empty_masks: np.ndarray | None) -> None:
    """Empty generation does no inference; empty initialization releases history."""
    backend = backend_factory()
    output = backend.generate(frame(), [])
    assert output.masks.shape == (0, 8, 12)
    assert not backend._runtime.sessions
    backend.initialize(frame(), backend.generate(frame(), [tracklet(0)]))
    backend.add_masks(frame(), MaskOutput(empty_masks, {}))
    assert backend.session_frame_count == 1
    backend.initialize(frame(), MaskOutput(empty_masks, {}))
    assert backend.propagate(frame(1)) is None
    assert all(reference() is None for reference in backend._runtime.sessions)


def test_addition_batches_latest_masks_with_new_objects(backend_factory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid mixing HF propagated and mask-prompt score shapes on one stored frame."""
    backend = backend_factory()
    backend.initialize(frame(), backend.generate(frame(), [tracklet(0), tracklet(20)]))
    backend.propagate(frame(1))
    original_prompt = backend._runtime.prompt_masks

    def require_complete_batch(session: FakeSession, frame_idx: int, masks: dict[int, np.ndarray]) -> None:
        assert set(masks) == {0, 20, 30, 40}
        original_prompt(session, frame_idx, masks)

    addition = backend.generate(frame(1), [tracklet(30), tracklet(40)])
    monkeypatch.setattr(backend._runtime, "prompt_masks", require_complete_batch)
    backend.add_masks(frame(1), addition)
    assert set(backend.propagate(frame(2)).tracklet_mask_dict) == {0, 20, 30, 40}
    assert backend.session_frame_count == 3


@pytest.mark.parametrize("operation", ["initialize", "add_masks"])
@pytest.mark.parametrize("index", [False, True, 0.0, "0"])
def test_malformed_row_indices_preserve_previous_session(backend_factory: Any, operation: str, index: Any) -> None:
    """Reject Python equality aliases before NumPy can reinterpret them as indexing."""
    backend = backend_factory()
    backend.initialize(frame(), backend.generate(frame(), [tracklet(0)]))
    invalid = MaskOutput(np.ones((1, 8, 12), dtype=bool), {20: index})
    with pytest.raises(ValueError, match="mapping"):
        getattr(backend, operation)(frame(), invalid)
    assert backend.propagate(frame(1)).tracklet_mask_dict == {0: 0}


@pytest.mark.parametrize("invalid_id", [True, 1.0, "1"])
def test_invalid_removal_ids_cannot_alias_active_objects(backend_factory: Any, invalid_id: Any) -> None:
    """A mixed removal list must be rejected atomically, preserving every object."""
    backend = backend_factory()
    backend.initialize(frame(), backend.generate(frame(), [tracklet(1), tracklet(20)]))
    with pytest.raises(ValueError, match="IDs"):
        backend.remove_masks([20, invalid_id])
    assert set(backend.propagate(frame(1)).tracklet_mask_dict) == {1, 20}
