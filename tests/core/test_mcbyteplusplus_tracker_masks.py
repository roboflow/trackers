# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Observation ownership and occlusion-aware refresh without model downloads."""

import numpy as np
import pytest
import supervision as sv

from trackers import McBytePlusPlusTracker
from trackers.core.masks.base import MaskOutput, TrackletSnapshot
from trackers.core.masks.dummy import DummyBoxMaskGenerator, DummyIdentityMaskPropagator
from trackers.core.masks.manager import MaskManager
from trackers.core.mcbyteplusplus.tracklet import McBytePlusPlusTracklet


class RecordingGenerator(DummyBoxMaskGenerator):
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, list[TrackletSnapshot]]] = []

    def generate(self, frame: np.ndarray, tracklets: list[TrackletSnapshot]) -> MaskOutput:
        self.calls.append((frame.copy(), tracklets))
        return super().generate(frame, tracklets)


class RefreshingPropagator(DummyIdentityMaskPropagator):
    def __init__(self) -> None:
        super().__init__()
        self.refreshes: list[tuple[np.ndarray, set[int]]] = []
        self.propagations = 0
        self.omit_id: int | None = None
        self.fail = False

    def refresh_masks(self, frame: np.ndarray, output: MaskOutput) -> None:
        assert self._mask_output is not None and self._mask_output.masks is not None
        assert output.masks is not None
        self.refreshes.append((frame.copy(), set(output.tracklet_mask_dict)))
        for key, row in output.tracklet_mask_dict.items():
            self._mask_output.masks[self._mask_output.tracklet_mask_dict[key]] = output.masks[row]

    def propagate(self, frame: np.ndarray) -> MaskOutput | None:
        self.propagations += 1
        if self.fail:
            raise RuntimeError("inference failed")
        output = super().propagate(frame)
        if self.omit_id is not None and output is not None:
            return MaskOutput(output.masks, {k: v for k, v in output.tracklet_mask_dict.items() if k != self.omit_id})
        return output


@pytest.fixture
def pipeline() -> tuple[McBytePlusPlusTracker, RecordingGenerator, RefreshingPropagator]:
    generator, propagator = RecordingGenerator(), RefreshingPropagator()
    manager = MaskManager(generator, propagator, mask_creation_bbox_overlap_threshold=0.1)
    return (
        McBytePlusPlusTracker(
            mask_manager=manager,
            enable_cmc=False,
            minimum_mask_creation_frames=1,
            enable_mask_refresh=True,
            mask_refresh_coverage_threshold=0.5,
            mask_refresh_interval=1,
        ),
        generator,
        propagator,
    )


def advance(tracker: McBytePlusPlusTracker, frame: np.ndarray) -> MaskOutput | None:
    """Run the tracker mask step on an RGB image without invoking association."""
    assert tracker.mask_manager is not None
    tracker._last_mask_output = tracker._run_mask_manager(tracker.mask_manager, frame)
    return tracker.last_mask_output


def record(tracker: McBytePlusPlusTracker, frame: np.ndarray, snapshots: list[TrackletSnapshot]) -> None:
    """Store internal-ID detections, including unconfirmed occlusion references."""
    detections = sv.Detections(
        xyxy=np.array([t.xyxy for t in snapshots]).reshape(-1, 4),
        tracker_id=np.array([t.tracker_id for t in snapshots], dtype=int),
    )
    tracker._store_previous_mask_inputs(frame, detections, [])


def seed(pipeline: McBytePlusPlusTracker, snapshots: list[TrackletSnapshot]) -> np.ndarray:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    advance(pipeline, frame)
    record(pipeline, frame, snapshots)
    advance(pipeline, frame)
    return frame


def associate_two_tracks(tracker, similarity, minimum_similarity=0.1):
    """Exercise assignment where the first mask can overturn the geometric result."""
    boxes = np.array([[0.0, 0.0, 4.0, 4.0], [10.0, 0.0, 14.0, 4.0]])
    tracks = [McBytePlusPlusTracklet(box) for box in boxes]
    for track, key in zip(tracks, [7, 8]):
        track.tracker_id = key
    return tracker._get_mask_conditioned_associated_indices(similarity, similarity, tracks, boxes, minimum_similarity)[
        0
    ]


@pytest.mark.parametrize("minimum_similarity", [0.1, 0.5, 0.3])
def test_empty_mask_cannot_reappear_and_change_assignment(pipeline, minimum_similarity) -> None:
    """An unmatched empty mask stays excluded when it reappears on another object."""
    tracker, _, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    other = TrackletSnapshot(8, np.array([20.0, 0.0, 24.0, 4.0]))
    frame = seed(tracker, [target, other])
    propagator._mask_output.masks[0] = False
    advance(tracker, frame)
    record(tracker, frame, [])  # Lost objects have no detector box for refreshing.
    propagator._mask_output.masks[0, :4, 10:14] = True
    output = advance(tracker, frame)
    assert output is not None and output.masks is not None
    original_masks = output.masks.copy()
    similarity = np.array([[0.6, 0.5], [0.5, 0.6]])

    assert associate_two_tracks(tracker, similarity, minimum_similarity) == [(0, 0), (1, 1)]
    assert tracker.untrusted_mask_ids == frozenset({7})
    assert output.tracklet_mask_dict == {7: 0, 8: 1}
    np.testing.assert_array_equal(output.masks, original_masks)
    np.testing.assert_array_equal(propagator._mask_output.masks, original_masks)
    assert propagator.refreshes == []


@pytest.mark.parametrize("blocker", ["occlusion", "cooldown", "refresh_disabled"])
def test_displaced_mask_is_excluded_after_match_even_if_refresh_is_blocked(pipeline, blocker) -> None:
    tracker, _, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    frame = seed(tracker, [target])
    propagator._mask_output.masks[:] = False
    propagator._mask_output.masks[0, :4, 10:14] = True
    advance(tracker, frame)
    similarity = np.array([[0.6, 0.5], [0.5, 0.6]])
    # No predicted-box veto: displacement has not yet been assessed on this image.
    assert associate_two_tracks(tracker, similarity) == [(0, 1), (1, 0)]

    references = [target]
    if blocker == "occlusion":
        references.append(TrackletSnapshot(-1, np.array([3.0, 0.0, 8.0, 5.0])))
    elif blocker == "cooldown":
        tracker._mask_last_refresh[7] = tracker._mask_image_count
    else:
        tracker.enable_mask_refresh = False
    record(tracker, frame, references)

    assert tracker.untrusted_mask_ids == frozenset({7})
    assert associate_two_tracks(tracker, similarity) == [(0, 0), (1, 1)]
    advance(tracker, frame)
    assert propagator.refreshes == []
    assert tracker.untrusted_mask_ids == frozenset({7})


def test_only_validated_refresh_restores_association(pipeline) -> None:
    tracker, _, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    frame = seed(tracker, [target])
    propagator._mask_output.masks[:] = False
    advance(tracker, frame)
    record(tracker, frame, [])
    # Spontaneous recovery inside the right box still needs a fresh detector prompt.
    propagator._mask_output.masks[0, :4, :4] = True
    advance(tracker, frame)
    record(tracker, frame, [target])
    assert tracker.untrusted_mask_ids == frozenset({7})
    advance(tracker, frame)
    assert propagator.refreshes[-1][1] == {7}

    similarity = np.array([[0.5, 0.6], [0.6, 0.5]])
    assert associate_two_tracks(tracker, similarity) == [(0, 1), (1, 0)]
    record(tracker, frame, [target])  # Validate propagated output against this detector match.
    assert tracker.untrusted_mask_ids == frozenset()
    assert associate_two_tracks(tracker, similarity) == [(0, 0), (1, 1)]


def test_exclusion_preserves_other_masks_and_channel_mapping(pipeline) -> None:
    """Removing an ID from association must not renumber remaining mask channels."""
    tracker, _, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    other = TrackletSnapshot(8, np.array([10.0, 0.0, 14.0, 4.0]))
    frame = seed(tracker, [target, other])
    propagator._mask_output.masks[0] = False
    advance(tracker, frame)
    similarity = np.array([[0.5, 0.6], [0.6, 0.5]])
    # Geometry favours the crossed assignment; only the surviving channel corrects it.
    assert associate_two_tracks(tracker, similarity) == [(0, 0), (1, 1)]
    assert tracker.last_mask_output.tracklet_mask_dict == {7: 0, 8: 1}


@pytest.mark.parametrize("quality", ["empty", "displaced", "tiny"])
def test_bad_box_prompt_cannot_restore_trust_even_if_propagation_looks_good(pipeline, monkeypatch, quality) -> None:
    tracker, generator, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 10.0, 10.0]))
    frame = seed(tracker, [target])
    propagator._mask_output.masks[:] = False
    advance(tracker, frame)
    record(tracker, frame, [target])
    generate = generator.generate

    def bad_prompt(image, tracklets):
        output = generate(image, tracklets)
        output.masks[:] = False
        if quality == "displaced":
            output.masks[0, :10, 15:25] = True
        elif quality == "tiny":
            output.masks[0, 0, 0] = True  # Full coverage, but below the association fill threshold.
        return output

    monkeypatch.setattr(generator, "generate", bad_prompt)
    monkeypatch.setattr(propagator, "propagate", lambda image: DummyBoxMaskGenerator().generate(image, [target]))
    advance(tracker, frame)
    record(tracker, frame, [target])
    assert len(propagator.refreshes) == 1
    assert tracker.untrusted_mask_ids == frozenset({7})


@pytest.mark.parametrize("validation", ["no_match", "occluded", "displaced", "tiny", "empty"])
def test_refresh_requires_a_suitable_current_match(pipeline, validation) -> None:
    tracker, _, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 10.0, 10.0]))
    frame = seed(tracker, [target])
    propagator._mask_output.masks[:] = False
    advance(tracker, frame)
    record(tracker, frame, [target])
    output = advance(tracker, frame)
    assert output is not None and output.masks is not None
    assert len(propagator.refreshes) == 1
    matches = [target]
    if validation == "no_match":
        matches = []
    elif validation == "occluded":
        matches.append(TrackletSnapshot(-1, np.array([9.0, 0.0, 19.0, 11.0])))
    else:
        output.masks[:] = False
        if validation == "displaced":
            output.masks[0, :10, 15:25] = True
        elif validation == "tiny":
            output.masks[0, 0, 0] = True
    record(tracker, frame, matches)
    assert tracker.untrusted_mask_ids == frozenset({7})
    # A later good observation cannot validate the previous failed refresh.
    tracker._invalidate_mask_observation()
    advance(tracker, frame)
    record(tracker, frame, [target])
    assert len(propagator.refreshes) == 1
    assert tracker.untrusted_mask_ids == frozenset({7})


@pytest.mark.parametrize("area,refresh", [(0, True), (8, False), (9, True)])
def test_empty_and_strict_half_coverage_refresh(pipeline, area: int, refresh: bool) -> None:
    coordinator, generator, propagator = pipeline
    snapshot = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    frame = seed(coordinator, [snapshot])
    mask = np.zeros((1, 20, 30), dtype=bool)
    mask[0, 0, :area] = True
    coordinator._last_mask_output = MaskOutput(mask, {7: 0})
    record(coordinator, frame, [snapshot])
    assert coordinator.untrusted_mask_ids == (frozenset({7}) if refresh else frozenset())
    advance(coordinator, frame)
    assert bool(propagator.refreshes) is refresh
    assert len(generator.calls) == 1 + int(refresh)
    assert coordinator._last_mask_output.tracklet_mask_dict == {7: 0}


@pytest.mark.parametrize("front_bottom,front_x,occluded", [(11, 9, True), (11, 9.01, False), (10, 9, False)])
def test_refresh_occlusion_threshold_and_bottom_order(pipeline, front_bottom, front_x, occluded) -> None:
    coordinator, _, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 10.0, 10.0]))
    frame = seed(coordinator, [target])
    coordinator._last_mask_output = MaskOutput(np.zeros((1, 20, 30), dtype=bool), {7: 0})
    # The foreground box does not need to own a mask, or even a confirmed ID.
    foreground = TrackletSnapshot(-2, np.array([front_x, 0.0, 19.0, front_bottom]))
    record(coordinator, frame, [target, foreground])
    advance(coordinator, frame)
    assert bool(propagator.refreshes) is not occluded


def test_disabled_trust_filter_preserves_previous_refresh_policy(pipeline) -> None:
    coordinator, _, propagator = pipeline
    coordinator.enable_mask_trust_filter = False
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    frame = seed(coordinator, [target])
    propagator._mask_output.masks[:] = False
    advance(coordinator, frame)
    record(coordinator, frame, [])
    propagator._mask_output.masks[0, :4, :4] = True
    advance(coordinator, frame)
    assert propagator.refreshes == []
    assert coordinator.untrusted_mask_ids == frozenset()
    record(coordinator, frame, [target])  # identity propagation has recovered the box mask
    advance(coordinator, frame)
    assert propagator.refreshes == []


def test_terminated_refresh_is_not_generated_and_survivor_remains(pipeline) -> None:
    coordinator, generator, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    survivor = TrackletSnapshot(8, np.array([10.0, 0.0, 14.0, 4.0]))
    frame = seed(coordinator, [target, survivor])
    coordinator._last_mask_output.masks[0] = False
    record(coordinator, frame, [target, survivor])
    assert coordinator.untrusted_mask_ids == frozenset({7})
    coordinator._merge_pending_removals([7])
    advance(coordinator, frame)
    assert len(generator.calls) == 1 and propagator.refreshes == []
    assert coordinator._last_mask_output.tracklet_mask_dict == {8: 0}
    assert coordinator.untrusted_mask_ids == frozenset()


def test_missing_observation_clears_prompts_but_retains_anchor(pipeline) -> None:
    coordinator, generator, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    frame = seed(coordinator, [target])
    coordinator._last_mask_output.masks[:] = False
    newcomer = TrackletSnapshot(8, np.array([10.0, 0.0, 14.0, 4.0]))
    record(coordinator, frame, [target, newcomer])
    coordinator._invalidate_mask_observation()
    assert coordinator.untrusted_mask_ids == frozenset({7})
    advance(coordinator, np.full_like(frame, 1))
    assert len(generator.calls) == 1 and propagator.refreshes == []
    assert propagator.propagations == 2
    assert coordinator._last_mask_output.tracklet_mask_dict == {7: 0}
    assert coordinator.untrusted_mask_ids == frozenset({7})


def test_duplicate_updates_preserve_refresh_without_extra_propagation(pipeline) -> None:
    """Refresh uses the accepted image and box even after different duplicate inputs."""
    tracker, generator, propagator = pipeline
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    detections = sv.Detections(xyxy=np.array([[2.0, 2.0, 8.0, 12.0]]), confidence=np.ones(1))
    tracker.update(detections, frame, timestamp=0.0)
    tracker.update(detections, frame + 1, timestamp=0.04)
    propagator._mask_output.masks[:] = False
    tracker.update(detections, frame + 2, timestamp=0.08)
    assert tracker.untrusted_mask_ids == frozenset({0})
    duplicate_detections = sv.Detections(xyxy=np.array([[3.0, 2.0, 9.0, 12.0]]), confidence=np.ones(1))
    for _ in range(2):
        with pytest.warns(UserWarning, match="duplicate"):
            tracker.update(duplicate_detections, frame + 100, timestamp=0.08)
        assert tracker.last_mask_output is None
        assert tracker.untrusted_mask_ids == frozenset({0})
        assert propagator.propagations == 2
        assert propagator.refreshes == [] and len(generator.calls) == 1

    tracker.update(detections, frame + 3, timestamp=0.12)
    assert propagator.propagations == 3
    assert len(propagator.refreshes) == 1 and propagator.refreshes[0][1] == {0}
    np.testing.assert_array_equal(propagator.refreshes[0][0], frame + 2)
    np.testing.assert_array_equal(generator.calls[-1][1][0].xyxy, detections.xyxy[0])
    assert tracker.last_mask_output.masks.any()
    assert tracker.untrusted_mask_ids == frozenset()


def test_inference_failure_clears_pending_prompts(pipeline) -> None:
    coordinator, generator, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    frame = seed(coordinator, [target])
    coordinator._last_mask_output.masks[:] = False
    record(coordinator, frame, [target])
    assert coordinator.untrusted_mask_ids == frozenset({7})
    propagator.fail = True
    with pytest.raises(RuntimeError, match="inference failed"):
        advance(coordinator, frame)
    propagator.fail = False
    assert advance(coordinator, frame) is None
    assert len(generator.calls) == 2  # Initial creation and the prompt before propagation failed.
    assert coordinator.untrusted_mask_ids == frozenset()


def test_reset_clears_mask_exclusion(pipeline) -> None:
    tracker, _, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    frame = seed(tracker, [target])
    propagator._mask_output.masks[:] = False
    advance(tracker, frame)
    assert tracker.untrusted_mask_ids == frozenset({7})
    tracker.reset()
    assert tracker.untrusted_mask_ids == frozenset()
    assert tracker.last_mask_output is None
    assert propagator._mask_output is None


def test_minimum_creation_age_and_unmasked_occluder() -> None:
    generator, propagator = RecordingGenerator(), RefreshingPropagator()
    manager = MaskManager(generator, propagator, mask_creation_bbox_overlap_threshold=0.1)
    coordinator = McBytePlusPlusTracker(
        mask_manager=manager,
        enable_cmc=False,
        minimum_mask_creation_frames=2,
        enable_mask_refresh=False,
        mask_refresh_coverage_threshold=0.5,
        mask_refresh_interval=1,
    )
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 10.0, 10.0]))
    foreground = TrackletSnapshot(-2, np.array([9.0, 0.0, 19.0, 11.0]))
    record(coordinator, frame, [target])
    advance(coordinator, frame)
    assert generator.calls == []
    record(coordinator, frame, [target, foreground])
    advance(coordinator, frame)
    assert generator.calls == []
    record(coordinator, frame, [target])
    advance(coordinator, frame)
    assert coordinator._last_mask_output is not None
    assert coordinator._last_mask_output.tracklet_mask_dict == {7: 0}


def test_refresh_can_be_disabled_with_standard_propagator() -> None:
    manager = MaskManager(DummyBoxMaskGenerator(), DummyIdentityMaskPropagator())
    with pytest.raises(ValueError, match="refresh_masks"):
        McBytePlusPlusTracker(
            mask_manager=manager,
            enable_cmc=False,
            minimum_mask_creation_frames=1,
            enable_mask_refresh=True,
            mask_refresh_coverage_threshold=0.5,
            mask_refresh_interval=1,
        )
    coordinator = McBytePlusPlusTracker(
        mask_manager=manager,
        enable_cmc=False,
        minimum_mask_creation_frames=1,
        enable_mask_refresh=False,
        mask_refresh_coverage_threshold=0.5,
        mask_refresh_interval=1,
    )
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    frame = seed(coordinator, [target])
    assert coordinator._last_mask_output is not None and coordinator._last_mask_output.masks is not None
    coordinator._last_mask_output.masks[:] = False
    record(coordinator, frame, [target])
    advance(coordinator, frame)
    assert coordinator._last_mask_output.tracklet_mask_dict == {7: 0}
    assert coordinator.untrusted_mask_ids == frozenset({7})
    # With no refresher, even a healthy-looking propagated mask stays excluded.
    record(coordinator, frame, [target])
    assert coordinator.untrusted_mask_ids == frozenset({7})


@pytest.mark.parametrize("failure", ["omit_existing", "omit_new", "retain_removed"])
def test_custom_output_must_match_completed_lifecycle(pipeline, monkeypatch, failure) -> None:
    """Existing, newly added and removed IDs must agree with the backend's ownership."""
    coordinator, _, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    other = TrackletSnapshot(8, np.array([10.0, 0.0, 14.0, 4.0]))
    frame = seed(coordinator, [target] if failure == "omit_new" else [target, other])
    if failure == "retain_removed":
        monkeypatch.setattr(propagator, "remove_masks", lambda ids: None)
        coordinator._merge_pending_removals([8])
        record(coordinator, frame, [target])
    else:
        propagator.omit_id = 8
        record(coordinator, frame, [target, other])
    with pytest.raises(RuntimeError, match="tracklet ID"):
        advance(coordinator, frame)
    assert coordinator._last_mask_output is None and propagator._mask_output is None


@pytest.mark.parametrize("key,row", [(7, 0.0), (7, False), (7.0, 0)])
def test_custom_output_rejects_numeric_aliases_and_resets(pipeline, monkeypatch, key, row) -> None:
    """Reject float/bool aliases before downstream NumPy indexing can mutate tracking."""
    coordinator, _, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    frame = seed(coordinator, [target])
    original = propagator.propagate

    def alias_mapping(image):
        output = original(image)
        return MaskOutput(output.masks, {key: row})

    monkeypatch.setattr(propagator, "propagate", alias_mapping)
    record(coordinator, frame, [target])
    with pytest.raises(ValueError, match="integers"):
        advance(coordinator, frame)
    assert coordinator._last_mask_output is None and propagator._mask_output is None


@pytest.mark.parametrize("gap", ["none", "observation_gap", "oom"])
def test_pending_creation_stays_separate_from_backend_ownership(pipeline, monkeypatch, gap) -> None:
    """Occluded eligible IDs survive deferral and are created only after a visible match."""
    tracker, generator, propagator = pipeline
    owned = TrackletSnapshot(8, np.array([20.0, 0.0, 24.0, 4.0]))
    pending = TrackletSnapshot(7, np.array([0.0, 0.0, 10.0, 10.0]))
    occluder = TrackletSnapshot(-1, np.array([9.0, 0.0, 19.0, 11.0]))
    frame = seed(tracker, [owned])
    record(tracker, frame, [owned, pending, occluder])
    advance(tracker, frame)
    assert tracker._mask_tracklet_ids == {8}
    assert tracker.mask_manager.pending_tracklet_ids == frozenset({7})
    assert len(generator.calls) == 1

    if gap == "observation_gap":
        # An explicit image-less update invalidates prompts; duplicate images preserve them.
        tracker._store_previous_mask_inputs(None, sv.Detections.empty(), [])
        advance(tracker, frame + 1)
        assert tracker._mask_tracklet_ids == {8}
    elif gap == "oom":
        original = propagator.propagate

        def fail(image):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(propagator, "propagate", fail)
        record(tracker, frame, [owned, pending, occluder])
        assert advance(tracker, frame + 1) is None
        assert tracker._mask_tracklet_ids == set()
        monkeypatch.setattr(propagator, "propagate", original)

    record(tracker, frame + 2, [owned, pending])
    advance(tracker, frame + 3)
    assert tracker._mask_tracklet_ids == {7, 8}
    assert tracker.mask_manager.pending_tracklet_ids == frozenset()
    created = {t.tracker_id for t in generator.calls[-1][1]}
    assert created == ({7, 8} if gap == "oom" else {7})
    np.testing.assert_array_equal(generator.calls[-1][0], frame + 2)


def test_valid_propagation_clears_oom_streak_and_gaps_do_not(pipeline, monkeypatch) -> None:
    """Simulate OOM without an accelerator; only valid propagation clears the failure streak."""
    tracker, _, propagator = pipeline
    target = TrackletSnapshot(7, np.array([0.0, 0.0, 4.0, 4.0]))
    frame = seed(tracker, [target])
    original = propagator.propagate

    def fail(image):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(propagator, "propagate", fail)
    tracker._last_mask_output.masks[:] = False
    record(tracker, frame, [target])
    assert tracker.untrusted_mask_ids == frozenset({7})
    assert advance(tracker, frame) is None
    assert tracker.untrusted_mask_ids == frozenset()
    assert tracker._consecutive_mask_failures == 1
    tracker._invalidate_mask_observation()
    assert advance(tracker, frame) is None
    assert tracker._consecutive_mask_failures == 1
    monkeypatch.setattr(propagator, "propagate", original)
    record(tracker, frame, [target])
    advance(tracker, frame)
    assert tracker._consecutive_mask_failures == 0
    assert tracker.last_mask_output.tracklet_mask_dict == {7: 0}
