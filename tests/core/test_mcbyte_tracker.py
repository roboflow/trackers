# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import sys
import warnings
from types import ModuleType
from typing import cast

import numpy as np
import pytest
import supervision as sv
from pytest import MonkeyPatch

from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.masks.base import MaskOutput, TrackletSnapshot
from trackers.core.masks.dummy import (
    DummyBoxMaskGenerator,
    DummyIdentityMaskPropagator,
)
from trackers.core.masks.manager import MaskManager
from trackers.core.mcbyte.tracker import McByteMaskConfig, McByteTracker
from trackers.core.mcbyte.tracklet import McByteTracklet
from trackers.utils.cmc import CMCConfig


class SpyMaskManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_updated_masks(
        self,
        frame: np.ndarray,
        previous_frame: np.ndarray | None,
        previous_tracklets: list[TrackletSnapshot],
        new_tracklets: list[TrackletSnapshot] | None = None,
        removed_tracklet_ids: list[int] | None = None,
    ) -> MaskOutput | None:
        self.calls.append(
            {
                "frame": frame,
                "previous_frame": previous_frame,
                "previous_tracklets": previous_tracklets,
                "new_tracklets": [] if new_tracklets is None else new_tracklets,
                "removed_tracklet_ids": [] if removed_tracklet_ids is None else removed_tracklet_ids,
            }
        )
        return None

    def reset(self) -> None:
        pass


def _detection(xyxy: tuple[float, float, float, float], conf: float = 0.9) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([conf], dtype=np.float32),
    )


def _tracklet_with_id(
    tracker_id: int,
    xyxy: tuple[float, float, float, float],
) -> McByteTracklet:
    """Create a McByte tracklet with a stable tracker ID for association tests."""
    tracklet = McByteTracklet(
        initial_bbox=np.array(xyxy, dtype=np.float32),
    )
    tracklet.tracker_id = tracker_id
    return tracklet


def _make_frame(h: int = 480, w: int = 640, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _dummy_mask_manager() -> MaskManager:
    """Create a lightweight mask manager for tracker unit tests."""
    return MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
    )


def test_mcbyte_instantiates_and_updates_with_frame_and_sparse_opt_flow_cmc_returns_ids() -> None:
    tracker = McByteTracker(
        enable_cmc=True,
        cmc_method="sparseOptFlow",
        minimum_consecutive_frames=2,
    )

    frame = _make_frame()

    for _ in range(5):
        result = tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame)

    assert len(result) == 1
    assert result.tracker_id is not None
    assert result.tracker_id[0] >= 0
    assert len(tracker.tracks) == 1


def test_mcbyte_cmc_downscale_default_is_scoped_and_preserves_override() -> None:
    """Factor 6 is McByte-only; generic CMC, BoT-SORT, and explicit factor 2 remain stable."""
    default_tracker = McByteTracker(enable_mask_manager=False)
    conservative_tracker = McByteTracker(enable_mask_manager=False, cmc_downscale=2)
    botsort_tracker = BoTSORTTracker()

    assert default_tracker.cmc is not None
    assert default_tracker.cmc.downscale == 6
    assert conservative_tracker.cmc is not None
    assert conservative_tracker.cmc.downscale == 2
    assert CMCConfig().downscale == 2
    assert botsort_tracker.cmc is not None
    assert botsort_tracker.cmc.downscale == 2


def test_mcbyte_emits_unmatched_high_conf_detection_with_placeholder_id() -> None:
    """A high-conf detection that neither matches nor spawns is still returned with tracker_id -1."""
    # conf 0.65 is high (>= high_conf_det_threshold 0.6) but below the
    # activation threshold 0.7, so on an empty tracker it matches nothing and
    # spawns nothing. It must still be returned with tracker_id -1, matching the
    # documented contract and the handling of unmatched low-confidence dets.
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
        high_conf_det_threshold=0.6,
        track_activation_threshold=0.7,
    )

    result = tracker.update(_detection((100.0, 100.0, 200.0, 200.0), conf=0.65))

    assert len(result) == 1
    assert result.tracker_id is not None
    assert result.tracker_id[0] == -1
    assert len(tracker.tracks) == 0


def test_mcbyte_reset_restores_mask_manager_disabled_by_cuda_oom() -> None:
    """After repeated CUDA-OOM auto-disables the mask manager, reset() re-attaches the original manager."""

    class OOMMaskManager:
        def __init__(self) -> None:
            self.reset_calls = 0

        def get_updated_masks(
            self,
            frame: np.ndarray,
            previous_frame: np.ndarray | None,
            previous_tracklets: list[TrackletSnapshot],
            new_tracklets: list[TrackletSnapshot] | None = None,
            removed_tracklet_ids: list[int] | None = None,
        ) -> MaskOutput | None:
            raise RuntimeError("CUDA out of memory")

        def reset(self) -> None:
            self.reset_calls += 1

    manager = OOMMaskManager()
    tracker = McByteTracker(
        enable_cmc=False,
        mask_manager=manager,  # type: ignore[arg-type]
        minimum_consecutive_frames=1,
    )

    frame = _make_frame()
    for _ in range(3):
        tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame)

    # Three consecutive OOM failures disable the pipeline for the run.
    assert tracker.mask_manager is None

    tracker.reset()

    # reset() (documented new-video boundary) restores and clears the manager.
    assert tracker.mask_manager is manager
    assert manager.reset_calls == 1
    assert tracker._consecutive_mask_failures == 0


def test_mcbyte_does_not_advance_masks_on_duplicate_timestamp() -> None:
    """A duplicate timestamp skips Kalman predict and must not step the mask backend (masks stay in sync)."""
    mask_manager = SpyMaskManager()
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
        mask_manager=mask_manager,  # type: ignore[arg-type]
        minimum_consecutive_frames=1,
    )
    frame = _make_frame()

    tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame, timestamp=1.0)
    assert len(mask_manager.calls) == 1

    with pytest.warns(UserWarning, match="duplicate timestamp"):
        tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame, timestamp=1.0)

    # No extra mask-backend step on the duplicate frame.
    assert len(mask_manager.calls) == 1


def test_mcbyte_warns_once_for_mask_manager_with_dynamic_rate_timestamps() -> None:
    """enable_mask_manager + dynamic-rate timestamps warns once, not on every call.

    The mask backend advances one step per update() call regardless of elapsed time, while Kalman prediction and pruning
    scale by timestamp — a desync that grows with gap size. Disclosed in docs; this warning surfaces it at runtime too.
    """
    mask_manager = SpyMaskManager()
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
        mask_manager=mask_manager,  # type: ignore[arg-type]
        minimum_consecutive_frames=1,
    )
    frame = _make_frame()

    with pytest.warns(UserWarning, match="dynamic-rate"):
        tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame, timestamp=0.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame, timestamp=1.0 / 30.0)
    assert not any("dynamic-rate" in str(w.message) for w in caught)


def test_mcbyte_does_not_warn_for_mask_manager_without_timestamps() -> None:
    """Fixed-rate calls (no timestamp) never trigger the dynamic-rate mask warning."""
    mask_manager = SpyMaskManager()
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
        mask_manager=mask_manager,  # type: ignore[arg-type]
        minimum_consecutive_frames=1,
    )
    frame = _make_frame()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame)
    assert not any("dynamic-rate" in str(w.message) for w in caught)


def test_mcbyte_reports_early_ghost_id_prune_to_mask_cleanup() -> None:
    """A track killed by the early time-budget prune must have its mask actually dropped by the propagator.

    `_prune_lost_tracks` (ghost-ID prevention, timestamp mode only) can remove a track before association even runs,
    ahead of the late `_get_alive_tracklets` prune that `terminated_tracklet_ids` is built from. Without reporting that
    early removal too, the mask manager (and Cutie underneath it) never learns the track died and keeps propagating its
    mask indefinitely. Uses a real ``MaskManager`` + ``DummyIdentityMaskPropagator`` (not a call-recording spy) and
    ``minimum_mask_creation_frames=1`` so the track is actually masked before it dies -- otherwise there would be
    nothing orphaned in the propagator to begin with, and the test would only be exercising bookkeeping.
    """
    propagator = DummyIdentityMaskPropagator()
    mask_manager = MaskManager(mask_generator=DummyBoxMaskGenerator(), mask_propagator=propagator)
    tracker = McByteTracker(
        enable_cmc=False,
        mask_manager=mask_manager,
        minimum_consecutive_frames=1,
        minimum_mask_creation_frames=1,
        instant_first_frame_activation=True,
        lost_track_buffer=1,  # maximum_time_without_update = 1 / 30 seconds
    )
    frame = _make_frame()

    with pytest.warns(UserWarning, match="dynamic-rate"):
        result = tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame, timestamp=0.0)
    assert result.tracker_id is not None
    tracker_id = int(result.tracker_id[0])
    assert tracker_id >= 0, "track must be confirmed on frame 1"

    # Elapsed time blows way past the ~0.033s time budget: the early prune kills the track before the late
    # lifecycle prune even runs. This call's mask-manager invocation (at the top of update(), using frame 1's
    # stored tracklets) also initializes the propagator, so the track is genuinely masked when it dies.
    tracker.update(sv.Detections.empty(), frame, timestamp=100.0)
    assert tracker_id not in [t.tracker_id for t in tracker.tracks]
    assert propagator._mask_output is not None
    assert tracker_id in propagator._mask_output.tracklet_mask_dict, "track must be masked before it dies"

    # The removed_tracklet_ids from the frame above are only handed to the mask manager on the
    # NEXT call (masks are updated using events stored from the previous frame).
    tracker.update(sv.Detections.empty(), frame, timestamp=100.1)
    assert propagator._mask_output is not None
    assert tracker_id not in propagator._mask_output.tracklet_mask_dict, (
        "dead track's mask must be dropped from the propagator, not orphaned"
    )


def test_mcbyte_delivers_removed_ids_delayed_by_duplicate_timestamp() -> None:
    """A duplicate timestamp between a track's death and the next mask-manager call must not lose the removal.

    On a duplicate timestamp, `update()`'s `if timing.skip_predict: pass` branch skips the mask-manager invocation for
    that call entirely (see `update()`'s docstring "Warns" section). A death reported at the end of the prior frame must
    survive that skipped call and still reach the mask manager on the next real frame, rather than being silently
    overwritten by the skipped frame's own (empty) `removed_tracklet_ids` in `_store_previous_mask_inputs`.
    """
    mask_manager = SpyMaskManager()
    tracker = McByteTracker(
        enable_cmc=False,
        mask_manager=mask_manager,  # type: ignore[arg-type]
        minimum_consecutive_frames=1,
        instant_first_frame_activation=True,
        lost_track_buffer=1,  # maximum_time_without_update = 1 / 30 seconds
    )
    frame = _make_frame()

    with pytest.warns(UserWarning, match="dynamic-rate"):
        result = tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame, timestamp=0.0)
    assert result.tracker_id is not None
    tracker_id = int(result.tracker_id[0])

    # Kill the track via the early time-budget prune.
    tracker.update(sv.Detections.empty(), frame, timestamp=100.0)
    assert tracker_id not in [t.tracker_id for t in tracker.tracks]

    # Duplicate timestamp: mask-manager invocation is skipped for this call.
    with pytest.warns(UserWarning, match="duplicate timestamp"):
        tracker.update(sv.Detections.empty(), frame, timestamp=100.0)

    # The death must still reach the mask manager on the next non-duplicate call.
    tracker.update(sv.Detections.empty(), frame, timestamp=100.1)
    reported = cast(list, mask_manager.calls[-1]["removed_tracklet_ids"])
    assert tracker_id in reported


def test_mcbyte_delivers_removed_ids_delayed_by_frame_none_call() -> None:
    """A frame=None call must not drop a real removal that happened during that same call.

    Pruning doesn't need a frame, so a track can still die on a `frame=None` update(); the mask manager just can't be
    invoked that call (no frame to give it). The death must survive and reach the mask manager once a real frame is
    available again, not be dropped by `_store_previous_mask_inputs`'s `frame is None` branch.
    """
    mask_manager = SpyMaskManager()
    tracker = McByteTracker(
        enable_cmc=False,
        mask_manager=mask_manager,  # type: ignore[arg-type]
        minimum_consecutive_frames=1,
        instant_first_frame_activation=True,
        lost_track_buffer=1,  # maximum_time_without_update = 1 / 30 seconds
    )
    frame = _make_frame()

    with pytest.warns(UserWarning, match="dynamic-rate"):
        result = tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame, timestamp=0.0)
    assert result.tracker_id is not None
    tracker_id = int(result.tracker_id[0])

    # frame=None: the track still dies via the early time-budget prune, but there is no frame to hand
    # the mask manager, so the removal can't be delivered on this call.
    tracker.update(sv.Detections.empty(), frame=None, timestamp=100.0)
    assert tracker_id not in [t.tracker_id for t in tracker.tracks]

    tracker.update(sv.Detections.empty(), frame, timestamp=100.1)
    reported = cast(list, mask_manager.calls[-1]["removed_tracklet_ids"])
    assert tracker_id in reported


def test_mcbyte_rejects_high_conf_threshold_at_or_below_discard_floor() -> None:
    """high_conf_det_threshold at or below the 0.1 discard floor is rejected to keep the confidence split coherent."""
    with pytest.raises(ValueError, match="discard"):
        McByteTracker(enable_cmc=False, enable_mask_manager=False, high_conf_det_threshold=0.05)


def test_mcbyte_reset_clears_mask_state() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        mask_manager=_dummy_mask_manager(),
        minimum_consecutive_frames=1,
        minimum_mask_creation_frames=1,
    )

    frame = _make_frame()

    tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame)
    tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame)

    assert tracker._previous_frame is not None
    assert len(tracker._previous_tracklets) == 1
    assert tracker._last_mask_output is not None

    tracker.reset()

    assert tracker._previous_frame is None
    assert tracker._previous_tracklets == []
    assert tracker._last_mask_output is None


def test_mcbyte_does_not_store_previous_frame_without_mask_manager() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
        minimum_consecutive_frames=1,
    )

    frame = _make_frame()

    tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame)

    assert tracker._previous_frame is None
    assert tracker._previous_tracklets == []


def test_mcbyte_passes_new_tracklets_to_mask_manager_on_next_frame() -> None:
    mask_manager = SpyMaskManager()
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
        mask_manager=mask_manager,  # type: ignore[arg-type]
        minimum_consecutive_frames=1,
        minimum_mask_creation_frames=1,
    )

    frame = _make_frame()

    tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame)
    tracker.update(_detection((102.0, 102.0, 202.0, 202.0)), frame)

    assert len(mask_manager.calls) == 2

    first_call = mask_manager.calls[0]
    assert first_call["previous_frame"] is None
    assert first_call["previous_tracklets"] == []
    assert first_call["new_tracklets"] == []
    assert first_call["removed_tracklet_ids"] == []

    second_call = mask_manager.calls[1]
    assert second_call["previous_frame"] is frame

    previous_tracklets = second_call["previous_tracklets"]
    new_tracklets = second_call["new_tracklets"]

    assert isinstance(previous_tracklets, list)
    assert isinstance(new_tracklets, list)
    assert len(previous_tracklets) == 1
    assert len(new_tracklets) == 1
    assert previous_tracklets[0].tracker_id == new_tracklets[0].tracker_id
    assert second_call["removed_tracklet_ids"] == []


def test_mcbyte_mask_lifecycle_keeps_missing_tracklet_until_explicit_removal() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        mask_manager=_dummy_mask_manager(),
        minimum_consecutive_frames=1,
        minimum_mask_creation_frames=1,
    )

    frame = _make_frame()

    visible_result = sv.Detections(
        xyxy=np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
    )
    visible_result.tracker_id = np.array([7], dtype=int)

    empty_result = sv.Detections.empty()
    empty_result.tracker_id = np.array([], dtype=int)

    tracker._store_previous_mask_inputs(
        frame=frame,
        detections=visible_result,
        removed_tracklet_ids=[],
    )

    assert tracker._previous_new_tracklets[0].tracker_id == 7
    assert tracker._mask_tracklet_ids == {7}

    tracker._store_previous_mask_inputs(
        frame=frame,
        detections=empty_result,
        removed_tracklet_ids=[],
    )

    assert tracker._previous_new_tracklets == []
    assert tracker._previous_removed_tracklet_ids == set()
    assert tracker._mask_tracklet_ids == {7}

    tracker._store_previous_mask_inputs(
        frame=frame,
        detections=empty_result,
        removed_tracklet_ids=[7],
    )

    assert tracker._previous_new_tracklets == []
    assert tracker._previous_removed_tracklet_ids == {7}
    assert tracker._mask_tracklet_ids == set()


def _visible_single_tracklet(tracker_id: int = 7) -> sv.Detections:
    """One confirmed detection with the given tracker ID."""
    result = sv.Detections(xyxy=np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32))
    result.tracker_id = np.array([tracker_id], dtype=int)
    return result


def test_mcbyte_defers_mask_creation_until_minimum_frames() -> None:
    """A confirmed tracklet is masked only after minimum_mask_creation_frames consecutive visible frames."""
    tracker = McByteTracker(
        enable_cmc=False,
        mask_manager=_dummy_mask_manager(),
        minimum_consecutive_frames=1,
        minimum_mask_creation_frames=3,
    )
    frame = _make_frame()
    visible_result = _visible_single_tracklet(7)

    tracker._store_previous_mask_inputs(frame=frame, detections=visible_result, removed_tracklet_ids=[])
    assert tracker._previous_new_tracklets == []
    assert tracker._mask_tracklet_ids == set()
    assert tracker._mask_pending_ages == {7: 1}
    # A deferred tracklet is invisible to the mask pipeline (not init-masked).
    assert tracker._previous_tracklets == []

    tracker._store_previous_mask_inputs(frame=frame, detections=visible_result, removed_tracklet_ids=[])
    assert tracker._previous_new_tracklets == []
    assert tracker._mask_tracklet_ids == set()
    assert tracker._mask_pending_ages == {7: 2}
    assert tracker._previous_tracklets == []

    tracker._store_previous_mask_inputs(frame=frame, detections=visible_result, removed_tracklet_ids=[])
    assert [snapshot.tracker_id for snapshot in tracker._previous_new_tracklets] == [7]
    assert tracker._mask_tracklet_ids == {7}
    assert tracker._mask_pending_ages == {}
    # Once promoted, the tracklet is exposed to the mask pipeline.
    assert [snapshot.tracker_id for snapshot in tracker._previous_tracklets] == [7]


def test_mcbyte_defer_restarts_when_tracklet_disappears_before_threshold() -> None:
    """A tracklet that vanishes before the threshold restarts its visible-frame count on reappearance."""
    tracker = McByteTracker(
        enable_cmc=False,
        mask_manager=_dummy_mask_manager(),
        minimum_consecutive_frames=1,
        minimum_mask_creation_frames=3,
    )
    frame = _make_frame()
    visible_result = _visible_single_tracklet(7)
    empty_result = sv.Detections.empty()
    empty_result.tracker_id = np.array([], dtype=int)

    tracker._store_previous_mask_inputs(frame=frame, detections=visible_result, removed_tracklet_ids=[])
    tracker._store_previous_mask_inputs(frame=frame, detections=visible_result, removed_tracklet_ids=[])
    assert tracker._mask_pending_ages == {7: 2}

    # Tracklet not visible this frame: its pending count is dropped.
    tracker._store_previous_mask_inputs(frame=frame, detections=empty_result, removed_tracklet_ids=[])
    assert tracker._mask_pending_ages == {}

    # Reappearing tracklet must accumulate the full window again before masking.
    tracker._store_previous_mask_inputs(frame=frame, detections=visible_result, removed_tracklet_ids=[])
    assert tracker._previous_new_tracklets == []
    assert tracker._mask_tracklet_ids == set()
    assert tracker._mask_pending_ages == {7: 1}


def test_mcbyte_rejects_minimum_mask_creation_frames_below_one() -> None:
    """minimum_mask_creation_frames below 1 is rejected: 1 already means immediate creation."""
    with pytest.raises(ValueError, match="minimum_mask_creation_frames must be at least 1"):
        McByteTracker(enable_cmc=False, enable_mask_manager=False, minimum_mask_creation_frames=0)


def test_mcbyte_mask_conditioned_association_combines_locked_and_reduced_matches() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
    )

    tracklets = [
        _tracklet_with_id(10, (0.0, 0.0, 5.0, 5.0)),
        _tracklet_with_id(20, (5.0, 5.0, 10.0, 10.0)),
        _tracklet_with_id(30, (10.0, 10.0, 15.0, 15.0)),
    ]
    detection_boxes = np.array(
        [
            [0.0, 0.0, 5.0, 5.0],
            [5.0, 5.0, 10.0, 10.0],
            [10.0, 10.0, 15.0, 15.0],
        ],
        dtype=np.float32,
    )

    similarity = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.7, 0.6],
            [0.0, 0.6, 0.7],
        ],
        dtype=np.float32,
    )

    matched, unmatched_tracks, unmatched_detections = tracker._get_mask_conditioned_associated_indices(
        similarity_matrix=similarity,
        raw_iou_similarity=similarity,
        tracklets=tracklets,
        detection_boxes=detection_boxes,
        min_similarity_thresh=0.5,
    )

    # Locked match: (0,0). Reduced matches: (1,1), (2,2) - mapped back from (0,0), (1,1)
    assert matched == [
        (0, 0),
        (1, 1),
        (2, 2),
    ]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_mcbyte_mask_conditioned_association_remaps_unmatched_indices() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
    )

    tracklets = [
        _tracklet_with_id(10, (0.0, 0.0, 5.0, 5.0)),
        _tracklet_with_id(20, (5.0, 5.0, 10.0, 10.0)),
        _tracklet_with_id(30, (10.0, 10.0, 15.0, 15.0)),
    ]
    detection_boxes = np.array(
        [
            [0.0, 0.0, 5.0, 5.0],
            [5.0, 5.0, 10.0, 10.0],
            [10.0, 10.0, 15.0, 15.0],
        ],
        dtype=np.float32,
    )

    similarity = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.4, 0.3],
            [0.0, 0.2, 0.1],
        ],
        dtype=np.float32,
    )

    matched, unmatched_tracks, unmatched_detections = tracker._get_mask_conditioned_associated_indices(
        similarity_matrix=similarity,
        raw_iou_similarity=similarity,
        tracklets=tracklets,
        detection_boxes=detection_boxes,
        min_similarity_thresh=0.5,
    )

    assert matched == [(0, 0)]
    assert unmatched_tracks == [1, 2]  # mapped back from [0, 1]
    assert unmatched_detections == [1, 2]  # mapped back from [0, 1]


def test_mcbyte_mask_conditioned_association_changes_ambiguous_assignment() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
    )

    tracklets = [
        _tracklet_with_id(10, (0.0, 0.0, 5.0, 5.0)),
        _tracklet_with_id(20, (5.0, 5.0, 10.0, 10.0)),
    ]
    detection_boxes = np.array(
        [
            [0.0, 0.0, 5.0, 5.0],
            [5.0, 5.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )

    similarity = np.array(
        [
            [0.6, 0.7],
            [0.7, 0.6],
        ],
        dtype=np.float32,
    )

    matches_without_masks, _, _ = tracker._get_mask_conditioned_associated_indices(
        similarity_matrix=similarity,
        raw_iou_similarity=similarity,
        tracklets=tracklets,
        detection_boxes=detection_boxes,
        min_similarity_thresh=0.5,
    )

    masks = np.zeros((2, 10, 10), dtype=bool)
    # it will result in fill-ratio bonus of 1.0 for first detection (0.6 + 1.0 = 1.6)
    masks[0, 0:5, 0:5] = True
    # it will result in fill-ratio bonus of 1.0 for second detection (0.6 + 1.0 = 1.6)
    masks[1, 5:10, 5:10] = True

    tracker._last_mask_output = MaskOutput(
        masks=masks,
        tracklet_mask_dict={
            10: 0,
            20: 1,
        },
        mask_avg_prob_dict={
            10: 0.9,
            20: 0.9,
        },
    )

    matches_with_masks, unmatched_tracks, unmatched_detections = tracker._get_mask_conditioned_associated_indices(
        similarity_matrix=similarity,
        raw_iou_similarity=similarity,
        tracklets=tracklets,
        detection_boxes=detection_boxes,
        min_similarity_thresh=0.5,
    )

    assert matches_without_masks == [(0, 1), (1, 0)]
    assert matches_with_masks == [(0, 0), (1, 1)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_mcbyte_mask_conditioned_association_rescues_isolated_pair_when_enabled() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
        enable_isolated_mask_matching=True,
    )

    tracklets = [
        _tracklet_with_id(10, (0.0, 0.0, 10.0, 10.0)),
    ]
    detection_boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )

    tracker._last_mask_output = MaskOutput(
        masks=np.ones((1, 10, 10), dtype=bool),
        tracklet_mask_dict={10: 0},
        mask_avg_prob_dict={10: 0.9},
    )

    matched, unmatched_tracks, unmatched_detections = tracker._get_mask_conditioned_associated_indices(
        similarity_matrix=np.array([[0.2]], dtype=np.float32),
        raw_iou_similarity=np.array([[0.2]], dtype=np.float32),
        tracklets=tracklets,
        detection_boxes=detection_boxes,
        min_similarity_thresh=0.5,
    )

    # Without isolation, match would be empty, unmatched_tracks would be [0],
    # unmatched_detections would be [0]
    assert matched == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_mcbyte_builds_real_mask_pipeline_when_enabled(
    monkeypatch: MonkeyPatch,
) -> None:
    created: dict[str, object] = {}

    class FakeSAMBoxMaskGenerator:
        def __init__(
            self,
            checkpoint_path: str | None = None,
            model_type: str = "vit_b",
            device: str = "cpu",
        ) -> None:
            created["sam"] = {
                "checkpoint_path": checkpoint_path,
                "model_type": model_type,
                "device": device,
            }

    class FakeCutieMaskPropagator:
        def __init__(
            self,
            weights_path: str | None = None,
            model_type: str = "base-mega",
            config_path: str | None = None,
            config_name: str = "eval_config",
            device: str = "auto",
            use_amp: bool = False,
            max_internal_size: int = 480,
            mem_every: int | None = 10,
            use_long_term: bool | None = True,
            channels_last: bool = False,
            compile_model: bool = False,
        ) -> None:
            created["cutie"] = {
                "weights_path": weights_path,
                "model_type": model_type,
                "config_path": config_path,
                "config_name": config_name,
                "device": device,
                "use_amp": use_amp,
                "max_internal_size": max_internal_size,
                "mem_every": mem_every,
                "use_long_term": use_long_term,
                "channels_last": channels_last,
                "compile_model": compile_model,
            }

        def reset(self) -> None:
            pass

    # mcbyte_tracker_module.SAMBoxMaskGenerator and
    # mcbyte_tracker_module.CutieMaskPropagator are imported locally in
    # _build_default_mask_manager() in tracker.py
    fake_sam_module = ModuleType("trackers.core.masks.sam")
    fake_sam_module.SAMBoxMaskGenerator = FakeSAMBoxMaskGenerator  # type: ignore[attr-defined]

    fake_cutie_module = ModuleType("trackers.core.masks.cutie")
    fake_cutie_module.CutieMaskPropagator = FakeCutieMaskPropagator  # type: ignore[attr-defined]

    monkeypatch.setitem(
        sys.modules,
        "trackers.core.masks.sam",
        fake_sam_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "trackers.core.masks.cutie",
        fake_cutie_module,
    )

    # "cuda:1" only for testing of the passed values. No actual CUDA device is used,
    # because the classes are fake and do not initialize PyTorch or allocate anything.
    # enable_mask_manager=True must lead to creating MaskManager with
    # SAMBoxMaskGenerator and CutieMaskPropagator (being replaced by
    # FakeSAMBoxMaskGenerator and FakeCutieMaskPropagator)
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=True,
        mask_config=McByteMaskConfig(
            device="cuda:1",
            sam_model_type="vit_b",
            cutie_model_type="base-mega",
            cutie_config_name="eval_config",
            cutie_use_amp=False,
            cutie_max_internal_size=576,
            cutie_mem_every=7,
            cutie_use_long_term=False,
            cutie_channels_last=True,
            cutie_compile=True,
            mask_creation_bbox_overlap_threshold=0.7,
        ),
    )

    # Verify a real MaskManager was created
    assert isinstance(tracker.mask_manager, MaskManager)

    # Verify SAM received the right parameters (via McByteMaskConfig at McByteTracker)
    assert created["sam"] == {
        "checkpoint_path": None,
        "model_type": "vit_b",
        "device": "cuda:1",
    }

    # Verify Cutie received the right parameters (via McByteMaskConfig at McByteTracker)
    assert created["cutie"] == {
        "weights_path": None,
        "model_type": "base-mega",
        "config_path": None,
        "config_name": "eval_config",
        "device": "cuda:1",
        "use_amp": False,
        "max_internal_size": 576,
        "mem_every": 7,
        "use_long_term": False,
        "channels_last": True,
        "compile_model": True,
    }

    # Verify the MaskManager-specific threshold
    assert tracker.mask_manager.mask_creation_bbox_overlap_threshold == 0.7


def test_mcbyte_uses_custom_mask_manager_without_real_model_construction() -> None:
    custom_manager = _dummy_mask_manager()

    tracker = McByteTracker(
        enable_cmc=False,
        mask_manager=custom_manager,
    )

    # There is some conditional flow in the MaskManager, hence we are ensuring the one
    # below here.
    assert tracker.mask_manager is custom_manager


def test_mcbyte_rejects_mask_config_with_custom_manager() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be used together",
    ):
        McByteTracker(
            enable_cmc=False,
            mask_manager=_dummy_mask_manager(),
            mask_config=McByteMaskConfig(),
        )


def test_mcbyte_rejects_unused_mask_config() -> None:
    with pytest.raises(
        ValueError,
        match="requires enable_mask_manager=True",
    ):
        McByteTracker(
            enable_cmc=False,
            enable_mask_manager=False,
            mask_config=McByteMaskConfig(),
        )


def test_get_iou_matrix_raises_contextual_error_on_cache_miss() -> None:
    """_get_iou_matrix raises a contextual KeyError when a tracklet is absent from the cache.

    The decode-once map must contain every tracklet passed to the helper (it is built from ``self.tracks`` once per
    ``update()``). A miss is an internal-invariant violation; the helper surfaces it with a message naming the cache
    contract rather than a bare ``KeyError: <id int>``.
    """
    tracker = McByteTracker(enable_cmc=False, enable_mask_manager=False)
    tracklet = McByteTracklet(initial_bbox=np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32))
    detections = np.array([[0.0, 0.0, 10.0, 10.0]])

    with pytest.raises(KeyError, match="decode-once box cache"):
        tracker._get_iou_matrix([tracklet], detections, {})


def test_mcbyte_update_resets_the_seconds_clock() -> None:
    """A re-match clears the seconds clock, so a second miss starts from zero."""
    tracker = McByteTracker(enable_cmc=False)
    box = (100.0, 100.0, 150.0, 200.0)
    for i in range(6):
        tracker.update(_detection(box), timestamp=i / 30.0)
    for i in range(6, 12):
        tracker.update(sv.Detections.empty(), timestamp=i / 30.0)
    assert tracker.tracks[0].time_since_update_seconds > 0.0

    tracker.update(_detection(box), timestamp=12 / 30.0)
    assert tracker.tracks[0].time_since_update_seconds == 0.0

    # Second occurrence: the clock has to advance again, not stay pinned.
    tracker.update(sv.Detections.empty(), timestamp=13 / 30.0)
    assert tracker.tracks[0].time_since_update_seconds > 0.0


def test_mcbyte_tracklet_list_does_not_grow_without_bound_in_timestamp_mode() -> None:
    """Objects that appear once and leave must not accumulate as live tracklets."""
    tracker = McByteTracker(enable_cmc=False)
    frame_index = 0
    peak = 0
    for k in range(12):
        x = 40.0 + (k % 4) * 200.0
        y = 40.0 + (k // 4) * 200.0
        for _ in range(8):
            tracker.update(_detection((x, y, x + 60.0, y + 80.0)), timestamp=frame_index / 30.0)
            frame_index += 1
        for _ in range(4):
            tracker.update(sv.Detections.empty(), timestamp=frame_index / 30.0)
            frame_index += 1
        peak = max(peak, len(tracker.tracks))

    assert peak <= 4, f"tracklet list grew to {peak} for 12 objects seen one at a time"
