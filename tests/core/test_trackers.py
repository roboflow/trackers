# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared behavioural contracts for all concrete tracker implementations.

Every test in this file is parametrized over all registered tracker IDs so
that adding a new tracker only requires appending its ID to ``ALL_TRACKER_IDS``.

Sections
--------
1. Input-mutation contract  — update() must not mutate the caller's sv.Detections
2. Track lifecycle / pruning — expiry, time_since_update, short-occlusion survival
3. tracked_objects property  — Kalman-predicted boxes while tracks are alive
4. Tracker-specific contracts — one-off tests tied to a single tracker
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from trackers.core.base import BaseTracker
from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.bytetrack.tracker import ByteTrackTracker
from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.core.sort.tracker import SORTTracker
from trackers.utils.iou import BaseIoU

from .shared_ids import ALL_TRACKER_IDS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _instantiate(tracker_id: str, **kwargs: object) -> BaseTracker:
    """Instantiate any registered tracker by its short ID."""
    import trackers  # noqa: F401 - triggers auto-registration

    info = BaseTracker._lookup_tracker(tracker_id)
    assert info is not None, f"tracker {tracker_id!r} not registered"
    return info.tracker_class(**kwargs)  # type: ignore[arg-type]


def _detection(xyxy: tuple[float, float, float, float]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([0.95], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )


def _no_confidence_detection(xyxy: tuple[float, float, float, float]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )


class _TrackingIoU(BaseIoU):
    """Test-double IoU that records non-empty compute calls."""

    def __init__(self) -> None:
        self.compute_calls = 0

    def _compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        self.compute_calls += 1
        return np.ones((len(boxes_1), len(boxes_2)), dtype=np.float64)


# ==========================================================================
# 1. Input-mutation contract
# ==========================================================================


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
@pytest.mark.parametrize(
    "xyxy, confidence",
    [
        (
            np.array([[10.0, 20.0, 30.0, 40.0]]),
            np.array([0.9]),
        ),
        (
            np.array([[0.0, 0.0, 100.0, 100.0], [50.0, 50.0, 150.0, 150.0]]),
            np.array([0.8, 0.85]),
        ),
    ],
    ids=["single_detection", "two_detections"],
)
def test_tracker_update_does_not_mutate_input(tracker_id: str, xyxy: np.ndarray, confidence: np.ndarray) -> None:
    """update() must not assign tracker_id on the caller's sv.Detections."""
    tracker = _instantiate(tracker_id, minimum_consecutive_frames=1)
    dets = sv.Detections(xyxy=xyxy)
    dets.confidence = confidence
    assert dets.tracker_id is None

    result = tracker.update(dets)

    assert dets.tracker_id is None, "update() must not assign tracker_id on input"
    assert result is not dets, "update() must return a new sv.Detections instance"


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_tracker_update_empty_does_not_mutate_input(tracker_id: str) -> None:
    """update() with empty detections must not mutate input."""
    tracker = _instantiate(tracker_id, minimum_consecutive_frames=1)
    dets = sv.Detections(xyxy=np.zeros((0, 4), dtype=float))
    dets.confidence = np.array([], dtype=float)
    assert dets.tracker_id is None

    result = tracker.update(dets)

    assert dets.tracker_id is None, "update() must not assign tracker_id on empty input"
    assert result is not dets, "update() must return a new sv.Detections instance"


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_tracker_uses_configured_iou_variant(tracker_id: str) -> None:
    """Trackers should use the configured IoU implementation for matching."""
    tracking_iou = _TrackingIoU()
    tracker = _instantiate(tracker_id, minimum_consecutive_frames=1, iou=tracking_iou)
    tracker.update(_detection((100.0, 100.0, 200.0, 200.0)))
    tracker.update(_detection((105.0, 105.0, 205.0, 205.0)))
    assert tracking_iou.compute_calls > 0


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_no_confidence_detections_can_spawn_confirmed_tracks(tracker_id: str) -> None:
    """Missing confidence should behave like usable detections, not suppress tracking."""
    tracker = _instantiate(tracker_id, minimum_consecutive_frames=1)
    detection = _no_confidence_detection((100.0, 100.0, 200.0, 200.0))

    for _ in range(4):
        result = tracker.update(detection)
        if result.tracker_id is not None and np.any(result.tracker_id >= 0):
            return

    raise AssertionError(f"{tracker_id} did not confirm any track for confidence=None detections")


@pytest.mark.parametrize(
    "xyxy_boxes",
    [
        np.array([[100.0, 100.0, 200.0, 200.0]], dtype=np.float32),
        np.array(
            [
                [100.0, 100.0, 200.0, 200.0],
                [400.0, 400.0, 500.0, 500.0],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [10.0, 10.0, 60.0, 60.0],
                [200.0, 200.0, 260.0, 260.0],
                [500.0, 500.0, 560.0, 560.0],
            ],
            dtype=np.float32,
        ),
    ],
    ids=["single_box", "two_boxes", "three_boxes_non_overlapping"],
)
def test_bytetrack_no_confidence_matches_explicit_ones_confidence(xyxy_boxes: np.ndarray) -> None:
    """ByteTrack treats confidence=None the same as all-ones across multi-box batches.

    The batched scenarios exercise the high/low split machinery in
    `ByteTrackTracker.update()` that single-box equivalence cannot trigger; a
    regression that mis-buckets `confidence=None` in a multi-detection batch
    would still pass single-box equality but would diverge here.
    """
    no_confidence_tracker = ByteTrackTracker(minimum_consecutive_frames=1)
    explicit_confidence_tracker = ByteTrackTracker(minimum_consecutive_frames=1)
    class_ids = np.zeros(len(xyxy_boxes), dtype=int)
    detection_without_confidence = sv.Detections(xyxy=xyxy_boxes.copy(), class_id=class_ids.copy())
    detection_with_ones_confidence = sv.Detections(
        xyxy=xyxy_boxes.copy(),
        confidence=np.ones(len(xyxy_boxes), dtype=np.float32),
        class_id=class_ids.copy(),
    )

    no_confidence_tracker.reset()
    no_confidence_results = [no_confidence_tracker.update(detection_without_confidence) for _ in range(4)]
    explicit_confidence_tracker.reset()
    explicit_confidence_results = [explicit_confidence_tracker.update(detection_with_ones_confidence) for _ in range(4)]

    for no_confidence_result, explicit_confidence_result in zip(no_confidence_results, explicit_confidence_results):
        assert len(no_confidence_result) == len(explicit_confidence_result)
        assert no_confidence_result.tracker_id is not None
        assert explicit_confidence_result.tracker_id is not None
        np.testing.assert_array_equal(no_confidence_result.tracker_id, explicit_confidence_result.tracker_id)
        np.testing.assert_array_equal(no_confidence_result.xyxy, explicit_confidence_result.xyxy)


def test_bytetrack_no_confidence_spawns_tracks_below_activation_threshold() -> None:
    """confidence=None must route every detection to Stage 1 even when explicit-low-conf would not spawn.

    Asserts the actual semantic of treating `None` as 1.0: explicit
    confidences that fall under `track_activation_threshold` get suppressed
    in the low-confidence branch, but the same boxes with `confidence=None`
    still produce confirmed tracker IDs.
    """
    activation_threshold = 0.6
    xyxy_boxes = np.array(
        [
            [100.0, 100.0, 200.0, 200.0],
            [400.0, 400.0, 500.0, 500.0],
        ],
        dtype=np.float32,
    )
    class_ids = np.zeros(len(xyxy_boxes), dtype=int)
    detection_without_confidence = sv.Detections(xyxy=xyxy_boxes.copy(), class_id=class_ids.copy())
    detection_with_low_confidence = sv.Detections(
        xyxy=xyxy_boxes.copy(),
        confidence=np.array([0.2, 0.3], dtype=np.float32),
        class_id=class_ids.copy(),
    )

    no_confidence_tracker = ByteTrackTracker(
        minimum_consecutive_frames=1,
        track_activation_threshold=activation_threshold,
        high_conf_det_threshold=activation_threshold,
    )
    low_confidence_tracker = ByteTrackTracker(
        minimum_consecutive_frames=1,
        track_activation_threshold=activation_threshold,
        high_conf_det_threshold=activation_threshold,
    )

    no_confidence_tracker.reset()
    low_confidence_tracker.reset()
    for _ in range(4):
        no_confidence_result = no_confidence_tracker.update(detection_without_confidence)
        low_confidence_result = low_confidence_tracker.update(detection_with_low_confidence)

    assert no_confidence_result.tracker_id is not None
    assert low_confidence_result.tracker_id is not None
    assert np.all(no_confidence_result.tracker_id >= 0), "confidence=None should spawn confirmed tracks for every box"
    assert np.all(low_confidence_result.tracker_id < 0), (
        "explicit low confidence below activation threshold should NOT spawn tracks"
    )


def test_bytetrack_calls_iou_in_low_confidence_branch() -> None:
    """ByteTrack must call the configured IoU in its low-confidence association branch."""
    from trackers import ByteTrackTracker

    spy = _TrackingIoU()
    tracker = ByteTrackTracker(
        iou=spy,
        track_activation_threshold=0.5,
        minimum_consecutive_frames=1,
    )

    # Frame 1: establish a confirmed track with high confidence
    frame1 = sv.Detections(
        xyxy=np.array([[0.0, 0.0, 50.0, 50.0]]),
        confidence=np.array([0.9]),
    )
    tracker.update(frame1)

    calls_before = spy.compute_calls

    # Frame 2: same position but LOW confidence (below track_activation_threshold)
    # This should trigger the low-confidence association branch
    frame2 = sv.Detections(
        xyxy=np.array([[2.0, 2.0, 52.0, 52.0]]),
        confidence=np.array([0.3]),  # below threshold -> low-conf branch
    )
    tracker.update(frame2)

    assert spy.compute_calls > calls_before, "ByteTrack should call iou.compute during low-confidence association"


@pytest.mark.parametrize(
    "tracker_cls",
    [SORTTracker, ByteTrackTracker, OCSORTTracker, BoTSORTTracker],
)
def test_default_iou_is_standard_iou(tracker_cls) -> None:
    """Tracker(iou=None) must default to a standard IoU instance."""
    from trackers.utils.iou import IoU

    tracker = tracker_cls()
    assert isinstance(tracker.iou, IoU), (
        f"{tracker_cls.__name__}: expected iou=IoU() by default, got {type(tracker.iou)}"
    )


def test_fuse_score_ordering_preserved_for_signed_iou() -> None:
    """_fuse_score via normalize_for_fusion must preserve ranking for signed variants."""
    from trackers.core.botsort.utils import _fuse_score
    from trackers.utils.iou import GIoU

    # Two pairs: pair A has better GIoU than pair B
    # Pair A: slight negative overlap region -> GIoU = -0.3
    # Pair B: further apart -> GIoU = -0.8
    iou_matrix = np.array([[-0.3, -0.8]])  # shape (1, 2)
    scores = np.array([0.9, 0.9])  # equal scores — ordering must come from IoU

    metric = GIoU()
    normalized = metric.normalize_for_fusion(iou_matrix)
    fused = _fuse_score(normalized, scores)

    # After normalization, pair A should score higher than pair B
    assert fused[0, 0] > fused[0, 1], (
        "normalize_for_fusion + _fuse_score must preserve GIoU ranking: "
        f"pair A ({fused[0, 0]:.3f}) should > pair B ({fused[0, 1]:.3f})"
    )


# ==========================================================================
# 2. Reset contract
# ==========================================================================


def _run_until_confirmed(
    tracker: BaseTracker,
    detection: sv.Detections,
    max_steps: int = 8,
) -> None:
    """Advance tracker until at least one confirmed track exists."""
    for _ in range(max_steps):
        tracker.update(detection)
        if any(t.tracker_id >= 0 for t in tracker.tracks):
            return
    raise AssertionError("expected at least one confirmed track after warmup")


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_reset_clears_tracks_and_restarts_ids(tracker_id: str) -> None:
    """reset() must clear state and restart tracker IDs from zero."""
    tracker = _instantiate(tracker_id, minimum_consecutive_frames=1)
    det = _detection((100.0, 100.0, 200.0, 200.0))

    _run_until_confirmed(tracker, det)
    assert len(tracker.tracks) > 0

    tracker.reset()

    assert len(tracker.tracks) == 0

    _run_until_confirmed(tracker, det)
    confirmed_ids = [t.tracker_id for t in tracker.tracks if t.tracker_id >= 0]
    assert len(confirmed_ids) > 0
    assert min(confirmed_ids) == 0


# ==========================================================================
# 3. Track lifecycle / pruning
# ==========================================================================
#
# ``time_since_update`` is incremented inside ``predict()``, which every
# tracker calls at the top of its update loop for all live tracks.
# Unmatched tracks therefore have their missed-frame clock advanced
# automatically without any explicit call from the tracker.
#
# These tests pin the contract for every concrete tracker:
# 1. A confirmed track is pruned after ``lost_track_buffer + N`` empty frames.
# 2. ``time_since_update`` actually advances when frames are missed.
# 3. A confirmed track survives a short occlusion.
# 4. Tracks spawned after frame 1 start unconfirmed.
# 5. Tracks are confirmed after minimum_consecutive_frames matches.


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_track_spawned_after_frame_one_starts_unconfirmed(tracker_id: str) -> None:
    """A track first seen on frame 2 must start with tracker_id == -1."""
    tracker = _instantiate(
        tracker_id,
        minimum_consecutive_frames=3,
    )
    det = _detection((100.0, 100.0, 200.0, 200.0))

    tracker.update(sv.Detections.empty())  # frame 1
    r2 = tracker.update(det)  # frame 2: spawns track

    assert r2.tracker_id is not None
    assert r2.tracker_id[0] == -1


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_track_confirms_after_minimum_consecutive_frames(tracker_id: str) -> None:
    """Unconfirmed tracks gain real IDs once minimum_consecutive_frames is reached."""
    tracker = _instantiate(
        tracker_id,
        minimum_consecutive_frames=3,
    )
    det = _detection((100.0, 100.0, 200.0, 200.0))

    tracker.update(sv.Detections.empty())  # frame 1
    r2 = tracker.update(det)  # frame 2: spawn, unconfirmed
    assert r2.tracker_id is not None
    assert r2.tracker_id[0] == -1

    tracker.update(det)  # frame 3
    r4 = tracker.update(det)  # frame 4

    assert r4.tracker_id is not None
    assert r4.tracker_id[0] >= 0


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_track_expires_after_buffer(tracker_id: str) -> None:
    """A confirmed track is pruned after maximum_frames_without_update empty frames."""
    tracker = _instantiate(tracker_id)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_detection(bbox))
    assert len(tracker.tracks) == 1, "track should be alive after warmup"

    buffer = tracker.maximum_frames_without_update
    for _ in range(buffer + 5):
        tracker.update(sv.Detections.empty())

    assert len(tracker.tracks) == 0, "track should be pruned after maximum_frames_without_update empty frames"


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_time_since_update_advances_for_unmatched(tracker_id: str) -> None:
    """`predict()` advances `time_since_update` for unmatched tracks."""
    tracker = _instantiate(tracker_id)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_detection(bbox))

    for _ in range(5):
        tracker.update(sv.Detections.empty())

    assert len(tracker.tracks) == 1, "track is still within lost buffer"
    assert tracker.tracks[0].time_since_update == 5, "time_since_update should reflect 5 missed frames"


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_track_survives_short_occlusion(tracker_id: str) -> None:
    """A confirmed track stays alive across a short detection gap."""
    tracker = _instantiate(tracker_id)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_detection(bbox))
    confirmed_id = tracker.tracks[0].tracker_id
    assert confirmed_id != -1, "track should be confirmed after warmup"

    for _ in range(3):
        tracker.update(sv.Detections.empty())

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].tracker_id == confirmed_id, "confirmed track must survive a short gap"


# ==========================================================================
# 4. tracked_objects property
# ==========================================================================
#
# Verifies that all concrete trackers expose alive tracks via the
# ``tracked_objects`` property as ``sv.Detections`` with Kalman-predicted
# boxes and stable tracker IDs, including the critical occlusion-survival
# scenario that motivates the feature.


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_tracked_objects_empty_before_update(tracker_id: str) -> None:
    """Before the first update, no tracked objects are exposed."""
    tracker = _instantiate(tracker_id)

    result = tracker.tracked_objects
    assert len(result) == 0
    assert result.tracker_id is not None
    assert result.tracker_id.size == 0
    assert result.tracker_id.dtype.kind == "i"
    assert result.confidence is None
    assert result.class_id is None


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_tracked_objects_exposes_mature_track(tracker_id: str) -> None:
    """After enough consistent frames the track is mature and visible."""
    tracker = _instantiate(tracker_id)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_detection(bbox))

    exposed = tracker.tracked_objects
    assert len(exposed) == 1
    assert exposed.tracker_id[0] != -1
    pred = exposed.xyxy[0]
    assert np.allclose(pred, np.array(bbox), atol=10.0), f"predicted box {pred} drifted far from input {bbox}"
    assert exposed.confidence is None
    assert exposed.class_id is None
    assert exposed.xyxy.dtype == np.float32
    assert exposed.tracker_id.dtype.kind == "i"


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_tracked_objects_multiple_simultaneous_tracks(tracker_id: str) -> None:
    """Two mature, simultaneous tracks are both exposed with valid IDs."""
    tracker = _instantiate(tracker_id)

    detections = sv.Detections(
        xyxy=np.array(
            [[10.0, 10.0, 50.0, 50.0], [200.0, 200.0, 300.0, 300.0]],
            dtype=np.float32,
        ),
        confidence=np.array([0.95, 0.95], dtype=np.float32),
        class_id=np.array([0, 0], dtype=int),
    )

    for _ in range(6):
        tracker.update(detections)

    exposed = tracker.tracked_objects
    assert len(exposed) == 2
    assert exposed.xyxy.shape == (2, 4)

    tracker_ids = exposed.tracker_id
    assert tracker_ids.shape == (2,)
    assert np.all(tracker_ids >= 0)
    assert len(set(map(int, tracker_ids))) == 2


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_tracked_objects_survives_occlusion(tracker_id: str) -> None:
    """Track stays in tracked_objects while absent from update() during occlusion."""
    tracker = _instantiate(tracker_id, lost_track_buffer=5, frame_rate=30)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_detection(bbox))

    update_result = tracker.update(sv.Detections.empty())
    assert len(update_result) == 0, "update() must not return occluded track"

    exposed = tracker.tracked_objects
    assert len(exposed) == 1, "tracked_objects must keep alive-but-occluded track"
    assert exposed.tracker_id[0] != -1
    assert np.all(np.isfinite(exposed.xyxy)), "Kalman prediction must be finite"


@pytest.mark.parametrize("tracker_id", ALL_TRACKER_IDS)
def test_tracked_objects_expires_after_buffer(tracker_id: str) -> None:
    """Track is removed from tracked_objects once lost_track_buffer is exceeded."""
    tracker = _instantiate(tracker_id, lost_track_buffer=3, frame_rate=30)
    bbox = (100.0, 100.0, 200.0, 200.0)

    for _ in range(6):
        tracker.update(_detection(bbox))

    max_f = tracker.maximum_frames_without_update
    for _ in range(max_f + 1):
        tracker.update(sv.Detections.empty())

    assert len(tracker.tracked_objects) == 0, "track must expire after buffer exceeded"


# ==========================================================================
# 5. Tracker-specific contracts
# ==========================================================================


def test_bytetrack_consecutive_counter_resets_on_miss() -> None:
    """ByteTrack resets the consecutive-update counter after a missed frame."""
    tracker = ByteTrackTracker(minimum_consecutive_frames=2, lost_track_buffer=30)
    bbox = (100.0, 100.0, 200.0, 200.0)

    tracker.update(_detection(bbox))
    assert tracker.tracks[0].number_of_successful_consecutive_updates == 1
    assert tracker.tracks[0].tracker_id == -1

    tracker.tracks[0].predict()

    tracker.update(_detection(bbox))
    assert tracker.tracks[0].number_of_successful_consecutive_updates == 1
    assert tracker.tracks[0].tracker_id == -1
