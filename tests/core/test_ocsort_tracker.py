# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""OC-SORT-specific tracker tests.

Generic lifecycle contracts are covered in test_trackers.py via ALL_TRACKER_IDS.
OC-SORT-specific contracts (confidence gating, OCR remap) also live in
test_trackers.py alongside the other tracker-specific one-offs.

This file covers the predict-reuse decode-once cache introduced to remove the
redundant ``get_state_bbox()`` decode after ``predict()`` (mirrors the
cache-contract tests for BoT-SORT/CBIoU in test_botsort_tracker.py and
test_cbiou_tracker.py):
  - Normal predict step: association reads predict()'s cached return, never
    re-decoding via get_state_bbox().
  - Duplicate timestamp: predict() is skipped and the cache is empty, so
    association must fall back to get_state_bbox().
  - Pre-association pruning: a track removed from self.tracks between the
    cache being built and the cache being read must not corrupt the
    surviving track's id()-keyed lookup.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from trackers.core.ocsort.tracker import OCSORTTracker


def _detection(xyxy: tuple[float, float, float, float], conf: float = 0.9) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([conf], dtype=np.float32),
    )


def test_update_reuses_predicted_box_without_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Association after a normal predict step reads predict()'s cached box, never re-decoding it.

    OC-SORT builds an id()-keyed cache of each tracklet's ``predict()`` return value once per ``update()`` call and
    reuses it for the primary (OCM) association stage. If association instead re-decoded state through
    ``get_state_bbox()``, this test's monkeypatch makes that call raise -- guarding against a silent revert to the pre-
    optimization per-track double-decode.
    """
    tracker = OCSORTTracker(minimum_consecutive_frames=1)
    tracker.update(_detection((100.0, 100.0, 200.0, 200.0)))  # frame 1: spawn track

    tracklet = tracker.tracks[0]

    def _fail() -> np.ndarray:
        raise AssertionError("get_state_bbox must not be called; boxes come from the predict() cache")

    monkeypatch.setattr(tracklet, "get_state_bbox", _fail)

    result = tracker.update(_detection((101.0, 101.0, 201.0, 201.0)))  # frame 2: normal predict + association

    assert result.tracker_id is not None
    assert int(result.tracker_id[0]) != -1


def test_duplicate_timestamp_falls_back_to_get_state_bbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """A duplicate timestamp skips predict(), leaving the cache empty; association must decode via get_state_bbox().

    Duplicate timestamps set ``frame_step=0``, so ``_predict_tracklets`` returns an empty cache instead of calling
    ``predict()``. OC-SORT's fallback (``else [t.get_state_bbox() for t in self.tracks]``) must still run so association
    has a box to compare against -- an empty ``predicted_boxes`` array would silently drop every track from matching.
    """
    tracker = OCSORTTracker(minimum_consecutive_frames=1)
    box = (100.0, 100.0, 200.0, 200.0)
    tracker.update(_detection(box), timestamp=1.0)  # frame 1: spawn track

    tracklet = tracker.tracks[0]
    calls: list[None] = []
    original_get_state_bbox = tracklet.get_state_bbox

    def _spy() -> np.ndarray:
        calls.append(None)
        return original_get_state_bbox()

    monkeypatch.setattr(tracklet, "get_state_bbox", _spy)

    with pytest.warns(UserWarning, match="duplicate timestamp"):
        result = tracker.update(_detection(box), timestamp=1.0)  # frame 2: duplicate timestamp

    assert len(calls) == 1
    assert result.tracker_id is not None
    assert int(result.tracker_id[0]) != -1


def test_prune_between_predict_and_association_uses_correct_cached_box(monkeypatch: pytest.MonkeyPatch) -> None:
    """A track pruned between predict() and association must not corrupt the surviving track's cache entry.

    Two tracks are predicted in the same ``update()`` call and keyed into the cache by ``id(tracklet)``. Variable-FPS
    pruning then removes one track from ``self.tracks`` before the surviving track's predicted box is looked up. If the
    lookup ever degraded from id()-keyed to positional indexing, the surviving track would silently receive the pruned
    track's (far-away) predicted box and fail to re-associate with its detection.
    """
    tracker = OCSORTTracker(minimum_consecutive_frames=1)
    box_a = (10.0, 10.0, 50.0, 50.0)
    box_b = (500.0, 500.0, 540.0, 540.0)
    both = sv.Detections(
        xyxy=np.array([box_a, box_b], dtype=np.float32),
        confidence=np.array([0.9, 0.9], dtype=np.float32),
    )
    tracker.update(both, timestamp=0.0)  # frame 1: spawn A and B
    tracker.update(both, timestamp=1.0)  # frame 2: both match, real IDs assigned
    assert len(tracker.tracks) == 2

    track_a = next(t for t in tracker.tracks if t.last_observation[0] < 100.0)
    track_b = next(t for t in tracker.tracks if t.last_observation[0] >= 100.0)
    track_b_id = track_b.tracker_id

    # Force A past its lost-track time budget so it is pruned before association on the next
    # update, while B (just matched, time_since_update_seconds == 0) stays comfortably inside it.
    track_a.time_since_update_seconds = tracker.maximum_time_without_update + 100.0
    monkeypatch.setattr(track_a, "predict", lambda timing: np.array([-1000.0, -1000.0, -960.0, -960.0]))

    def _fail() -> np.ndarray:
        raise AssertionError("get_state_bbox must not be called; box comes from the predict() cache")

    monkeypatch.setattr(track_b, "get_state_bbox", _fail)

    result = tracker.update(_detection(box_b, conf=0.9), timestamp=1.0333)  # frame 3: A pruned pre-association

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0] is track_b
    assert result.tracker_id is not None
    assert int(result.tracker_id[0]) == track_b_id
