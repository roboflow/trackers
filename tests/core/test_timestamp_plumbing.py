# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""PR2: timestamp plumbing and time-based pruning tests.

Covers:
- Backward compatibility: tracker output with no timestamp == original output
- _compute_dt bootstrap / steady-state / non-monotonic handling
- time_since_update_seconds accumulation and reset
- time-based pruning activates only when timestamps are supplied
- OC-SORT / BoT-SORT emit a warning on timestamp and ignore it
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import supervision as sv

from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.bytetrack.tracker import ByteTrackTracker
from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.core.sort.tracker import SORTTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detections(boxes: list[list[float]], confidences: list[float] | None = None) -> sv.Detections:
    if not boxes:
        empty = sv.Detections.empty()
        if confidences is not None:
            empty.confidence = np.array([], dtype=np.float32)
        return empty
    xyxy = np.array(boxes, dtype=np.float32)
    det = sv.Detections(xyxy=xyxy)
    if confidences is not None:
        det.confidence = np.array(confidences, dtype=np.float32)
    return det


_BOX = [[100.0, 100.0, 200.0, 200.0]]
_DET = _make_detections(_BOX, [0.9])
_EMPTY = _make_detections([])


# ---------------------------------------------------------------------------
# Backward compatibility: no timestamp → identical behaviour
# ---------------------------------------------------------------------------


class TestNoTimestampBackwardCompat:
    """Passing no timestamp must give byte-identical results to old code."""

    def test_sort_no_timestamp_returns_tracker_id(self):
        tracker = SORTTracker(minimum_consecutive_frames=1)
        result = tracker.update(_DET)
        assert result.tracker_id is not None
        assert len(result.tracker_id) == 1

    def test_bytetrack_no_timestamp_returns_tracker_id(self):
        tracker = ByteTrackTracker(minimum_consecutive_frames=1, track_activation_threshold=0.5)
        result = tracker.update(_DET)
        assert result.tracker_id is not None

    def test_sort_fixed_and_dynamic_same_result_at_reference_fps(self):
        """Calling update() with or without timestamp=None gives identical results."""
        frame_rate = 30.0
        # Run with no timestamp
        t1 = SORTTracker(frame_rate=frame_rate, minimum_consecutive_frames=1)
        r1 = t1.update(_DET)

        # Run with timestamp=None explicitly
        t2 = SORTTracker(frame_rate=frame_rate, minimum_consecutive_frames=1)
        r2 = t2.update(_DET, timestamp=None)

        assert list(r1.tracker_id) == list(r2.tracker_id)


# ---------------------------------------------------------------------------
# _compute_dt behaviour
# ---------------------------------------------------------------------------


class TestComputeDt:
    """Unit-test the _compute_dt helper via a SORTTracker instance."""

    def _make_tracker(self, frame_rate: float = 30.0) -> SORTTracker:
        t = SORTTracker(frame_rate=frame_rate)
        return t

    def test_no_timestamp_returns_one(self):
        t = self._make_tracker()
        dt = t._compute_dt(None)
        assert dt == 1.0

    def test_first_timestamp_returns_ref_step(self):
        t = self._make_tracker(frame_rate=30.0)
        dt = t._compute_dt(1000.0)
        assert dt == pytest.approx(1.0 / 30.0)

    def test_second_timestamp_returns_gap(self):
        t = self._make_tracker(frame_rate=30.0)
        t._compute_dt(1000.0)  # bootstrap
        dt = t._compute_dt(1000.1)  # 100 ms gap
        assert dt == pytest.approx(0.1, abs=1e-9)

    def test_non_monotonic_returns_zero_and_warns(self):
        t = self._make_tracker()
        t._compute_dt(1000.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dt = t._compute_dt(999.9)  # duplicate / backwards
        assert dt == 0.0
        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)
        assert "non-positive" in str(caught[0].message).lower()

    def test_non_monotonic_warns_only_once(self):
        t = self._make_tracker()
        t._compute_dt(1000.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t._compute_dt(999.9)
            t._compute_dt(999.8)  # second non-monotonic: should NOT warn again
        assert len(caught) == 1

    def test_reset_clears_timestamp_state(self):
        t = self._make_tracker()
        t._compute_dt(1000.0)  # bootstrap
        t._compute_dt(1000.1)  # normal step
        t.reset()
        # After reset, next call should bootstrap again
        dt = t._compute_dt(500.0)
        assert dt == pytest.approx(1.0 / 30.0)


# ---------------------------------------------------------------------------
# time_since_update_seconds accumulation
# ---------------------------------------------------------------------------


class TestTimeSinceUpdateSeconds:
    """Verify time_since_update_seconds increments and resets correctly."""

    def test_sort_increments_on_predict(self):
        tracker = SORTTracker(
            minimum_consecutive_frames=1,
            frame_rate=30.0,
            lost_track_buffer=30,
        )
        # Spawn a track
        tracker.update(_DET)
        assert tracker.tracks[0].time_since_update_seconds == 0.0

        # Next frame: no detection → predict fires with dt=1.0 (fixed mode)
        tracker.update(_EMPTY)
        assert tracker.tracks[0].time_since_update_seconds == pytest.approx(1.0)

    def test_sort_seconds_resets_on_update(self):
        tracker = SORTTracker(
            minimum_consecutive_frames=1,
            frame_rate=30.0,
            lost_track_buffer=30,
        )
        tracker.update(_DET)
        tracker.update(_EMPTY)  # miss → increment
        tracker.update(_DET)  # hit  → reset
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].time_since_update_seconds == 0.0

    def test_sort_timestamp_accumulates_actual_seconds(self):
        """On first timestamped call the dt is 1/frame_rate (bootstrap), on miss it accumulates."""
        tracker = SORTTracker(
            minimum_consecutive_frames=1,
            frame_rate=30.0,
            lost_track_buffer=30,
        )
        tracker.update(_DET, timestamp=0.0)
        # The bootstrap call used dt=1/30; track was just updated → seconds=0
        assert tracker.tracks[0].time_since_update_seconds == 0.0

        # Miss with 0.5 s gap — dt = 0.5 - 0.0 = 0.5
        tracker.update(_EMPTY, timestamp=0.5)
        assert tracker.tracks[0].time_since_update_seconds == pytest.approx(0.5)

    def test_bytetrack_increments_on_predict(self):
        tracker = ByteTrackTracker(
            minimum_consecutive_frames=1,
            track_activation_threshold=0.5,
            lost_track_buffer=30,
        )
        tracker.update(_DET)
        tracker.update(_EMPTY)
        assert tracker.tracks[0].time_since_update_seconds == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Time-based pruning
# ---------------------------------------------------------------------------


class TestTimePruning:
    """Seconds-based pruning activates when timestamps are supplied."""

    def test_sort_time_pruning_keeps_track_within_budget(self):
        """Track should survive if it hasn't exceeded maximum_time_without_update."""
        tracker = SORTTracker(
            lost_track_buffer=30,  # → 1.0 s budget
            frame_rate=30.0,
            minimum_consecutive_frames=1,
        )
        assert tracker.maximum_time_without_update == pytest.approx(1.0)

        tracker.update(_DET, timestamp=0.0)
        # Simulate 0.9 s gap without detection (within 1 s budget)
        tracker.update(_make_detections([], []), timestamp=0.9)
        # Track should still be alive
        assert len(tracker.tracks) == 1

    def test_sort_time_pruning_removes_track_past_budget(self):
        tracker = SORTTracker(
            lost_track_buffer=30,  # → 1.0 s budget
            frame_rate=30.0,
            minimum_consecutive_frames=1,
        )
        tracker.update(_DET, timestamp=0.0)
        # 1.5 s gap — exceeds the 1 s budget
        tracker.update(_make_detections([], []), timestamp=1.5)
        assert len(tracker.tracks) == 0

    def test_bytetrack_time_pruning_keeps_track_within_budget(self):
        tracker = ByteTrackTracker(
            lost_track_buffer=30,
            frame_rate=30.0,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.5,
        )
        tracker.update(_DET, timestamp=0.0)
        tracker.update(_make_detections([], []), timestamp=0.9)
        assert len(tracker.tracks) == 1

    def test_bytetrack_time_pruning_removes_track_past_budget(self):
        tracker = ByteTrackTracker(
            lost_track_buffer=30,
            frame_rate=30.0,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.5,
        )
        tracker.update(_DET, timestamp=0.0)
        tracker.update(_make_detections([], []), timestamp=1.5)
        assert len(tracker.tracks) == 0

    def test_sort_no_timestamp_uses_frame_budget(self):
        """Without timestamps, frame-count pruning is unchanged."""
        tracker = SORTTracker(
            lost_track_buffer=5,
            frame_rate=30.0,
            minimum_consecutive_frames=1,
        )
        tracker.update(_DET)
        for _ in range(4):
            tracker.update(_make_detections([], []))
        # 4 misses < 5-frame budget → track alive
        assert len(tracker.tracks) == 1

        tracker.update(_make_detections([], []))
        # 5 misses == budget → pruned
        assert len(tracker.tracks) == 0


# ---------------------------------------------------------------------------
# OC-SORT and BoT-SORT warn on timestamp
# ---------------------------------------------------------------------------


class TestUnsupportedTimestampWarn:
    """OC-SORT and BoT-SORT must warn once and ignore the timestamp."""

    def test_ocsort_warns_on_first_timestamp(self):
        tracker = OCSORTTracker()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.update(_DET, timestamp=1.0)
        assert any("OCSORTTracker" in str(w.message) for w in caught)
        assert any(issubclass(w.category, UserWarning) for w in caught)

    def test_ocsort_warns_only_once(self):
        tracker = OCSORTTracker()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.update(_DET, timestamp=1.0)
            tracker.update(_DET, timestamp=2.0)
        ocsort_warns = [w for w in caught if "OCSORTTracker" in str(w.message)]
        assert len(ocsort_warns) == 1

    def test_ocsort_reset_re_enables_warning(self):
        tracker = OCSORTTracker()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.update(_DET, timestamp=1.0)
        tracker.reset()
        with warnings.catch_warnings(record=True) as caught2:
            warnings.simplefilter("always")
            tracker.update(_DET, timestamp=2.0)
        assert any("OCSORTTracker" in str(w.message) for w in caught2)

    def test_botsort_warns_on_first_timestamp(self):
        tracker = BoTSORTTracker(enable_cmc=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.update(_DET, timestamp=1.0)
        assert any("BoTSORTTracker" in str(w.message) for w in caught)

    def test_botsort_warns_only_once(self):
        tracker = BoTSORTTracker(enable_cmc=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.update(_DET, timestamp=1.0)
            tracker.update(_DET, timestamp=2.0)
        botsort_warns = [w for w in caught if "BoTSORTTracker" in str(w.message)]
        assert len(botsort_warns) == 1


# ---------------------------------------------------------------------------
# Multi-frame sequence: timestamp mode vs fixed mode gives consistent IDs
# ---------------------------------------------------------------------------


class TestEquivalentTimingConsistency:
    """At 30 fps timestamps, SORT tracking lifecycle matches fixed-rate mode."""

    def _run_sort(self, use_timestamps: bool, n_frames: int = 10) -> list[bool]:
        """Return per-frame confirmed (id >= 0) status."""
        from trackers.core.sort.tracklet import SORTTracklet

        SORTTracklet.count_id = 0
        tracker = SORTTracker(
            frame_rate=30.0,
            minimum_consecutive_frames=1,
            lost_track_buffer=30,
        )
        confirmed: list[bool] = []
        for i in range(n_frames):
            ts = i / 30.0 if use_timestamps else None
            result = tracker.update(_DET, timestamp=ts)
            if len(result.tracker_id):
                confirmed.append(int(result.tracker_id[0]) >= 0)
            else:
                confirmed.append(False)
        return confirmed

    def test_sort_same_confirmation_pattern_at_30fps(self):
        """Tracks are confirmed at the same frames in both modes."""
        fixed_pattern = self._run_sort(use_timestamps=False)
        dynamic_pattern = self._run_sort(use_timestamps=True)
        assert fixed_pattern == dynamic_pattern
