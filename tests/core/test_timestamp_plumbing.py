# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Timestamp plumbing and time-based pruning tests.

``_predict_timing`` unit tests run on SORT only (shared ``BaseTracker`` impl).
Integration behaviour is parametrized over timestamp-aware trackers.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
import supervision as sv

from trackers.core.base import BaseTracker
from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.botsort.tracklet import BoTSORTTracklet
from trackers.core.bytetrack.tracker import ByteTrackTracker
from trackers.core.bytetrack.tracklet import ByteTrackTracklet
from trackers.core.ocsort.tracker import OCSORTTracker
from trackers.core.ocsort.tracklet import OCSORTTracklet
from trackers.core.sort.tracker import SORTTracker
from trackers.core.sort.tracklet import SORTTracklet
from trackers.utils.base_tracklet import BaseTracklet

TIMESTAMP_AWARE_TRACKERS: list[Any] = [
    pytest.param(SORTTracker, SORTTracklet, {}, id="sort"),
    pytest.param(
        ByteTrackTracker,
        ByteTrackTracklet,
        {"track_activation_threshold": 0.5},
        id="bytetrack",
    ),
    pytest.param(
        OCSORTTracker,
        OCSORTTracklet,
        {"high_conf_det_threshold": 0.5},
        id="ocsort",
    ),
    pytest.param(
        BoTSORTTracker,
        BoTSORTTracklet,
        {"enable_cmc": False, "track_activation_threshold": 0.5},
        id="botsort",
    ),
]

PREDICT_TIMING_TRACKER: list[Any] = [
    pytest.param(SORTTracker, SORTTracklet, {}, id="sort"),
]

FRAME_BUDGET_TRACKERS: list[Any] = [
    pytest.param(SORTTracker, SORTTracklet, {}, id="sort"),
    pytest.param(
        BoTSORTTracker,
        BoTSORTTracklet,
        {"enable_cmc": False, "track_activation_threshold": 0.5},
        id="botsort",
    ),
]


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


def _make_timestamp_aware_tracker(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
    **kwargs: Any,
) -> BaseTracker:
    tracklet_cls.count_id = 0
    params = {
        "minimum_consecutive_frames": 1,
        "frame_rate": 30.0,
        "lost_track_buffer": 30,
        **extra_kwargs,
        **kwargs,
    }
    return tracker_cls(**params)


_BOX = [[100.0, 100.0, 200.0, 200.0]]
_DET = _make_detections(_BOX, [0.9])
_EMPTY = _make_detections([])


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    TIMESTAMP_AWARE_TRACKERS,
)
def test_explicit_none_timestamp_matches_omitted(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    frame_rate = 30.0
    t1 = _make_timestamp_aware_tracker(tracker_cls, tracklet_cls, extra_kwargs, frame_rate=frame_rate)
    r1 = t1.update(_DET)

    t2 = _make_timestamp_aware_tracker(tracker_cls, tracklet_cls, extra_kwargs, frame_rate=frame_rate)
    r2 = t2.update(_DET, timestamp=None)

    assert list(r1.tracker_id) == list(r2.tracker_id)


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    PREDICT_TIMING_TRACKER,
)
def test_predict_timing_no_timestamp_uses_one_frame_step(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(tracker_cls, tracklet_cls, extra_kwargs)
    timing = tracker._predict_timing(None)
    assert timing.frame_step == 1.0
    assert timing.elapsed_seconds is None
    assert not timing.skip_predict


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    PREDICT_TIMING_TRACKER,
)
def test_predict_timing_first_timestamp_bootstraps(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(tracker_cls, tracklet_cls, extra_kwargs, frame_rate=30.0)
    timing = tracker._predict_timing(1000.0)
    assert timing.elapsed_seconds == pytest.approx(1.0 / 30.0)
    assert timing.frame_step == pytest.approx(1.0)
    assert not timing.skip_predict


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    PREDICT_TIMING_TRACKER,
)
def test_predict_timing_second_timestamp_returns_gap(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(tracker_cls, tracklet_cls, extra_kwargs, frame_rate=30.0)
    tracker._predict_timing(1000.0)
    timing = tracker._predict_timing(1000.1)
    assert timing.elapsed_seconds == pytest.approx(0.1, abs=1e-9)
    assert timing.frame_step == pytest.approx(3.0)
    assert not timing.skip_predict


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    PREDICT_TIMING_TRACKER,
)
def test_predict_timing_constant_fps_maps_to_one_frame_step(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(tracker_cls, tracklet_cls, extra_kwargs, frame_rate=25.0)
    timing = tracker._predict_timing(0.04)
    assert timing.elapsed_seconds == pytest.approx(0.04)
    assert timing.frame_step == pytest.approx(1.0)


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    PREDICT_TIMING_TRACKER,
)
def test_predict_timing_frame_gap_scales_frame_step(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(tracker_cls, tracklet_cls, extra_kwargs, frame_rate=25.0)
    tracker._predict_timing(0.04)
    timing = tracker._predict_timing(0.24)
    assert timing.elapsed_seconds == pytest.approx(0.20)
    assert timing.frame_step == pytest.approx(5.0)


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    PREDICT_TIMING_TRACKER,
)
def test_predict_timing_non_monotonic_skips_predict_and_warns_each_time(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(tracker_cls, tracklet_cls, extra_kwargs)
    tracker._predict_timing(1000.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        timing = tracker._predict_timing(999.9)
        tracker._predict_timing(999.8)
    assert timing.skip_predict
    assert timing.elapsed_seconds == 0.0
    assert len(caught) == 2
    assert all(issubclass(w.category, UserWarning) for w in caught)
    assert all("non-positive" in str(w.message).lower() for w in caught)


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    PREDICT_TIMING_TRACKER,
)
def test_predict_timing_reset_clears_timestamp_state(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(tracker_cls, tracklet_cls, extra_kwargs)
    tracker._predict_timing(1000.0)
    tracker._predict_timing(1000.1)
    tracker.reset()
    timing = tracker._predict_timing(500.0)
    assert timing.elapsed_seconds == pytest.approx(1.0 / 30.0)
    assert timing.frame_step == pytest.approx(1.0)


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    TIMESTAMP_AWARE_TRACKERS,
)
def test_fixed_mode_does_not_accumulate_seconds(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(tracker_cls, tracklet_cls, extra_kwargs)
    tracker.update(_DET)
    tracker.update(_EMPTY)
    assert tracker.tracks[0].time_since_update_seconds == 0.0


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    TIMESTAMP_AWARE_TRACKERS,
)
def test_seconds_reset_on_match(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(tracker_cls, tracklet_cls, extra_kwargs)
    tracker.update(_DET)
    tracker.update(_EMPTY)
    tracker.update(_DET)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].time_since_update_seconds == 0.0


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    TIMESTAMP_AWARE_TRACKERS,
)
def test_timestamp_mode_accumulates_elapsed_seconds_on_miss(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(tracker_cls, tracklet_cls, extra_kwargs)
    tracker.update(_DET, timestamp=0.0)
    assert tracker.tracks[0].time_since_update_seconds == 0.0
    tracker.update(_EMPTY, timestamp=0.5)
    assert tracker.tracks[0].time_since_update_seconds == pytest.approx(0.5)


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    TIMESTAMP_AWARE_TRACKERS,
)
def test_time_pruning_keeps_track_within_budget(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(
        tracker_cls,
        tracklet_cls,
        extra_kwargs,
        lost_track_buffer=30,
    )
    assert tracker.maximum_time_without_update == pytest.approx(1.0)
    tracker.update(_DET, timestamp=0.0)
    tracker.update(_make_detections([], []), timestamp=0.9)
    assert len(tracker.tracks) == 1


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    TIMESTAMP_AWARE_TRACKERS,
)
def test_time_pruning_removes_track_past_budget(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(
        tracker_cls,
        tracklet_cls,
        extra_kwargs,
        lost_track_buffer=30,
    )
    tracker.update(_DET, timestamp=0.0)
    tracker.update(_make_detections([], []), timestamp=1.5)
    assert len(tracker.tracks) == 0


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    FRAME_BUDGET_TRACKERS,
)
def test_no_timestamp_uses_frame_budget(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(
        tracker_cls,
        tracklet_cls,
        extra_kwargs,
        lost_track_buffer=5,
    )
    tracker.update(_DET)
    for _ in range(4):
        tracker.update(_make_detections([], []))
    assert len(tracker.tracks) == 1
    tracker.update(_make_detections([], []))
    assert len(tracker.tracks) == 0


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    FRAME_BUDGET_TRACKERS,
)
def test_mixed_timestamp_then_fixed_uses_frame_budget(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    tracker = _make_timestamp_aware_tracker(
        tracker_cls,
        tracklet_cls,
        extra_kwargs,
        lost_track_buffer=3,
    )
    tracker.update(_DET, timestamp=0.0)
    assert tracker.tracks[0].time_since_update_seconds == 0.0
    for _ in range(2):
        tracker.update(_make_detections([], []))
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].time_since_update_seconds == 0.0
    tracker.update(_make_detections([], []))
    assert len(tracker.tracks) == 0


def _confirmation_pattern(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
    *,
    use_timestamps: bool,
    n_frames: int = 10,
    frame_rate: float = 30.0,
) -> list[bool]:
    tracker = _make_timestamp_aware_tracker(
        tracker_cls,
        tracklet_cls,
        extra_kwargs,
        frame_rate=frame_rate,
        lost_track_buffer=30,
    )
    confirmed: list[bool] = []
    for i in range(n_frames):
        ts = i / frame_rate if use_timestamps else None
        result = tracker.update(_DET, timestamp=ts)
        if len(result.tracker_id):
            confirmed.append(int(result.tracker_id[0]) >= 0)
        else:
            confirmed.append(False)
    return confirmed


@pytest.mark.parametrize(
    "tracker_cls, tracklet_cls, extra_kwargs",
    TIMESTAMP_AWARE_TRACKERS,
)
def test_same_confirmation_pattern_at_reference_fps(
    tracker_cls: type[BaseTracker],
    tracklet_cls: type[BaseTracklet],
    extra_kwargs: dict[str, Any],
) -> None:
    fixed_pattern = _confirmation_pattern(tracker_cls, tracklet_cls, extra_kwargs, use_timestamps=False)
    dynamic_pattern = _confirmation_pattern(tracker_cls, tracklet_cls, extra_kwargs, use_timestamps=True)
    assert fixed_pattern == dynamic_pattern
