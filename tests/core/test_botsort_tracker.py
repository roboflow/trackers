# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""BoT-SORT-specific tracker tests.

Input-mutation and empty-input contracts are covered for all trackers in
test_trackers.py::test_tracker_update_does_not_mutate_input and
test_tracker_update_empty_does_not_mutate_input.
Shared lifecycle/reset/tracked_objects contracts are covered in
test_trackers.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCSRStateEstimator,
    XCYCWHStateEstimator,
    XYXYStateEstimator,
)


def _detection(xyxy: tuple[float, float, float, float], conf: float = 0.9) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([conf], dtype=np.float32),
    )


def _make_frame(h: int = 480, w: int = 640, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


class TestBoTSORTTrackerLifecycle:
    """BoT-SORT-specific lifecycle behavior."""

    def test_first_frame_initializes_track_id(self) -> None:
        """Frame 1 spawns a track with a real ID (BoT-SORT special-case behavior)."""
        tracker = BoTSORTTracker(
            enable_cmc=False,
            minimum_consecutive_frames=3,
            track_activation_threshold=0.5,
        )

        result = tracker.update(_detection((100.0, 100.0, 200.0, 200.0), conf=0.9))

        assert len(result) == 1
        assert result.tracker_id is not None
        assert result.tracker_id[0] >= 0

    def test_instant_activated_track_survives_a_miss(self) -> None:
        """A first-frame track keeps its ID after a single miss (no ID switch).

        With the default ``minimum_consecutive_frames=2`` an instant-activated
        track has only one update, so without sticky maturity it is treated as
        unconfirmed and deleted on a miss — an ID switch when the object returns.
        """
        tracker = BoTSORTTracker(enable_cmc=False)
        obj = (10.0, 10.0, 50.0, 50.0)

        first = tracker.update(_detection(obj))
        track_id = int(first.tracker_id[0])

        tracker.update(sv.Detections.empty())  # no detections: object missed this frame
        assert any(t.tracker_id == track_id for t in tracker.tracks)

        returned = tracker.update(_detection(obj))  # object reappears
        assert track_id in returned.tracker_id.tolist()

    def test_instant_activation_off_track_pruned_on_miss(self) -> None:
        """With instant activation off, an unmatured track is pruned on a miss."""
        tracker = BoTSORTTracker(enable_cmc=False, instant_first_frame_activation=False)
        tracker.update(_detection((10.0, 10.0, 50.0, 50.0)))

        tracker.update(sv.Detections.empty())

        # Track has tracker_id == -1 (not instant-activated); sticky-maturity guard
        # is inactive, so the unmatured unconfirmed track must be pruned.
        assert len(tracker.tracks) == 0

    def test_instant_activated_track_survives_multiple_misses(self) -> None:
        """Track keeps its ID through two consecutive misses (confirmed then lost).

        After the first miss the track sits in confirmed_tracks (time_since_update=1).
        After the second miss it moves to lost_tracks (time_since_update=2).
        get_alive_tracklets must keep it alive via the tracker_id != -1 guard.
        """
        tracker = BoTSORTTracker(enable_cmc=False)
        obj = (10.0, 10.0, 50.0, 50.0)

        first = tracker.update(_detection(obj))
        track_id = int(first.tracker_id[0])

        tracker.update(sv.Detections.empty())  # miss 1: time_since_update=1 → confirmed
        tracker.update(sv.Detections.empty())  # miss 2: time_since_update=2 → lost_tracks

        assert any(t.tracker_id == track_id for t in tracker.tracks)

        returned = tracker.update(_detection(obj))
        assert track_id in returned.tracker_id.tolist()

    @pytest.mark.parametrize(
        "mcf",
        [
            pytest.param(1, id="mcf-1"),
            pytest.param(2, id="mcf-2-default"),
            pytest.param(3, id="mcf-3"),
        ],
    )
    def test_instant_activated_track_survives_miss_across_mcf(self, mcf: int) -> None:
        """Sticky-maturity keeps the track alive on a miss for any mcf value."""
        tracker = BoTSORTTracker(enable_cmc=False, minimum_consecutive_frames=mcf)
        obj = (10.0, 10.0, 50.0, 50.0)

        first = tracker.update(_detection(obj))
        track_id = int(first.tracker_id[0])

        tracker.update(sv.Detections.empty())

        assert any(t.tracker_id == track_id for t in tracker.tracks)

        returned = tracker.update(_detection(obj))
        assert track_id in returned.tracker_id.tolist()

    @pytest.mark.parametrize(
        "estimator_class",
        [XCYCWHStateEstimator, XYXYStateEstimator, XCYCSRStateEstimator],
        ids=["xcycwh", "xyxy", "xcycsr"],
    )
    def test_supports_state_estimator(
        self,
        estimator_class: type[BaseStateEstimator],
    ) -> None:
        """BoTSORTTracker constructs and updates with any supported estimator."""
        tracker = BoTSORTTracker(
            enable_cmc=False,
            state_estimator_class=estimator_class,
            minimum_consecutive_frames=1,
        )
        result = tracker.update(_detection((100.0, 100.0, 200.0, 200.0), conf=0.9))
        assert len(result) == 1


class TestBoTSORTTrackerCMC:
    """BoT-SORT camera motion compensation integration."""

    def test_update_without_frame_skips_cmc_silently(self) -> None:
        """CMC enabled but frame=None must not raise and must track normally."""
        tracker = BoTSORTTracker(enable_cmc=True, minimum_consecutive_frames=1)
        for _ in range(3):
            result = tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame=None)

        assert len(tracker.tracks) == 1
        assert result.tracker_id is not None

    def test_cmc_disabled_ignores_frame(self) -> None:
        """When enable_cmc=False, passing frame to update is harmless."""
        tracker = BoTSORTTracker(enable_cmc=False, minimum_consecutive_frames=1)
        frame = _make_frame()
        for _ in range(3):
            result = tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame=frame)

        assert len(tracker.tracks) == 1
        assert result.tracker_id is not None

    def test_update_with_frame_applies_cmc_without_error(self) -> None:
        """update() with a real textured frame and CMC enabled runs without error."""
        tracker = BoTSORTTracker(
            enable_cmc=True,
            cmc_method="sparseOptFlow",
            minimum_consecutive_frames=1,
        )
        frame = _make_frame()
        for _ in range(5):
            result = tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame=frame)

        assert len(tracker.tracks) == 1
        assert result.tracker_id is not None
        assert result.tracker_id[0] >= 0

    def test_cmc_reset_clears_cmc_state(self) -> None:
        """reset() also resets the internal CMC state."""
        tracker = BoTSORTTracker(
            enable_cmc=True,
            cmc_method="sparseOptFlow",
            minimum_consecutive_frames=1,
        )
        frame = _make_frame()
        for _ in range(3):
            tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame=frame)

        tracker.reset()

        assert tracker.cmc is not None
        assert not tracker.cmc._initialized, "CMC must be uninitialized after tracker reset"
