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
from trackers.core.botsort.tracklet import BoTSORTTracklet
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


def _translation(dx: float, dy: float) -> np.ndarray:
    """Pure-translation 2x3 affine (previous -> current frame)."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def _state_shift_after_one_step(tracker: BoTSORTTracker, baseline: BoTSORTTracker, h_cmc: np.ndarray) -> np.ndarray:
    """Seed both trackers with one box, advance one empty step (h_cmc only on ``tracker``), return the bbox delta.

    The empty step matters: a detection would re-update the Kalman state and mask the CMC shift.
    """
    seed = _detection((100.0, 100.0, 200.0, 200.0))
    tracker.update(seed)
    baseline.update(seed)
    tracker.update(sv.Detections.empty(), h_cmc=h_cmc)
    baseline.update(sv.Detections.empty())
    return tracker.tracks[0].get_state_bbox() - baseline.tracks[0].get_state_bbox()


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

        With the default ``minimum_consecutive_frames=2`` an instant-activated track has only one update, so without
        sticky maturity it is treated as unconfirmed and deleted on a miss — an ID switch when the object returns.
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

        After the first miss the track sits in confirmed_tracks (time_since_update=1). After the second miss it moves to
        lost_tracks (time_since_update=2). get_alive_tracklets must keep it alive via the tracker_id != -1 guard.
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
        """Update() with a real textured frame and CMC enabled runs without error."""
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
        """Reset() also resets the internal CMC state."""
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


class TestBoTSORTTrackerExternalCMC:
    """BoT-SORT with a precomputed camera-motion transform passed via ``update(h_cmc=...)``."""

    def test_h_cmc_shifts_state_without_internal_estimate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A translation h_cmc shifts the predicted box by (dx, dy) and never calls the internal estimator.

        Twin trackers see identical input except for h_cmc, so the state-bbox delta isolates the compensation; the spy
        proves the external transform replaces, rather than adds to, the frame-based estimate.
        """
        tracker = BoTSORTTracker(enable_cmc=True)
        assert tracker.cmc is not None
        estimate_calls: list[tuple[object, ...]] = []
        monkeypatch.setattr(tracker.cmc, "estimate", lambda *args: estimate_calls.append(args))

        shift = _state_shift_after_one_step(tracker, BoTSORTTracker(enable_cmc=True), _translation(12.0, -7.0))

        np.testing.assert_allclose(shift, [12.0, -7.0, 12.0, -7.0], atol=1e-9)
        assert estimate_calls == []

    def test_h_cmc_applied_when_enable_cmc_false(self) -> None:
        """enable_cmc=False only disables the internal estimator; a precomputed h_cmc is still applied.

        Users relying solely on external registration should not need to build an unused internal estimator.
        """
        shift = _state_shift_after_one_step(
            BoTSORTTracker(enable_cmc=False), BoTSORTTracker(enable_cmc=False), _translation(12.0, -7.0)
        )

        np.testing.assert_allclose(shift, [12.0, -7.0, 12.0, -7.0], atol=1e-9)

    def test_positional_third_argument_binds_timestamp(self) -> None:
        """Update(detections, frame, timestamp) keeps the BaseTracker positional order.

        h_cmc is keyword-only, so existing positional callers passing a timestamp third must not have it rebound to the
        transform.
        """
        tracker = BoTSORTTracker(enable_cmc=False)

        tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), None, 1.5)

        assert tracker._last_timestamp == 1.5

    def test_positional_h_cmc_rejected(self) -> None:
        """h_cmc passed as a fourth positional argument raises TypeError instead of binding silently."""
        tracker = BoTSORTTracker(enable_cmc=False)

        with pytest.raises(TypeError):
            tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), None, None, _translation(1.0, 1.0))  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("bad_h_cmc", "message"),
        [
            pytest.param(np.array([[1.0, 0.0, np.nan], [0.0, 1.0, 0.0]]), "non-finite", id="nan"),
            pytest.param(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, np.inf]]), "non-finite", id="inf"),
            pytest.param(np.eye(3), "3x3 homographies", id="homography-3x3"),
            pytest.param(np.eye(2), "shape", id="shape-2x2"),
            pytest.param(np.zeros(6), "shape", id="flat-6"),
            pytest.param(np.array(1.0), "shape", id="scalar"),
            pytest.param(np.array([["a", "b", "c"], ["d", "e", "f"]]), "real numbers", id="strings"),
            pytest.param(np.eye(2, 3, dtype=np.complex128), "real numbers", id="complex"),
        ],
    )
    def test_invalid_h_cmc_raises_and_leaves_state_unchanged(self, bad_h_cmc: np.ndarray, message: str) -> None:
        """Malformed h_cmc raises a ValueError naming h_cmc before predict, so tracker state is untouched.

        A NaN or wrongly shaped transform applied to the Kalman state would poison every later association; failing fast
        at the boundary keeps the tracker usable after the caller fixes the input.
        """
        tracker = BoTSORTTracker(enable_cmc=False)
        tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), timestamp=0.0)
        box_before = tracker.tracks[0].get_state_bbox().copy()

        with pytest.raises(ValueError, match=f"h_cmc.*{message}"):
            tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), timestamp=1.0, h_cmc=bad_h_cmc)

        assert (tracker.frame_id, tracker._last_timestamp, len(tracker.tracks)) == (1, 0.0, 1)
        np.testing.assert_array_equal(tracker.tracks[0].get_state_bbox(), box_before)

    def test_frame_and_h_cmc_together_raises(self) -> None:
        """Passing both CMC sources is ambiguous and raises before any state change."""
        tracker = BoTSORTTracker(enable_cmc=True)

        with pytest.raises(ValueError, match="both frame and h_cmc"):
            tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame=_make_frame(), h_cmc=_translation(1.0, 1.0))

        assert (tracker.frame_id, len(tracker.tracks)) == (0, 0)

    def test_h_cmc_resets_internal_estimator(self) -> None:
        """Switching frame -> h_cmc resets the internal estimator so a later frame call cannot double-compensate.

        Without the reset, the next frame-based estimate would measure motion against the pre-switch frame and re-apply
        motion the external transforms already compensated.
        """
        tracker = BoTSORTTracker(enable_cmc=True, minimum_consecutive_frames=1)
        assert tracker.cmc is not None
        frame = _make_frame()
        for _ in range(2):
            tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame=frame)
        initialized_after_frames = tracker.cmc._initialized

        tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), h_cmc=_translation(0.0, 0.0))

        assert (initialized_after_frames, tracker.cmc._initialized) == (True, False)


def test_get_iou_matrix_reads_cache_without_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """_get_iou_matrix reads predicted boxes from the per-frame cache, never re-decoding.

    P4-2 decode-once: each tracklet's box is decoded a single time per ``update()``
    and passed to all three association stages via a map keyed by ``id()``. The
    helper must read that map and never call ``get_state_bbox`` itself — doing so
    would reintroduce the per-stage redundant decode the cache exists to remove.
    """
    tracker = BoTSORTTracker(enable_cmc=False)
    tracklet = BoTSORTTracklet(np.array([0.0, 0.0, 10.0, 10.0]))

    def _fail() -> np.ndarray:
        raise AssertionError("get_state_bbox must not be called; boxes come from the cache")

    monkeypatch.setattr(tracklet, "get_state_bbox", _fail)
    cached_box = np.array([5.0, 5.0, 15.0, 15.0])
    detections = np.array([[5.0, 5.0, 15.0, 15.0]])  # identical to the cached box -> IoU 1.0

    result = tracker._get_iou_matrix([tracklet], detections, {id(tracklet): cached_box})

    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(1.0)


def test_get_iou_matrix_raises_contextual_error_on_cache_miss() -> None:
    """_get_iou_matrix raises a contextual KeyError when a tracklet is absent from the cache.

    The decode-once map must contain every tracklet passed to the helper (it is built from ``self.tracks`` once per
    ``update()``). A miss is an internal-invariant violation; the helper surfaces it with a message naming the cache
    contract rather than a bare ``KeyError: <id int>``.
    """
    tracker = BoTSORTTracker(enable_cmc=False)
    tracklet = BoTSORTTracklet(np.array([0.0, 0.0, 10.0, 10.0]))
    detections = np.array([[0.0, 0.0, 10.0, 10.0]])

    with pytest.raises(KeyError, match="decode-once box cache"):
        tracker._get_iou_matrix([tracklet], detections, {})
