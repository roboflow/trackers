# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Long-term identity integration through the ordinary McByte++ update API."""

import numpy as np
import pytest
import supervision as sv

from trackers import LongTermReIDConfig, McBytePlusPlusTracker
from trackers.core.masks.dummy import DummyBoxMaskGenerator, DummyIdentityMaskPropagator
from trackers.core.masks.manager import MaskManager
from trackers.utils.conditional_cmc import ConditionalCMC


class ConstantEncoder:
    """Return the same appearance for controlled disappearance/return tests."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        self.frames: list[np.ndarray] = []

    def extract_embeddings(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        self.calls.append(detections.xyxy.copy())
        self.frames.append(frame.copy())
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(detections), 1))


def observation(x: float = 20) -> sv.Detections:
    """Create a valid human-sized detector box away from image borders."""
    return sv.Detections(xyxy=np.array([[x, 20, x + 40, 110]]), confidence=np.array([0.9]))


def test_default_camera_pause_accounts_for_birth_before_confirmation(monkeypatch) -> None:
    """Default camera-motion pauses preserve the decision window and buffered history."""
    model = ConstantEncoder()
    tracker = McBytePlusPlusTracker(
        reid_model=model,
        minimum_consecutive_frames=3,
        reid_joining_config=LongTermReIDConfig(maximum_reconnection_age=1),
    )
    assert isinstance(tracker.cmc, ConditionalCMC)
    calls = []

    def estimate(frame, boxes):
        calls.append(tracker.frame_id)
        tracker.cmc._last_estimate_available = True
        dx = 8 if tracker.frame_id in (3, 4, 5) else 0
        return np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]])

    monkeypatch.setattr(tracker.cmc, "estimate", estimate)
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    tracker.update(sv.Detections.empty(), frame)
    for step in range(2, 7):
        current = tracker.update(observation(), frame)
        assert current.tracker_id.tolist() == [-1]
        assert current.data["reid_pending"].tolist() == [step >= 4]
    current = tracker.update(observation(), frame)
    assert current.tracker_id.tolist() == [0]
    assert len(model.calls) == 1  # first suitable confirmed image was frame 6
    assert calls == list(range(1, 8))  # one CMC pass per supplied image
    history = tracker.pop_finalized_results()
    assert list(history) == [4, 5, 6]
    assert all(row.tracker_id.tolist() == [0] for row in history.values())
    tracker.reset()
    assert not tracker.cmc.last_estimate_available


@pytest.mark.parametrize("camera_pause", [False, True])
def test_occlusion_pause_preserves_birth_window_through_confirmation(monkeypatch, camera_pause: bool) -> None:
    """Occluded tentative updates count once, even alongside a CMC pause."""
    model = ConstantEncoder()
    tracker = McBytePlusPlusTracker(
        reid_model=model,
        enable_cmc=camera_pause,
        minimum_consecutive_frames=4,
        reid_joining_config=LongTermReIDConfig(maximum_reconnection_age=1),
    )
    if camera_pause:
        assert isinstance(tracker.cmc, ConditionalCMC)

        def estimate(frame, boxes):
            tracker.cmc._last_estimate_available = True
            dx = 8 if tracker.frame_id in (3, 4, 5) else 0
            return np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]])

        monkeypatch.setattr(tracker.cmc, "estimate", estimate)
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    tracker.update(sv.Detections.empty(), frame)
    for step in range(2, 7):
        detections = observation()
        if step in (3, 4, 5):
            # The foreground detection does not itself qualify for a new track.
            detections = sv.Detections(
                xyxy=np.array([[20, 20, 60, 110], [20, 50, 60, 140]]),
                confidence=np.array([0.9, 0.1]),
            )
        current = tracker.update(detections, frame)
        assert current.tracker_id[0] == -1
        assert bool(current.data["reid_pending"][0]) == (step >= 5)
    assert tracker.update(observation(), frame).tracker_id.tolist() == [0]
    assert len(model.calls) == 1  # first clear confirmed crop: step 6
    history = tracker.pop_finalized_results()
    assert list(history) == [5, 6]
    assert all(row.tracker_id.tolist() == [0] for row in history.values())


def test_camera_pause_does_not_reuse_signal_on_missing_duplicate_or_unavailable_images(monkeypatch) -> None:
    model = ConstantEncoder()
    tracker = McBytePlusPlusTracker(
        reid_model=model,
        enable_reid_camera_motion_pause=True,
        reid_joining_config=LongTermReIDConfig(minimum_feature_frame_gap=1),
    )
    assert isinstance(tracker.cmc, ConditionalCMC)
    calls = []

    def estimate(frame, boxes):
        calls.append(tracker.frame_id)
        tracker.cmc._last_estimate_available = tracker.frame_id != 5
        return np.array([[1.0, 0.0, 8.0 if tracker.frame_id == 2 else 0.0], [0.0, 1.0, 0.0]])

    monkeypatch.setattr(tracker.cmc, "estimate", estimate)
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    tracker.update(observation(), frame, timestamp=0)
    tracker.update(observation(), frame, timestamp=0.04)
    assert len(model.calls) == 1
    tracker.update(observation(), None, timestamp=0.08)
    with pytest.warns(UserWarning, match="duplicate"):
        tracker.update(observation(), frame, timestamp=0.08)
    assert len(model.calls) == 1
    tracker.update(observation(), frame, timestamp=0.12)  # unavailable estimate allows sampling
    assert len(model.calls) == 2
    assert calls == [1, 2, 5]


@pytest.mark.parametrize("threshold", [0, -1, np.nan, np.inf, True])
def test_camera_pause_rejects_invalid_threshold(threshold) -> None:
    with pytest.raises(ValueError, match="reid_camera_motion_threshold"):
        McBytePlusPlusTracker(reid_camera_motion_threshold=threshold)


def test_camera_pause_compatibility_with_legacy_and_disabled_cmc() -> None:
    with pytest.raises(ValueError, match="conditional CMC"):
        McBytePlusPlusTracker(reid_model=ConstantEncoder(), enable_conditional_cmc=False)
    legacy_model = ConstantEncoder()
    legacy = McBytePlusPlusTracker(
        reid_model=legacy_model, enable_conditional_cmc=False, enable_reid_camera_motion_pause=False
    )
    legacy.update(observation(), np.zeros((160, 240, 3), dtype=np.uint8))
    assert len(legacy_model.calls) == 1
    model = ConstantEncoder()
    tracker = McBytePlusPlusTracker(reid_model=model, enable_cmc=False)
    tracker.update(observation(), np.zeros((160, 240, 3), dtype=np.uint8))
    assert len(model.calls) == 1


def test_reconnection_preserves_internal_owner_and_delivers_only_history() -> None:
    """The old public ID is recovered once, independently of the new tracklet ID."""
    model = ConstantEncoder()
    tracker = McBytePlusPlusTracker(
        enable_cmc=False,
        lost_track_buffer=1,
        reid_model=model,
        reid_joining_config=LongTermReIDConfig(minimum_feature_samples=2, minimum_feature_frame_gap=1),
    )
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    assert tracker.update(observation(), frame).tracker_id.tolist() == [0]
    assert tracker.update(observation(), frame).tracker_id.tolist() == [0]
    tracker.update(sv.Detections.empty(), frame)
    tracker.update(sv.Detections.empty(), frame)

    tentative = tracker.update(observation(120), frame)
    assert tentative.tracker_id.tolist() == [-1]
    assert tentative.data["reid_pending"].tolist() == [False]
    pending = tracker.update(observation(120), frame)
    assert pending.tracker_id.tolist() == [-1]
    assert pending.data["reid_pending"].tolist() == [True]
    assert tracker.pop_finalized_results() == {}

    returned = tracker.update(observation(120), frame)
    assert returned.tracker_id.tolist() == [0]
    assert returned.data["tracklet_id"].tolist() == [1]
    assert tracker.tracks[0].tracker_id == 1
    assert tracker.tracks[0].public_id == 0
    historical = tracker.pop_finalized_results()
    assert list(historical) == [6]
    assert historical[6].tracker_id.tolist() == [0]
    assert historical[6].data["reid_pending"].tolist() == [False]
    assert tracker.pop_finalized_results() == {}


def test_masks_use_new_internal_id_after_public_identity_reconnection() -> None:
    """Re-ID changes output labels; newly generated masks belong to tracklet 1."""
    model = ConstantEncoder()
    manager = MaskManager(DummyBoxMaskGenerator(), DummyIdentityMaskPropagator())
    tracker = McBytePlusPlusTracker(
        enable_cmc=False,
        mask_manager=manager,
        enable_mask_refresh=False,
        lost_track_buffer=1,
        reid_model=model,
    )
    bgr = np.full((160, 240, 3), [11, 35, 71], dtype=np.uint8)
    tracker.update(observation(), bgr)
    tracker.update(observation(), bgr)
    tracker.update(sv.Detections.empty(), bgr)
    tracker.update(sv.Detections.empty(), bgr)
    tracker.update(observation(120), bgr)
    returned = tracker.update(observation(120), bgr)
    assert returned.tracker_id.tolist() == [0]
    assert returned.data["tracklet_id"].tolist() == [1]
    tracker.update(observation(120), bgr)
    assert tracker.last_mask_output is not None
    assert set(tracker.last_mask_output.tracklet_mask_dict) == {1}
    assert tracker.tracked_objects.tracker_id.tolist() == [0]
    assert tracker.tracked_objects.data["tracklet_id"].tolist() == [1]
    np.testing.assert_array_equal(model.frames[-1], bgr)
    np.testing.assert_array_equal(tracker._previous_frame, bgr[..., ::-1])


def test_duplicate_and_missing_images_preserve_pending_history_and_birth_clock() -> None:
    """Only new accepted steps advance the window; a missing image loses no rows."""
    model = ConstantEncoder()
    tracker = McBytePlusPlusTracker(
        enable_cmc=False,
        reid_model=model,
        reid_joining_config=LongTermReIDConfig(maximum_reconnection_age=2, minimum_feature_frame_gap=1),
    )
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    tracker.update(sv.Detections.empty(), frame, timestamp=0)
    tracker.update(observation(), frame, timestamp=0.04)  # birth, still unconfirmed
    confirmed = tracker.update(observation(), frame, timestamp=0.08)
    assert confirmed.data["reid_pending"].tolist() == [True]
    assert len(model.calls) == 1
    with pytest.warns(UserWarning, match="duplicate"):
        duplicate = tracker.update(observation(), frame, timestamp=0.08)
    assert duplicate.data["reid_pending"].tolist() == [True]
    assert len(model.calls) == 1
    with pytest.warns(UserWarning, match="earlier"):
        tracker.update(observation(), frame, timestamp=0.07)
    assert tracker.frame_id == 4
    tracker.update(observation(), frame=None, timestamp=0.12)
    assert tracker.pop_finalized_results() == {}
    current = tracker.update(observation(), frame, timestamp=0.16)
    assert current.tracker_id.tolist() == [0]
    assert len(model.calls) == 2
    history = tracker.pop_finalized_results()
    assert list(history) == [3, 4, 5]
    assert all(rows.tracker_id.tolist() == [0] for rows in history.values())


def test_timestamp_removal_survives_mask_failure_and_keeps_processing_references(monkeypatch) -> None:
    """An early expiry remains visible to re-ID after a partially failed update."""
    manager = MaskManager(DummyBoxMaskGenerator(), DummyIdentityMaskPropagator())
    tracker = McBytePlusPlusTracker(
        enable_cmc=False,
        mask_manager=manager,
        enable_mask_refresh=False,
        lost_track_buffer=3,
        reid_model=ConstantEncoder(),
    )
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    tracker.update(observation(), frame, timestamp=0)
    tracker.update(observation(), frame, timestamp=0.04)
    original = tracker._run_mask_manager

    def fail_once(*args):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(tracker, "_run_mask_manager", fail_once)
    with pytest.raises(RuntimeError, match="inference failed"):
        tracker.update(sv.Detections.empty(), frame, timestamp=0.5)
    monkeypatch.setattr(tracker, "_run_mask_manager", original)
    tracker.update(observation(120), frame, timestamp=0.54)
    result = tracker.update(observation(120), frame, timestamp=0.58)
    assert result.tracker_id.tolist() == [0]
    assert result.data["tracklet_id"].tolist() == [1]
    assert tracker.reid_joiner is not None
    assert tracker.reid_joiner.frame_number == tracker.frame_id == 5
    assert tracker._pending_reid_removals == set()


def test_reid_occlusion_uses_discarded_current_detections_without_masks() -> None:
    """Even a low-confidence non-trackable box can block a feature crop."""
    model = ConstantEncoder()
    tracker = McBytePlusPlusTracker(
        enable_cmc=False,
        reid_model=model,
        reid_joining_config=LongTermReIDConfig(enable_feature_occlusion_filter=True),
    )
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    detections = sv.Detections(xyxy=np.array([[20, 20, 60, 110], [20, 21, 60, 120]]), confidence=np.array([0.9, 0.05]))
    result = tracker.update(detections, frame)
    assert result.tracker_id.tolist() == [0]
    assert model.calls == []
    tracker.update(observation(), frame)
    assert len(model.calls) == 1  # no previous-image occluder, no sampling-gap penalty


def test_duplicate_after_failed_mask_update_does_not_extract_features(monkeypatch) -> None:
    """A clock advanced by the failed call does not turn its retry into a sample."""
    model = ConstantEncoder()
    tracker = McBytePlusPlusTracker(
        enable_cmc=False,
        mask_manager=MaskManager(DummyBoxMaskGenerator(), DummyIdentityMaskPropagator()),
        enable_mask_refresh=False,
        reid_model=model,
        reid_joining_config=LongTermReIDConfig(minimum_feature_frame_gap=1),
    )
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    tracker.update(observation(), frame, timestamp=0)
    original = tracker._run_mask_manager

    def fail_once(*args):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(tracker, "_run_mask_manager", fail_once)
    with pytest.raises(RuntimeError, match="inference failed"):
        tracker.update(observation(), frame, timestamp=0.04)
    monkeypatch.setattr(tracker, "_run_mask_manager", original)
    with pytest.warns(UserWarning, match="duplicate"):
        current = tracker.update(observation(), frame, timestamp=0.04)
    assert current.tracker_id.tolist() == [0]
    assert len(model.calls) == 1
    tracker.update(observation(), frame, timestamp=0.08)
    assert len(model.calls) == 2


@pytest.mark.parametrize("mode", ["disabled", "buffered", "unbuffered"])
def test_finish_reset_and_disabled_retrieval(mode) -> None:
    """Closure is safe in every mode; reset retains the encoder, not identities."""
    model = ConstantEncoder()
    tracker = McBytePlusPlusTracker(
        enable_cmc=False,
        reid_model=None if mode == "disabled" else model,
        reid_joining_config=None if mode == "disabled" else LongTermReIDConfig(buffer_results=mode == "buffered"),
    )
    tracker.update(sv.Detections.empty())
    tracker.update(observation())
    pending = tracker.update(observation())
    assert pending.tracker_id.tolist() == ([0] if mode == "disabled" else [-1])
    final = tracker.finish()
    assert list(final) == ([3] if mode == "buffered" else [])
    assert tracker.tracked_objects.tracker_id.tolist() == [0]
    assert tracker.finish() == {}
    with pytest.raises(RuntimeError, match="reset"):
        tracker.update(observation())
    if mode != "buffered":
        with pytest.raises(RuntimeError, match="buffer_results"):
            tracker.pop_finalized_results()
    tracker.reset()
    assert tracker.update(observation()).tracker_id.tolist() == [0]
    if mode != "disabled":
        assert tracker.reid_joiner is not None
        assert tracker.reid_joiner.model is model


def test_joining_config_requires_model_and_disabled_mode_does_not_load_one(monkeypatch) -> None:
    """No accidental dependency loading or silently ignored joining config."""

    def unexpected_load(*args, **kwargs):
        pytest.fail("Disabled re-ID must not load an encoder")

    monkeypatch.setattr("trackers.core.mcbyteplusplus.tracker._load_long_term_encoder", unexpected_load)
    with pytest.raises(ValueError, match="requires reid_model"):
        McBytePlusPlusTracker(reid_joining_config=LongTermReIDConfig())
    result = McBytePlusPlusTracker(enable_cmc=False).update(observation())
    assert result.tracker_id.tolist() == [0]
    assert "reid_pending" not in result.data and "tracklet_id" not in result.data
