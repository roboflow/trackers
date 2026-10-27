# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""McByte++ API, color, identity, timing and lifecycle regressions."""

import numpy as np
import pytest
import supervision as sv

from trackers import McBytePlusPlusMaskConfig, McBytePlusPlusTracker
from trackers.core.base import BaseTracker
from trackers.core.masks.base import MaskOutput, TrackletSnapshot
from trackers.core.masks.dummy import DummyBoxMaskGenerator, DummyIdentityMaskPropagator
from trackers.core.masks.manager import MaskManager
from trackers.core.mcbyteplusplus.tracklet import McBytePlusPlusTracklet
from trackers.utils.iou import DIoU, GIoU, IoU


class RecordingGenerator(DummyBoxMaskGenerator):
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, list[TrackletSnapshot]]] = []

    def generate(self, frame: np.ndarray, tracklets: list[TrackletSnapshot]) -> MaskOutput:
        self.calls.append((frame.copy(), tracklets))
        return super().generate(frame, tracklets)


def detection(box=(2.0, 2.0, 8.0, 12.0), confidence=1.0) -> sv.Detections:
    return sv.Detections(xyxy=np.array([box]), confidence=np.array([confidence]))


@pytest.fixture
def masked_tracker() -> tuple[McBytePlusPlusTracker, RecordingGenerator]:
    generator = RecordingGenerator()
    manager = MaskManager(generator, DummyIdentityMaskPropagator(), mask_creation_bbox_overlap_threshold=0.1)
    return McBytePlusPlusTracker(enable_cmc=False, mask_manager=manager, enable_mask_refresh=False), generator


def test_independent_registered_tracker_and_reset() -> None:
    tracker = McBytePlusPlusTracker(enable_cmc=False)
    assert McBytePlusPlusTracker.__bases__ == (BaseTracker,)
    assert tracker.update(detection()).tracker_id.tolist() == [0]
    assert tracker.update(detection()).tracker_id.tolist() == [0]
    tracker.reset()
    assert tracker.tracks == [] and tracker.frame_id == 0
    assert tracker.update(detection()).tracker_id.tolist() == [0]


@pytest.mark.parametrize("iou_class", [IoU, GIoU, DIoU])
@pytest.mark.parametrize("use_timestamps", [False, True])
def test_confirmation_low_scores_and_expiry_have_explicit_ids(iou_class, use_timestamps: bool) -> None:
    """Only confirmed tracks get public IDs; expired IDs are never reassigned."""
    # GIoU/DIoU are normalized before score fusion. Use a threshold that keeps
    # these two disjoint boxes ineligible for matching for every IoU variant.
    tracker = McBytePlusPlusTracker(
        enable_cmc=False, lost_track_buffer=2, iou=iou_class(), minimum_iou_threshold_first_assoc=0.5
    )
    frames = [
        detection(),
        detection(confidence=0.3),
        sv.Detections.empty(),
        detection(),
        detection(box=(20.0, 2.0, 26.0, 12.0)),
        detection(box=(20.0, 2.0, 26.0, 12.0)),
        sv.Detections.empty(),
        sv.Detections.empty(),
        sv.Detections.empty(),
        detection(),
    ]
    expected_ids = [[0], [0], [], [0], [-1], [1], [], [], [], [-1]]
    # The long time gap expires both tracks immediately in timestamp mode.
    # In fixed-rate mode ID 1 survives two misses, then expires on the third.
    expected_alive = [[0], [0], [0], [0], [0], [0, 1], [], [], [], []]
    if not use_timestamps:
        expected_alive[6:8] = [[1], [1]]
    times = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 1.0, 1.03, 1.06, 1.09]
    boxes = np.array([[2.0, 2.0, 8.0, 12.0], [20.0, 2.0, 26.0, 12.0]])
    for index, detections in enumerate(frames):
        actual = tracker.update(detections, timestamp=times[index] if use_timestamps else None)
        np.testing.assert_array_equal(actual.xyxy, detections.xyxy)
        assert actual.tracker_id.tolist() == expected_ids[index], f"Unexpected IDs at step {index + 1}"
        assert tracker.tracked_objects.tracker_id.tolist() == expected_alive[index]
        np.testing.assert_allclose(tracker.tracked_objects.xyxy, boxes[expected_alive[index]])


def test_bgr_input_is_converted_once_and_mask_ids_ignore_public_identity(masked_tracker) -> None:
    """Keep the accepted mask image and internal owner even if the caller reuses its buffer."""
    tracker, generator = masked_tracker
    rgb = np.full((20, 30, 3), [33, 22, 11], dtype=np.uint8)
    bgr = rgb[..., ::-1]
    tracker.update(detection(), bgr)
    # Reversing this view back to RGB can share caller memory. A queued mask
    # prompt must still use the accepted image after that buffer is overwritten.
    bgr[:] = [44, 55, 66]
    tracker.tracks[0].public_id = 900
    result = tracker.update(detection(), bgr)
    assert result.tracker_id.tolist() == [900]
    assert tracker.tracked_objects.tracker_id.tolist() == [900]
    assert tracker.last_mask_output.tracklet_mask_dict == {0: 0}
    assert generator.calls[0][1][0].tracker_id == 0
    np.testing.assert_array_equal(generator.calls[0][0][0, 0], [33, 22, 11])


def test_cmc_receives_bgr_and_skips_duplicate_timestamps(monkeypatch) -> None:
    tracker = McBytePlusPlusTracker()
    bgr = np.full((20, 30, 3), [11, 22, 33], dtype=np.uint8)
    observed = []

    def estimate(frame, boxes):
        observed.append(frame.copy())
        return None

    monkeypatch.setattr(tracker.cmc, "estimate", estimate)
    tracker.update(detection(), bgr, timestamp=0.0)
    with pytest.warns(UserWarning, match="duplicate"):
        tracker.update(detection(), bgr + 1, timestamp=0.0)
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], bgr)


@pytest.mark.parametrize("minimum_mask_creation_frames", [1, 3])
@pytest.mark.parametrize("duplicates_per_frame", [0, 2])
def test_duplicates_preserve_mask_creation_timing_and_original_prompts(
    masked_tracker, minimum_mask_creation_frames: int, duplicates_per_frame: int
) -> None:
    """Repeated timestamps neither starve creation nor count as new image matches."""
    tracker, generator = masked_tracker
    tracker.minimum_mask_creation_frames = minimum_mask_creation_frames
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    for index in range(minimum_mask_creation_frames + 2):
        timestamp = index * 0.04
        tracker.update(detection(), frame + index, timestamp=timestamp)
        created = index >= minimum_mask_creation_frames
        assert len(generator.calls) == int(created)
        if created:
            assert tracker.last_mask_output.tracklet_mask_dict == {0: 0}
        ages = tracker._mask_pending_ages.copy()
        assert ages == ({} if created else {0: index + 1})
        for _ in range(duplicates_per_frame):
            with pytest.warns(UserWarning, match="duplicate"):
                tracker.update(detection(box=(3.0, 2.0, 9.0, 12.0)), frame + 100, timestamp=timestamp)
            assert tracker.last_mask_output is None
            assert tracker._mask_pending_ages == ages
            assert len(generator.calls) == int(created)
    np.testing.assert_array_equal(generator.calls[0][0], frame + minimum_mask_creation_frames - 1)
    np.testing.assert_array_equal(generator.calls[0][1][0].xyxy, detection().xyxy[0])


@pytest.mark.parametrize("timestamp", [0.0, 0.04])
def test_missing_image_discards_queued_box_even_at_duplicate_timestamp(masked_tracker, timestamp: float) -> None:
    tracker, generator = masked_tracker
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    tracker.update(detection(), frame, timestamp=0.0)
    if timestamp == 0.0:
        with pytest.warns(UserWarning, match="duplicate"):
            tracker.update(detection(), None, timestamp=timestamp)
    else:
        tracker.update(detection(), None, timestamp=timestamp)
    assert tracker._mask_pending_ages == {}
    tracker.update(detection(), frame + 2, timestamp=0.08)
    assert generator.calls == []
    tracker.update(detection(), frame + 3, timestamp=0.12)
    assert len(generator.calls) == 1
    np.testing.assert_array_equal(generator.calls[0][0], frame + 2)


def test_missing_image_keeps_existing_mask_memory(masked_tracker) -> None:
    tracker, generator = masked_tracker
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    tracker.update(detection(), frame)
    tracker.update(detection(), frame)
    tracker.update(detection(), None)
    assert tracker.last_mask_output is None
    tracker.update(detection(), frame)
    output = tracker.last_mask_output
    assert output is not None and output.tracklet_mask_dict == {0: 0}
    assert len(generator.calls) == 1


@pytest.mark.parametrize("mask_created", [False, True])
def test_timestamp_expiry_removes_mask_before_reassociation(masked_tracker, mask_created: bool) -> None:
    """Preserved duplicate-call prompts must not resurrect a subsequently expired ID."""
    tracker, generator = masked_tracker
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    tracker.update(detection(), frame, timestamp=0.0)
    if mask_created:
        tracker.update(detection(), frame, timestamp=0.04)
    with pytest.warns(UserWarning, match="duplicate"):
        tracker.update(detection(), frame + 1, timestamp=0.04 if mask_created else 0.0)
    tracker.update(detection(), frame, timestamp=5.0)
    assert all(t.tracker_id != 0 for t in tracker.tracks)
    assert tracker.last_mask_output is None or 0 not in tracker.last_mask_output.tracklet_mask_dict
    assert len(generator.calls) == int(mask_created)


def test_death_during_image_gap_is_removed_on_next_image(masked_tracker) -> None:
    tracker, _ = masked_tracker
    tracker.maximum_frames_without_update = 0
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    tracker.update(detection(), frame)
    tracker.update(detection(), frame)
    tracker.update(sv.Detections.empty(), None)
    assert tracker.tracks == []
    tracker.update(sv.Detections.empty(), frame)
    assert tracker.last_mask_output is None or tracker.last_mask_output.tracklet_mask_dict == {}


@pytest.mark.parametrize("timestamp", [-1.0, float("nan")])
def test_invalid_or_backward_timestamp_leaves_observation_unchanged(masked_tracker, timestamp: float) -> None:
    tracker, generator = masked_tracker
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    tracker.update(detection(), frame, timestamp=0.0)
    with pytest.warns(UserWarning):
        assert tracker.update(detection(), frame + 1, timestamp=timestamp).tracker_id.tolist() == [-1]
    assert tracker.frame_id == 1
    tracker.update(detection(), frame + 2, timestamp=0.04)
    np.testing.assert_array_equal(generator.calls[0][0], frame)


@pytest.mark.parametrize("frame", [np.zeros((4, 4, 3), dtype=float), np.zeros((4, 4), dtype=np.uint8)])
def test_bad_frame_is_rejected_before_state_mutation(frame) -> None:
    tracker = McBytePlusPlusTracker(enable_cmc=False)
    with pytest.raises(ValueError, match="BGR uint8"):
        tracker.update(detection(), frame, timestamp=1.0)
    assert tracker.frame_id == 0 and tracker._last_timestamp is None and tracker.tracks == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mask_refresh_coverage_threshold": float("nan")},
        {"mask_refresh_interval": 0},
        {"minimum_mask_creation_frames": 0},
        {"high_conf_det_threshold": 0.1},
        {"edgetam_config": McBytePlusPlusMaskConfig()},
    ],
)
def test_invalid_configuration_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        McBytePlusPlusTracker(enable_cmc=False, **kwargs)


@pytest.mark.parametrize(
    "config,expected_options,expected_overlap",
    [
        (
            None,
            {
                "model_id": "yonigozlan/EdgeTAM-hf",
                "revision": None,
                "device": "auto",
                "max_session_frames": 200,
                "mask_threshold": 0.0,
            },
            0.1,
        ),
        (
            McBytePlusPlusMaskConfig(
                model_id="example/custom-edgetam",
                revision="test-revision",
                device="cuda:1",
                max_session_frames=None,
                mask_threshold=0.4,
                mask_creation_bbox_overlap_threshold=0.7,
            ),
            {
                "model_id": "example/custom-edgetam",
                "revision": "test-revision",
                "device": "cuda:1",
                "max_session_frames": None,
                "mask_threshold": 0.4,
            },
            0.7,
        ),
    ],
)
def test_backend_is_shared_and_receives_all_config_options(
    monkeypatch, config, expected_options, expected_overlap
) -> None:
    """One backend receives all model options; the manager receives the overlap rule."""
    from trackers.core.masks import edgetam

    class FakeBackend(DummyBoxMaskGenerator, DummyIdentityMaskPropagator):
        def __init__(self, **kwargs):
            DummyIdentityMaskPropagator.__init__(self)
            self.options = kwargs

        def refresh_masks(self, frame, masks):
            pass

    monkeypatch.setattr(edgetam, "EdgeTAMMaskBackend", FakeBackend)
    tracker = McBytePlusPlusTracker(enable_cmc=False, enable_mask_manager=True, edgetam_config=config)
    manager = tracker.mask_manager
    assert manager is not None and isinstance(manager.mask_generator, FakeBackend)
    assert manager.mask_generator is manager.mask_propagator
    assert manager.mask_generator.options == expected_options
    assert manager.mask_creation_bbox_overlap_threshold == expected_overlap


def test_association_stage_order_preserves_detection_metadata() -> None:
    """Return high matches, low matches, then tentative or newly confirmed tracks."""
    tracker = McBytePlusPlusTracker(enable_cmc=False)
    boxes = np.array([[0.0, 0.0, 10.0, 20.0], [30.0, 0.0, 40.0, 20.0], [60.0, 0.0, 70.0, 20.0]])
    steps = [
        ([0, 1], [1.0, 1.0], [0, 1], [0, 1]),
        ([0, 2, 1], [0.3, 1.0, 1.0], [1, 0, 2], [1, 0, -1]),
        ([2, 1], [1.0, 1.0], [1, 2], [1, 2]),
    ]
    for indices, scores, expected_rows, expected_ids in steps:
        detections = sv.Detections(
            xyxy=boxes[indices],
            confidence=np.array(scores),
            class_id=np.array(indices),
            data={"source_row": np.array(indices)},
        )
        actual = tracker.update(detections)
        np.testing.assert_array_equal(actual.xyxy, boxes[expected_rows])
        assert actual.tracker_id.tolist() == expected_ids
        assert actual.class_id.tolist() == expected_rows
        assert actual.data["source_row"].tolist() == expected_rows
        assert detections.tracker_id is None
        np.testing.assert_array_equal(detections.xyxy, boxes[indices])
        np.testing.assert_array_equal(detections.confidence, scores)
        np.testing.assert_array_equal(detections.data["source_row"], indices)


def test_duplicate_low_confidence_detection_does_not_enter_second_stage() -> None:
    """Without prediction, a just-matched track has age zero rather than stage two's required age one."""
    tracker = McBytePlusPlusTracker(enable_cmc=False)
    tracker.update(detection(), timestamp=0.0)
    with pytest.warns(UserWarning, match="duplicate"):
        result = tracker.update(detection(confidence=0.3), timestamp=0.0)
    assert result.tracker_id.tolist() == [-1]
    assert tracker.tracks[0].tracker_id == 0
    assert tracker.tracks[0].number_of_successful_updates == 1
    assert tracker.tracks[0].time_since_update == 0


def test_unassigned_mature_track_survives_miss_and_gets_id_on_match() -> None:
    """Maturity by observation count remains sticky before an ID is assigned."""
    tracker = McBytePlusPlusTracker(
        enable_cmc=False, minimum_consecutive_frames=1, instant_first_frame_activation=False
    )
    assert tracker.update(detection()).tracker_id.tolist() == [-1]
    original = tracker.tracks[0]
    tracker.update(sv.Detections.empty())
    assert tracker.tracks == [original]
    assert tracker.update(detection()).tracker_id.tolist() == [0]
    assert tracker.tracks == [original]
    assert original.number_of_successful_updates == 2


def test_oom_streak_survives_deferred_creation_and_reset_restores_manager(caplog) -> None:
    """No-op calls between failed inference attempts must not prevent auto-disable."""

    class OOMGenerator(DummyBoxMaskGenerator):
        fail = True
        attempts = 0

        def generate(self, frame, tracklets):
            self.attempts += 1
            if self.fail:
                raise RuntimeError("CUDA out of memory")
            return super().generate(frame, tracklets)

    generator = OOMGenerator()
    manager = MaskManager(generator, DummyIdentityMaskPropagator())
    tracker = McBytePlusPlusTracker(
        enable_cmc=False, mask_manager=manager, enable_mask_refresh=False, minimum_mask_creation_frames=2
    )
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    for _ in range(7):
        assert tracker.update(detection(), frame).tracker_id.tolist() == [0]
    assert generator.attempts == 3
    assert tracker.mask_manager is None and tracker.last_mask_output is None
    assert "disabling" in caplog.text
    tracker.reset()
    assert tracker.mask_manager is manager
    generator.fail = False
    for _ in range(3):
        tracker.update(detection(), frame)
    assert tracker.last_mask_output.tracklet_mask_dict == {0: 0}


def test_tracker_cmc_helper_preserves_internal_and_public_ids() -> None:
    tracker = McBytePlusPlusTracker(enable_cmc=False)
    tracker.update(detection())
    tracker.tracks[0].public_id = 900
    before = tracker.tracked_objects.xyxy.copy()
    tracker.apply_cmc_batch(np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]]))
    tracker.apply_cmc_batch(None)
    np.testing.assert_allclose(tracker.tracked_objects.xyxy, before + np.array([5.0, -3.0, 5.0, -3.0]))
    assert tracker.tracks[0].tracker_id == 0
    assert tracker.tracked_objects.tracker_id.tolist() == [900]


@pytest.mark.parametrize("enable_mask_manager", [False, True])
def test_custom_manager_does_not_construct_default_backend(monkeypatch, enable_mask_manager: bool) -> None:
    from trackers.core.masks import edgetam

    def unexpected_backend(**kwargs):
        pytest.fail("A supplied manager must avoid default EdgeTAM construction.")

    monkeypatch.setattr(edgetam, "EdgeTAMMaskBackend", unexpected_backend)
    manager = MaskManager(DummyBoxMaskGenerator(), DummyIdentityMaskPropagator())
    tracker = McBytePlusPlusTracker(
        enable_cmc=False, enable_mask_manager=enable_mask_manager, mask_manager=manager, enable_mask_refresh=False
    )
    assert tracker.mask_manager is manager


def test_config_cannot_be_combined_with_custom_manager(masked_tracker) -> None:
    tracker, _ = masked_tracker
    with pytest.raises(ValueError, match="without a custom mask_manager"):
        McBytePlusPlusTracker(
            enable_mask_manager=True, mask_manager=tracker.mask_manager, edgetam_config=McBytePlusPlusMaskConfig()
        )


@pytest.mark.parametrize("overlap", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_overlap_is_rejected_before_model_loading(monkeypatch, overlap: float) -> None:
    from trackers.core.masks import edgetam

    def unexpected_backend(**kwargs):
        pytest.fail("Invalid overlap must be rejected before loading model weights.")

    monkeypatch.setattr(edgetam, "EdgeTAMMaskBackend", unexpected_backend)
    with pytest.raises(ValueError, match="mask_creation_bbox_overlap_threshold"):
        McBytePlusPlusTracker(
            enable_mask_manager=True,
            edgetam_config=McBytePlusPlusMaskConfig(mask_creation_bbox_overlap_threshold=overlap),
        )


@pytest.mark.parametrize(
    "scores,expected_matches,expected_unmatched",
    [
        ([[0.9, 0.1, 0.0], [0.1, 0.7, 0.6], [0.0, 0.6, 0.7]], [(0, 0), (1, 1), (2, 2)], []),
        ([[0.9, 0.1, 0.0], [0.1, 0.4, 0.3], [0.0, 0.2, 0.1]], [(0, 0)], [1, 2]),
    ],
)
def test_association_maps_reduced_matches_and_unmatched_indices(scores, expected_matches, expected_unmatched) -> None:
    tracker = McBytePlusPlusTracker(enable_cmc=False)
    boxes = np.array([[0.0, 0.0, 5.0, 5.0], [5.0, 5.0, 10.0, 10.0], [10.0, 10.0, 15.0, 15.0]])
    tracks = [McBytePlusPlusTracklet(box) for box in boxes]
    for track, tracker_id in zip(tracks, [10, 20, 30]):
        track.tracker_id = tracker_id
    similarity = np.array(scores)
    matched, unmatched_tracks, unmatched_detections = tracker._get_mask_conditioned_associated_indices(
        similarity_matrix=similarity,
        raw_iou_similarity=similarity,
        tracklets=tracks,
        detection_boxes=boxes,
        min_similarity_thresh=0.5,
    )
    assert matched == expected_matches
    assert unmatched_tracks == expected_unmatched
    assert unmatched_detections == expected_unmatched


def test_mask_creation_age_restarts_after_a_missed_detection(masked_tracker) -> None:
    tracker, generator = masked_tracker
    tracker.minimum_mask_creation_frames = 3
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    tracker.update(detection(), frame)
    tracker.update(detection(), frame + 1)
    assert tracker._mask_pending_ages == {0: 2}
    tracker.update(sv.Detections.empty(), frame + 2)
    assert tracker._mask_pending_ages == {}
    for index in range(3, 6):
        tracker.update(detection(), frame + index)
        assert generator.calls == []
    tracker.update(detection(), frame + 6)
    assert len(generator.calls) == 1
    np.testing.assert_array_equal(generator.calls[0][0], frame + 5)
    assert tracker.last_mask_output.tracklet_mask_dict == {0: 0}


def test_disabled_mask_manager_does_not_retain_images() -> None:
    tracker = McBytePlusPlusTracker(enable_cmc=False, enable_mask_manager=False)
    tracker.update(detection(), np.zeros((20, 30, 3), dtype=np.uint8))
    assert tracker._previous_frame is None
    assert tracker.last_mask_output is None


def test_rematch_restarts_elapsed_miss_clock() -> None:
    tracker = McBytePlusPlusTracker(enable_cmc=False)
    tracker.update(detection(), timestamp=0.0)
    tracker.update(sv.Detections.empty(), timestamp=0.2)
    assert tracker.tracks[0].time_since_update_seconds == pytest.approx(0.2)
    tracker.update(detection(), timestamp=0.3)
    assert tracker.tracks[0].time_since_update_seconds == 0.0
    tracker.update(sv.Detections.empty(), timestamp=0.4)
    assert tracker.tracks[0].time_since_update_seconds == pytest.approx(0.1)


def test_high_confidence_below_activation_is_returned_without_spawning() -> None:
    tracker = McBytePlusPlusTracker(enable_cmc=False, high_conf_det_threshold=0.6, track_activation_threshold=0.7)
    result = tracker.update(detection(confidence=0.65))
    assert result.tracker_id.tolist() == [-1]
    assert tracker.tracks == []


def test_cmc_default_downscale_and_explicit_override() -> None:
    default = McBytePlusPlusTracker()
    custom = McBytePlusPlusTracker(cmc_downscale=2)
    assert default.cmc is not None and default.cmc.downscale == 4
    assert custom.cmc is not None and custom.cmc.downscale == 2


def test_real_sparse_optical_flow_cmc_keeps_a_stationary_track() -> None:
    tracker = McBytePlusPlusTracker(enable_cmc=True, cmc_method="sparseOptFlow")
    frame = np.random.default_rng(42).integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    for _ in range(5):
        result = tracker.update(detection(box=(100.0, 80.0, 150.0, 180.0)), frame)
        assert result.tracker_id.tolist() == [0]
        assert len(tracker.tracks) == 1


def test_departed_tracks_do_not_accumulate_in_timestamp_mode() -> None:
    tracker = McBytePlusPlusTracker(enable_cmc=False)
    frame_index = 0
    for object_index in range(12):
        x = 40.0 + (object_index % 4) * 200.0
        y = 40.0 + (object_index // 4) * 200.0
        for _ in range(8):
            tracker.update(detection((x, y, x + 60.0, y + 80.0)), timestamp=frame_index / 30.0)
            frame_index += 1
        for _ in range(4):
            tracker.update(sv.Detections.empty(), timestamp=frame_index / 30.0)
            frame_index += 1
        assert len(tracker.tracks) <= 4, "Expired tracks must leave the live track list."
