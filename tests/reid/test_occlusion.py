# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Box geometry and optional extraction gating independent of mask support."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import supervision as sv

from trackers.reid.long_term import LongTermReID
from trackers.reid.long_term_config import LongTermReIDConfig
from trackers.reid.occlusion import foreground_overlap


class _Encoder:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def extract_embeddings(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        self.calls.append(detections.xyxy.copy())
        return np.tile([1.0, 0.0], (len(detections), 1))


def _observations(internal_id: int) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[10, 10, 60, 70]], dtype=np.float32),
        tracker_id=np.array([internal_id]),
        confidence=np.array([0.9]),
    )


def _frame() -> np.ndarray:
    return np.zeros((140, 140, 3), dtype=np.uint8)


def _occluder() -> np.ndarray:
    # The foreground box covers exactly 60% of the target's area.
    return np.array([[10, 34, 60, 100]], dtype=np.float32)


def test_foreground_overlap_uses_target_area_instead_of_iou() -> None:
    target = np.array([0, 0, 10, 10])
    much_larger_foreground = np.array([[0, 0, 100, 20]])

    assert foreground_overlap(target, much_larger_foreground) == 1.0


def test_foreground_overlap_uses_maximum_individual_box_not_sum_or_union() -> None:
    target = np.array([0, 0, 10, 10])
    references = np.array([[-1, 0, 4, 20], [6, 0, 11, 20], [2, 0, 7, 20]])

    assert foreground_overlap(target, references) == 0.5


def test_only_strictly_lower_bottom_valid_boxes_contribute() -> None:
    target = np.array([0, 0, 10, 10])
    references = np.array(
        [
            [0, 0, 10, 10],  # The target itself or another equal-bottom box.
            [0, 0, 10, 9],
            [0, 0, 0, 20],
            [10, 0, 0, 20],
            [0, 0, np.nan, 20],
            [0, 0, 10, np.inf],
            [20, 0, 30, 20],  # Foreground but no intersection.
            [0, 8, 10, 11],
        ]
    )

    assert foreground_overlap(target, references) == 0.2


def test_empty_references_and_zero_target_area_have_zero_overlap() -> None:
    assert foreground_overlap(np.array([0, 0, 10, 10]), np.empty((0, 4))) == 0
    assert foreground_overlap(np.array([0, 0, 0, 10]), np.array([[0, 0, 10, 20]])) == 0


@pytest.mark.parametrize("enable_filter, samples", [(False, 1), (True, 0)])
def test_filter_defaults_on_and_counts_unassigned_low_confidence_reference_boxes(
    enable_filter: bool, samples: int
) -> None:
    encoder = _Encoder()
    config = (
        LongTermReIDConfig()
        if enable_filter
        else LongTermReIDConfig(enable_feature_occlusion_filter=False, pause_reconnection_on_occlusion=False)
    )
    joiner = LongTermReID(encoder, config)
    observations = sv.Detections(
        xyxy=np.concatenate([_observations(20).xyxy, _occluder()]),
        tracker_id=np.array([20, -1]),
        confidence=np.array([0.9, 0.1]),
    )

    live = joiner.update(observations, _frame())

    assert len(encoder.calls) == samples
    np.testing.assert_array_equal(live.data["reid_pending"], [False, False])
    if samples:
        np.testing.assert_array_equal(encoder.calls[0], observations.xyxy[:1])


@pytest.mark.parametrize("threshold, samples", [(0.6, 0), (0.6001, 1)])
def test_occlusion_threshold_equality_defers_extraction_from_all_current_references(
    threshold: float, samples: int
) -> None:
    encoder = _Encoder()
    joiner = LongTermReID(
        encoder,
        LongTermReIDConfig(enable_feature_occlusion_filter=True, feature_occlusion_overlap_threshold=threshold),
    )

    # The reference need not be confirmed or present in output observations.
    joiner.update(_observations(20), _frame(), reference_boxes=_occluder())

    assert len(encoder.calls) == samples


def test_rejected_occlusion_does_not_set_sample_gap_or_retain_stale_references() -> None:
    encoder = _Encoder()
    joiner = LongTermReID(encoder, LongTermReIDConfig(enable_feature_occlusion_filter=True))
    joiner.update(_observations(20), _frame(), reference_boxes=_occluder())
    assert encoder.calls == []

    joiner.update(_observations(20), _frame())

    assert len(encoder.calls) == 1


@pytest.mark.parametrize("pause_on_occlusion", [False, True])
def test_occluded_current_crop_can_reconnect_using_previously_collected_feature(pause_on_occlusion: bool) -> None:
    encoder = _Encoder()
    joiner = LongTermReID(
        encoder,
        LongTermReIDConfig(
            enable_feature_occlusion_filter=True,
            pause_reconnection_on_occlusion=pause_on_occlusion,
            minimum_feature_frame_gap=1,
        ),
    )
    old = joiner.update(_observations(100), _frame())
    pending = joiner.update(_observations(200), _frame())
    assert pending.tracker_id[0] == -1

    live = joiner.update(_observations(200), _frame(), reference_boxes=_occluder(), removed_tracklet_ids=[100])

    assert live.tracker_id[0] == old.tracker_id[0]
    assert len(encoder.calls) == 2
    history = joiner.pop_finalized_results()
    assert set(history) == {2}
    np.testing.assert_array_equal(history[2].tracker_id, old.tracker_id)


@pytest.mark.parametrize("pause_on_occlusion", [False, True])
def test_occlusion_retry_preserves_samples_and_respects_optional_deadline_pause(pause_on_occlusion: bool) -> None:
    encoder = _Encoder()
    joiner = LongTermReID(
        encoder,
        LongTermReIDConfig(
            enable_feature_occlusion_filter=True,
            pause_reconnection_on_occlusion=pause_on_occlusion,
            minimum_feature_samples=2,
            minimum_feature_frame_gap=2,
            maximum_reconnection_age=2,
        ),
    )
    old = joiner.update(_observations(100), _frame())
    joiner.update(_observations(100))
    joiner.update(_observations(100), _frame())
    pending = joiner.update(_observations(200), _frame(), removed_tracklet_ids=[100])
    assert pending.tracker_id[0] == -1
    assert len(encoder.calls) == 3

    # The second occluded update reaches the sampling gap and decision-age
    # boundary. Rejecting it must not supply the missing second sample.
    for _ in range(2):
        pending = joiner.update(_observations(200), _frame(), reference_boxes=_occluder())
        assert pending.tracker_id[0] == -1
    assert len(encoder.calls) == 3

    # Retry immediately: the last successful sample was three steps ago.
    # Only the optional pause preserves the opportunity to join the old ID.
    live = joiner.update(_observations(200), _frame())

    assert len(encoder.calls) == 4
    assert live.tracker_id[0] >= 0
    assert (live.tracker_id[0] == old.tracker_id[0]) == pause_on_occlusion
    history = joiner.pop_finalized_results()
    assert set(history) == {4, 5, 6}
    assert all(row.tracker_id[0] == live.tracker_id[0] for row in history.values())


def test_occlusion_pauses_only_affected_track_between_scheduled_samples() -> None:
    encoder = _Encoder()
    joiner = LongTermReID(
        encoder,
        LongTermReIDConfig(
            enable_feature_occlusion_filter=True,
            pause_reconnection_on_occlusion=True,
            maximum_reconnection_age=1,
        ),
    )
    observations = sv.Detections(
        xyxy=np.array([[10, 10, 60, 70], [80, 10, 130, 70]], dtype=np.float32),
        tracker_id=np.array([20, 30]),
        confidence=np.array([0.9, 0.9]),
    )
    joiner.update(sv.Detections.empty())
    joiner.update(observations, _frame())  # one sample per track at step 2
    for _ in range(4):
        current = joiner.update(observations, _frame(), reference_boxes=_occluder())
    assert current.tracker_id.tolist() == [-1, 0]
    assert joiner._tracks[20].paused_steps == 4  # including steps before sampling is due
    assert joiner._tracks[30].paused_steps == 0
    assert joiner._tracks[20].samples == 1
    assert joiner._tracks[20].last_feature_step == 2
    history = joiner.pop_finalized_results()
    assert set(history) == {2, 3}
    assert all(rows.data["tracklet_id"].tolist() == [30] for rows in history.values())
    assert joiner.update(observations, _frame()).tracker_id.tolist() == [-1, 0]
    assert joiner._tracks[20].last_feature_step == 7  # immediate retry on a clear crop
    assert joiner.update(observations, _frame()).tracker_id.tolist() == [1, 0]
    assert set(joiner.pop_finalized_results()) == {2, 3, 4, 5, 6, 7}


def test_occlusion_and_host_pause_share_one_step_and_one_cap() -> None:
    encoder = _Encoder()
    joiner = LongTermReID(
        encoder,
        LongTermReIDConfig(
            enable_feature_occlusion_filter=True,
            pause_reconnection_on_occlusion=True,
            maximum_reconnection_age=1,
            maximum_reconnection_pause_steps=2,
        ),
    )
    joiner.update(sv.Detections.empty())
    for step in range(2, 7):
        current = joiner.update(_observations(20), _frame(), reference_boxes=_occluder(), pause_reconnection=step != 4)
        assert current.tracker_id.tolist() == ([0] if step == 6 else [-1])
        assert joiner._tracks[20].paused_steps == min(step - 2, 2)
    assert encoder.calls == []
    assert set(joiner.pop_finalized_results()) == {2, 3, 4, 5}


def test_occlusion_does_not_pause_duplicate_missing_or_unobserved_updates() -> None:
    joiner = LongTermReID(
        _Encoder(),
        LongTermReIDConfig(
            enable_feature_occlusion_filter=True,
            pause_reconnection_on_occlusion=True,
            maximum_reconnection_age=100,
        ),
    )
    joiner.update(sv.Detections.empty())
    joiner.update(_observations(20), _frame(), reference_boxes=_occluder(), step=2)
    joiner.update(_observations(20), _frame(), reference_boxes=_occluder(), step=3)
    joiner.update(_observations(20), _frame(), reference_boxes=_occluder(), step=3)
    joiner.update(_observations(20), None, reference_boxes=_occluder(), step=4)
    joiner.update(sv.Detections.empty(), _frame(), reference_boxes=_occluder(), step=5)
    joiner.update(_observations(20), _frame(), step=6)
    assert joiner._tracks[20].paused_steps == 1
    assert joiner._tracks[20].samples == 1
    assert joiner._tracks[20].last_feature_step == 6


@pytest.mark.parametrize("buffer_results", [False, True])
def test_occlusion_without_any_features_still_finalizes_on_termination_and_finish(buffer_results: bool) -> None:
    encoder = _Encoder()
    joiner = LongTermReID(
        encoder,
        LongTermReIDConfig(
            enable_feature_occlusion_filter=True,
            pause_reconnection_on_occlusion=True,
            maximum_reconnection_age=0,
            buffer_results=buffer_results,
        ),
    )
    joiner.update(sv.Detections.empty())
    for _ in range(3):
        assert joiner.update(_observations(20), _frame(), reference_boxes=_occluder()).tracker_id.tolist() == [-1]
    joiner.update(sv.Detections.empty(), removed_tracklet_ids=[20])
    history = joiner.pop_finalized_results() if buffer_results else {}
    assert list(history) == ([2, 3, 4] if buffer_results else [])
    assert all(rows.tracker_id.tolist() == [0] for rows in history.values())
    for _ in range(3):
        assert joiner.update(_observations(30), _frame(), reference_boxes=_occluder()).tracker_id.tolist() == [-1]
    history = joiner.finish()
    assert list(history) == ([6, 7, 8] if buffer_results else [])
    assert joiner.public_ids[30] == 1
    assert encoder.calls == []


def test_occlusion_configuration_requires_positive_threshold_and_compatible_filter() -> None:
    with pytest.raises(ValueError, match="feature_occlusion_overlap_threshold"):
        LongTermReIDConfig(feature_occlusion_overlap_threshold=0)
    # Zero confidence and full-overlap occlusion thresholds remain supported.
    LongTermReIDConfig(minimum_detection_confidence=0, feature_occlusion_overlap_threshold=1)
    with pytest.raises(ValueError, match="requires enable_feature_occlusion_filter"):
        LongTermReIDConfig(enable_feature_occlusion_filter=False)
    with pytest.raises(ValueError, match="pause_reconnection_on_occlusion must be a boolean"):
        invalid: Any = 1
        LongTermReIDConfig(enable_feature_occlusion_filter=True, pause_reconnection_on_occlusion=invalid)
