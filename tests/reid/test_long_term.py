# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Synthetic contracts for identity joining, extraction and historical delivery."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
import supervision as sv

from trackers.reid.long_term import LongTermReID
from trackers.reid.long_term_config import LongTermReIDConfig


class _Encoder:
    """Return controlled descriptors keyed by each box's left coordinate."""

    def __init__(self) -> None:
        self.default = (1.0, 0.0)
        self.features: dict[float, tuple[float, ...]] = {}
        self.calls: list[np.ndarray] = []

    def extract_embeddings(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        self.calls.append(detections.xyxy.copy())
        return np.array([self.features.get(float(box[0]), self.default) for box in detections.xyxy], dtype=np.float32)


def _frame() -> np.ndarray:
    return np.zeros((240, 400, 3), dtype=np.uint8)


def _observations(
    internal_ids: list[int],
    boxes: Sequence[tuple[float, float, float, float]] | None = None,
    confidence: list[float] | None = None,
) -> sv.Detections:
    if boxes is None:
        boxes = [(10 + 60 * index, 10, 40 + 60 * index, 70) for index in range(len(internal_ids))]
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
        tracker_id=np.array(internal_ids, dtype=int),
        confidence=np.array(confidence if confidence is not None else [0.9] * len(internal_ids)),
        class_id=np.zeros(len(internal_ids), dtype=int),
    )


def test_paused_candidate_can_join_using_existing_features_and_release_history() -> None:
    """Host-requested pauses preserve the joining window while allowing joins using existing descriptors."""
    encoder = _Encoder()
    joiner = LongTermReID(encoder, LongTermReIDConfig(maximum_reconnection_age=1))
    old = joiner.update(_observations([100]), _frame()).tracker_id[0]
    joiner.update(_observations([200]), _frame())
    for _ in range(4):
        current = joiner.update(_observations([200]), _frame(), extract_features=False, pause_reconnection=True)
        assert current.tracker_id.tolist() == [-1]
        assert joiner.pop_finalized_results() == {}
    # The old owner terminates on another flagged frame. Its descriptor and
    # the candidate's earlier descriptor suffice; no new crop is sampled.
    current = joiner.update(
        _observations([200]),
        _frame(),
        removed_tracklet_ids=[100],
        extract_features=False,
        pause_reconnection=True,
    )
    assert current.tracker_id.tolist() == [old]
    assert len(encoder.calls) == 2
    assert joiner._tracks[200].samples == 1
    history = joiner.pop_finalized_results()
    assert list(history) == [2, 3, 4, 5, 6]
    assert all(row.tracker_id.tolist() == [old] for row in history.values())
    assert joiner.pop_finalized_results() == joiner.finish() == {}


@pytest.mark.parametrize("cap,final_step", [(0, 4), (2, 6), (None, None)])
def test_pause_cap_limits_extra_wait_without_forcing_a_join(cap, final_step) -> None:
    joiner = LongTermReID(
        _Encoder(),
        LongTermReIDConfig(
            maximum_reconnection_age=1,
            maximum_reconnection_pause_steps=cap,
        ),
    )
    joiner.update(_observations([]))
    joiner.update(_observations([10]))  # birth at step 2, age zero
    positive_steps = []
    for step in range(3, 9):
        result = joiner.update(_observations([10]), pause_reconnection=True, extract_features=False)
        if result.tracker_id[0] >= 0:
            positive_steps.append(step)
    assert (positive_steps[0] if positive_steps else None) == final_step
    assert joiner._tracks[10].samples == 0
    tail = joiner.finish()
    assert tail and all(row.tracker_id.tolist() == [0] for row in tail.values())
    assert joiner.finish() == {}


def test_pause_bursts_are_cumulative_and_duplicate_calls_add_no_pause() -> None:
    joiner = LongTermReID(
        _Encoder(),
        LongTermReIDConfig(
            maximum_reconnection_age=2,
            maximum_reconnection_pause_steps=2,
        ),
    )
    joiner.update(_observations([]), step=1)
    joiner.update(_observations([10]), step=2)
    joiner.update(_observations([10]), step=3, pause_reconnection=True)
    joiner.update(_observations([10]), step=3, pause_reconnection=True)
    joiner.update(_observations([10]), step=4)
    joiner.update(_observations([10]), step=5, pause_reconnection=True)
    assert joiner.update(_observations([10]), step=6, pause_reconnection=True).tracker_id.tolist() == [-1]
    assert joiner.update(_observations([10]), step=7).tracker_id.tolist() == [0]
    assert list(joiner.pop_finalized_results()) == [2, 3, 4, 5, 6, 7]


def test_delayed_registration_seeds_birth_pauses_once_including_current_step() -> None:
    joiner = LongTermReID(_Encoder(), LongTermReIDConfig(maximum_reconnection_age=1))
    for _ in range(5):
        joiner.update(_observations([]))
    # Birth at step 2; four pauses after birth include the registration step 6.
    current = joiner.update(
        _observations([10]), tracklet_starts={10: (2, 2)}, tracklet_paused_steps={10: 4}, pause_reconnection=True
    )
    assert current.tracker_id.tolist() == [-1]
    assert joiner._tracks[10].paused_steps == 4
    assert joiner.update(_observations([10]), tracklet_paused_steps={10: 999}).tracker_id.tolist() == [-1]
    # A seed belongs to registration only; it cannot extend a known track again.
    assert joiner.update(_observations([10])).tracker_id.tolist() == [0]


def test_paused_missing_track_expires_normally_and_does_not_reopen_its_window() -> None:
    joiner = LongTermReID(_Encoder(), LongTermReIDConfig(maximum_reconnection_age=0))
    joiner.update(_observations([]))
    joiner.update(_observations([10]), _frame())
    joiner.update(_observations([]))  # now too old, but temporarily lost
    assert joiner.update(_observations([10]), pause_reconnection=True).tracker_id.tolist() == [0]
    joiner.update(_observations([20]))
    # True termination must resolve even without descriptors and during a pause.
    joiner.update(_observations([]), removed_tracklet_ids=[20], pause_reconnection=True)
    history = joiner.pop_finalized_results()
    assert history[5].tracker_id.tolist() == [1]


@pytest.mark.parametrize("first_frame", [1, 2], ids=["established", "pending"])
def test_pause_preserves_actual_samples_and_ordinary_sampling_interval(first_frame) -> None:
    encoder = _Encoder()
    joiner = LongTermReID(encoder, LongTermReIDConfig(maximum_reconnection_age=100))
    sampled = []
    for step in range(1, 18):
        count = len(encoder.calls)
        paused = step == 6
        joiner.update(
            _observations([10]),
            _frame(),
            frame_number=step + first_frame - 1,
            extract_features=not paused,
            pause_reconnection=paused,
        )
        if len(encoder.calls) != count:
            sampled.append(step)
    assert sampled == [1, 7, 12, 17]
    assert joiner._tracks[10].samples == 4


@pytest.mark.parametrize("buffer_results", [True, False])
def test_paused_track_does_not_block_ready_history_or_finish(buffer_results: bool) -> None:
    joiner = LongTermReID(
        _Encoder(),
        LongTermReIDConfig(
            maximum_reconnection_age=1,
            buffer_results=buffer_results,
        ),
    )
    joiner.update(_observations([]))
    joiner.update(_observations([10]))
    joiner.update(_observations([10]))
    assert joiner.update(_observations([10, 20])).tracker_id.tolist() == [0, -1]
    joiner.update(_observations([20]), pause_reconnection=True)
    if buffer_results:
        assert list(joiner.pop_finalized_results()) == [2, 3]
        assert joiner.public_ids[20] == -1
    tail = joiner.finish()
    assert list(tail) == ([4, 5] if buffer_results else [])
    assert joiner.public_ids[20] == 1


def test_pause_configuration_and_registration_seed_validation() -> None:
    invalid: list[Any] = [-1, True, 1.5]
    for cap in invalid:
        with pytest.raises(ValueError, match="maximum_reconnection_pause_steps"):
            LongTermReIDConfig(maximum_reconnection_pause_steps=cap)
    for seed in [*invalid, 2]:
        joiner = LongTermReID(_Encoder())
        joiner.update(_observations([]))
        with pytest.raises(ValueError, match="paused steps"):
            joiner.update(_observations([10]), tracklet_starts={10: (1, 1)}, tracklet_paused_steps={10: seed})


def test_registration_on_duplicate_step_does_not_add_pause() -> None:
    """Registering a track on an already consumed step does not pause its clock."""
    joiner = LongTermReID(_Encoder())
    joiner.update(_observations([]), step=2)
    joiner.update(_observations([10]), step=2, pause_reconnection=True, tracklet_starts={10: (1, 1)})
    assert joiner._tracks[10].paused_steps == 0


def test_retry_after_failed_extraction_does_not_add_extra_pause() -> None:
    """Retrying a failed extraction cannot count the same pause twice."""
    encoder = _Encoder()
    joiner = LongTermReID(encoder)
    joiner.update(_observations([]), step=1)
    joiner.update(_observations([20]), step=2)
    encoder.default = (np.nan, 0.0)
    with pytest.raises(ValueError):
        joiner.update(_observations([20]), _frame(), step=3, pause_reconnection=True)
    assert joiner._tracks[20].samples == 0
    # The failed call already advanced the step and counted its pause.
    joiner.update(_observations([20]), step=3, pause_reconnection=True)
    assert joiner._tracks[20].paused_steps == 1


def test_first_frame_ids_are_final_without_images_and_mapping_is_a_copy() -> None:
    joiner = LongTermReID(_Encoder())
    observations = _observations([70, 12])

    live = joiner.update(observations)
    identities = live.tracker_id.tolist()
    assert min(identities) >= 0
    assert len(set(identities)) == 2
    np.testing.assert_array_equal(live.data["tracklet_id"], [70, 12])
    np.testing.assert_array_equal(live.data["reid_pending"], [False, False])
    np.testing.assert_array_equal(observations.tracker_id, [70, 12])
    assert observations.data == {}

    mapping = joiner.public_ids
    mapping[70] = -99
    assert joiner.public_ids == dict(zip([70, 12], identities))
    assert joiner.pop_finalized_results() == {}


def test_matching_is_mutual_best_without_assigning_second_choice_pairs() -> None:
    encoder = _Encoder()
    encoder.features = {10: (1.0, 0.0), 70: (0.0, 1.0), 130: (0.8, 0.6), 190: (0.9, np.sqrt(0.19))}
    joiner = LongTermReID(encoder, LongTermReIDConfig(minimum_cosine_similarity=0.4))
    old = joiner.update(_observations([100, 200]), _frame())
    joiner.update(_observations([]), removed_tracklet_ids=[100, 200])

    # Both candidates prefer old[0]. Only candidate 400 is also its preferred
    # claimant; assigning candidate 300 to old[1] would change the policy.
    live = joiner.update(_observations([300, 400], [(130, 10, 160, 70), (190, 10, 220, 70)]), _frame())

    np.testing.assert_array_equal(live.tracker_id, [-1, old.tracker_id[0]])
    np.testing.assert_array_equal(live.data["reid_pending"], [True, False])
    np.testing.assert_array_equal(live.data["tracklet_id"], [300, 400])
    assert joiner.pop_finalized_results() == {}


def test_mutual_best_ties_are_deterministic_independent_of_current_row_order() -> None:
    joiner = LongTermReID(_Encoder())
    old = joiner.update(_observations([100, 200]), _frame())

    live = joiner.update(_observations([400, 300]), _frame(), removed_tracklet_ids=[200, 100])

    np.testing.assert_array_equal(live.tracker_id, [-1, min(old.tracker_id)])
    np.testing.assert_array_equal(live.data["tracklet_id"], [400, 300])


@pytest.mark.parametrize("threshold, reconnects", [(0.8, True), (0.8001, False)])
def test_cosine_similarity_threshold_is_inclusive(threshold: float, reconnects: bool) -> None:
    encoder = _Encoder()
    joiner = LongTermReID(encoder, LongTermReIDConfig(minimum_cosine_similarity=threshold))
    old = joiner.update(_observations([100]), _frame())
    encoder.default = (0.8, 0.6)

    live = joiner.update(_observations([200]), _frame(), removed_tracklet_ids=[100])

    assert live.tracker_id[0] == (old.tracker_id[0] if reconnects else -1)


@pytest.mark.parametrize("old_samples, new_samples, reconnects", [(1, 2, False), (2, 1, False), (2, 2, True)])
def test_actual_samples_gate_both_sides_independently_of_history_capacity(
    old_samples: int, new_samples: int, reconnects: bool
) -> None:
    joiner = LongTermReID(
        _Encoder(),
        LongTermReIDConfig(minimum_feature_samples=2, minimum_feature_frame_gap=1, feature_history_size=1),
    )
    old = joiner.update(_observations([100]), _frame())
    for _ in range(old_samples - 1):
        joiner.update(_observations([100]), _frame())

    live = joiner.update(_observations([200]), _frame(), removed_tracklet_ids=[100])
    assert live.tracker_id[0] == -1
    for _ in range(new_samples - 1):
        live = joiner.update(_observations([200]), _frame())

    assert live.tracker_id[0] == (old.tracker_id[0] if reconnects else -1)


def test_zero_descriptor_is_not_a_sample_and_does_not_delay_retry() -> None:
    encoder = _Encoder()
    joiner = LongTermReID(encoder)
    old = joiner.update(_observations([100]), _frame())
    encoder.default = (0.0, 0.0)
    pending = joiner.update(_observations([200]), _frame(), removed_tracklet_ids=[100])
    assert pending.tracker_id[0] == -1

    # The default five-step gap applies only after a successful extraction.
    encoder.default = (1.0, 0.0)
    live = joiner.update(_observations([200]), _frame())

    assert live.tracker_id[0] == old.tracker_id[0]
    assert len(encoder.calls) == 3


def test_descriptor_is_normalized_mean_of_latest_normalized_samples_not_ema() -> None:
    encoder = _Encoder()
    joiner = LongTermReID(
        encoder,
        LongTermReIDConfig(feature_history_size=2, minimum_feature_frame_gap=1, minimum_cosine_similarity=0.999),
    )
    encoder.default = (100.0, 0.0)
    old = joiner.update(_observations([100]), _frame())
    encoder.default = (0.0, 2.0)
    joiner.update(_observations([100]), _frame())
    encoder.default = (1.0, 0.0)
    joiner.update(_observations([100]), _frame())

    # Last two unit samples average to the diagonal. Raw averaging, keeping
    # the evicted first sample, and EMA each miss this deliberately tight gate.
    encoder.default = (1.0, 1.0)
    live = joiner.update(_observations([200]), _frame(), removed_tracklet_ids=[100])

    assert live.tracker_id[0] == old.tracker_id[0]


def test_lost_identity_is_reserved_until_termination_and_uses_last_observation() -> None:
    joiner = LongTermReID(_Encoder())
    old = joiner.update(_observations([100]), _frame())
    pending = joiner.update(_observations([200]), _frame())
    assert pending.tracker_id[0] == -1
    assert joiner.public_ids[100] == old.tracker_id[0]
    assert joiner.pop_finalized_results() == {}
    joiner.update(_observations([]))

    # Removal is later than the new birth, but the old last observation is
    # earlier. Existing features must work even with no image at this update.
    live = joiner.update(_observations([200]), removed_tracklet_ids=[100, 100])
    history = joiner.pop_finalized_results()

    assert live.tracker_id[0] == old.tracker_id[0]
    assert set(history) == {2}
    np.testing.assert_array_equal(history[2].tracker_id, old.tracker_id)
    assert joiner.public_ids == {200: old.tracker_id[0]}
    assert joiner.pop_finalized_results() == {}


def test_reconnected_identity_remains_reserved_while_its_new_owner_is_lost() -> None:
    joiner = LongTermReID(_Encoder())
    old = joiner.update(_observations([100]), _frame())
    reconnected = joiner.update(_observations([200]), _frame(), removed_tracklet_ids=[100])
    np.testing.assert_array_equal(reconnected.tracker_id, old.tracker_id)

    competing = joiner.update(_observations([300]), _frame())
    assert competing.tracker_id[0] == -1
    returned = joiner.update(_observations([200, 300]))

    np.testing.assert_array_equal(returned.tracker_id, [old.tracker_id[0], -1])


def test_terminating_pending_track_releases_history_before_same_update_reconnection() -> None:
    joiner = LongTermReID(_Encoder())
    joiner.update(_observations([]))
    pending = joiner.update(_observations([100]), _frame())
    assert pending.tracker_id[0] == -1

    live = joiner.update(_observations([200]), _frame(), removed_tracklet_ids=[100])
    history = joiner.pop_finalized_results()

    assert live.tracker_id[0] >= 0
    assert set(history) == {2}
    np.testing.assert_array_equal(history[2].tracker_id, live.tracker_id)
    np.testing.assert_array_equal(history[2].data["tracklet_id"], [100])
    np.testing.assert_array_equal(live.data["tracklet_id"], [200])
    assert joiner.pop_finalized_results() == {}


@pytest.mark.parametrize("buffer_results", [True, False])
def test_overlapping_higher_similarity_gallery_entry_cannot_hide_valid_alternative(buffer_results: bool) -> None:
    encoder = _Encoder()
    encoder.features = {10: (0.8, 0.6), 70: (1.0, 0.0), 130: (1.0, 0.0)}
    joiner = LongTermReID(encoder, LongTermReIDConfig(buffer_results=buffer_results))
    old = joiner.update(_observations([100, 200]), _frame())
    joiner.update(_observations([200], [(70, 10, 100, 70)]))
    pending = joiner.update(
        _observations([200, 300], [(70, 10, 100, 70), (130, 10, 160, 70)]),
        _frame(),
        tracklet_starts={300: (2, 2)},
    )
    assert pending.tracker_id[1] == -1

    # Identity 200 has a perfect appearance match, but overlaps the new
    # trajectory's history. Remove that pair before choosing the row maximum.
    live = joiner.update(_observations([300]), removed_tracklet_ids=[100, 200])

    assert live.tracker_id[0] == old.tracker_id[0]
    if buffer_results:
        history = joiner.pop_finalized_results()
        assert set(history) == {3}
        assert history[3].tracker_id[0] != pending.tracker_id[0]


def test_same_frame_incompatible_claimant_cannot_hide_valid_column_alternative() -> None:
    encoder = _Encoder()
    encoder.features = {10: (1.0, 0.0), 70: (1.0, 0.0), 130: (0.8, 0.6)}
    joiner = LongTermReID(encoder)
    old = joiner.update(_observations([100]), _frame())
    overlapping = joiner.update(_observations([100, 200]), _frame())
    assert overlapping.tracker_id[1] == -1

    live = joiner.update(
        _observations([200, 300], [(70, 10, 100, 70), (130, 10, 160, 70)]),
        _frame(),
        removed_tracklet_ids=[100],
    )

    np.testing.assert_array_equal(live.tracker_id, [-1, old.tracker_id[0]])


def test_termination_after_reconnection_archives_latest_segment_endpoint() -> None:
    joiner = LongTermReID(_Encoder())
    old = joiner.update(_observations([100]), _frame())
    joiner.update(_observations([200]), _frame(), removed_tracklet_ids=[100])
    joiner.update(_observations([200, 300]), _frame())

    live = joiner.update(_observations([300]), removed_tracklet_ids=[200])

    assert live.tracker_id[0] == -1
    tail = joiner.finish()
    assert all(row.tracker_id[0] != old.tracker_id[0] for row in tail.values())


def test_fresh_ids_never_reuse_prior_allocations_even_when_internal_ids_collide() -> None:
    encoder = _Encoder()
    encoder.features = {70: (0.0, 1.0)}
    joiner = LongTermReID(encoder, LongTermReIDConfig(maximum_reconnection_age=0))
    old = joiner.update(_observations([99, 100]), _frame())
    original_ids = old.tracker_id.copy()
    colliding_internal_ids = old.tracker_id.tolist()

    joiner.update(_observations(colliding_internal_ids), removed_tracklet_ids=[99, 100])
    fresh = joiner.update(_observations(colliding_internal_ids))
    assert set(fresh.tracker_id).isdisjoint(old.tracker_id)
    assert len(set(fresh.tracker_id)) == 2
    joiner.pop_finalized_results()

    # Reusing an earlier public ID is intentional only for an accepted join.
    reused = joiner.update(_observations([200]), _frame(), removed_tracklet_ids=colliding_internal_ids)
    assert reused.tracker_id[0] == old.tracker_id[0]
    joiner.update(_observations([300]))
    newest = joiner.update(_observations([300]))

    assert newest.tracker_id[0] not in set(old.tracker_id) | set(fresh.tracker_id)
    np.testing.assert_array_equal(old.tracker_id, original_ids)


def test_pending_tracks_release_independently_and_frame_keys_can_return_again() -> None:
    joiner = LongTermReID(_Encoder(), LongTermReIDConfig(maximum_reconnection_age=1))
    joiner.update(_observations([]))
    pending = joiner.update(_observations([-1, 20, 30]))
    np.testing.assert_array_equal(pending.data["reid_pending"], [False, True, True])
    assert joiner.pop_finalized_results() == {}

    live = joiner.update(_observations([30]), removed_tracklet_ids=[20])
    assert live.tracker_id[0] == -1
    first = joiner.pop_finalized_results()
    assert set(first) == {2}
    np.testing.assert_array_equal(first[2].data["tracklet_id"], [20])
    assert joiner.pop_finalized_results() == {}

    live = joiner.update(_observations([30]))
    second = joiner.pop_finalized_results()
    assert set(second) == {2, 3}
    assert first[2].tracker_id[0] != live.tracker_id[0]
    for row in second.values():
        np.testing.assert_array_equal(row.data["tracklet_id"], [30])
        np.testing.assert_array_equal(row.tracker_id, live.tracker_id)
        np.testing.assert_array_equal(row.data["reid_pending"], [False])
    assert joiner.finish() == {}


def test_lost_pending_track_keeps_history_until_observed_or_terminated() -> None:
    joiner = LongTermReID(_Encoder(), LongTermReIDConfig(maximum_reconnection_age=1))
    joiner.update(_observations([]))
    joiner.update(_observations([20]))
    joiner.update(_observations([]))
    joiner.update(_observations([]))

    assert joiner.public_ids == {20: -1}
    assert joiner.pop_finalized_results() == {}
    live = joiner.update(_observations([20]))
    history = joiner.pop_finalized_results()

    assert set(history) == {2}
    assert live.tracker_id[0] >= 0
    np.testing.assert_array_equal(history[2].tracker_id, live.tracker_id)


def test_historical_rows_own_copies_and_exclude_masks_and_collection_metadata() -> None:
    joiner = LongTermReID(_Encoder())
    joiner.update(_observations([]))
    observations = _observations([20])
    observations.mask = np.ones((1, 240, 400), dtype=bool)
    observations.metadata = {"frame": _frame(), "application": {"name": "original"}}
    observations.data = {
        "payload": np.array([{"tags": ["original"]}], dtype=object),
        "values": np.array([[2.0, 3.0]]),
        "tracklet_id": np.array([999]),
        "reid_pending": np.array([False]),
    }
    live = joiner.update(observations)
    np.testing.assert_array_equal(observations.data["tracklet_id"], [999])
    np.testing.assert_array_equal(observations.data["reid_pending"], [False])
    assert live.mask is not None
    assert live.metadata["application"] == {"name": "original"}

    observations.xyxy[:] = 0
    observations.confidence[:] = 0
    observations.class_id[:] = 9
    live.xyxy[:] = 5
    live.data["payload"][0]["tags"][0] = "mutated"
    live.data["values"][:] = -1
    joiner.update(_observations([]), removed_tracklet_ids=[20])
    history = joiner.pop_finalized_results()[2]

    np.testing.assert_array_equal(history.xyxy, [[10, 10, 40, 70]])
    np.testing.assert_allclose(history.confidence, [0.9])
    np.testing.assert_array_equal(history.class_id, [0])
    np.testing.assert_array_equal(history.data["values"], [[2.0, 3.0]])
    assert history.data["payload"][0] == {"tags": ["original"]}
    np.testing.assert_array_equal(history.data["tracklet_id"], [20])
    np.testing.assert_array_equal(history.data["reid_pending"], [False])
    assert history.mask is None
    assert history.metadata == {}


def test_finish_delivers_unread_ready_rows_and_pending_tail_exactly_once() -> None:
    joiner = LongTermReID(_Encoder())
    joiner.update(_observations([]))
    joiner.update(_observations([20, 30]))
    joiner.update(_observations([]), removed_tracklet_ids=[20, 999])

    tail = joiner.finish()

    assert set(tail) == {2}
    np.testing.assert_array_equal(tail[2].data["tracklet_id"], [20, 30])
    assert min(tail[2].tracker_id) >= 0
    assert len(set(tail[2].tracker_id)) == 2
    assert joiner.finish() == {}
    assert joiner.pop_finalized_results() == {}
    with pytest.raises(RuntimeError, match="reset"):
        joiner.update(_observations([]))


def test_reset_discards_pending_and_unread_results_and_restarts_identity_namespace() -> None:
    """Reset clears queued output and old appearances before starting fresh identities."""
    encoder = _Encoder()
    config = LongTermReIDConfig()
    joiner = LongTermReID(encoder, config)
    joiner.update(_observations([]))
    joiner.update(_observations([20, 30]), _frame())
    joiner.update(_observations([]), removed_tracklet_ids=[20])

    joiner.reset()

    assert joiner.model is encoder
    assert joiner.config is config
    assert joiner.frame_number == 0
    assert joiner.public_ids == {}
    assert joiner.pop_finalized_results() == {}
    # A new first-frame ID or overlapping times could conceal a stale gallery.
    # Start after the old track's last observation (frame 2) with the same appearance.
    joiner.update(_observations([]))
    joiner.update(_observations([]))
    pending = joiner.update(_observations([20]), _frame())
    assert pending.tracker_id.tolist() == [-1]
    assert pending.data["reid_pending"].tolist() == [True]
    assert joiner.frame_number == 3
    tail = joiner.finish()
    assert list(tail) == [3]
    assert tail[3].tracker_id.tolist() == [0]
    joiner.reset()
    assert joiner.update(_observations([20])).tracker_id.tolist() == [0]


def test_disabling_buffering_preserves_joining_and_finish_closes_sequence() -> None:
    joiner = LongTermReID(_Encoder(), LongTermReIDConfig(buffer_results=False))
    old = joiner.update(_observations([100]), _frame())
    pending = joiner.update(_observations([200]), _frame())
    assert pending.tracker_id[0] == -1
    with pytest.raises(RuntimeError, match="buffer_results"):
        joiner.pop_finalized_results()

    live = joiner.update(_observations([200]), removed_tracklet_ids=[100])

    assert live.tracker_id[0] == old.tracker_id[0]
    assert joiner.finish() == {}
    assert joiner.finish() == {}
    with pytest.raises(RuntimeError, match="reset"):
        joiner.update(_observations([]))


@pytest.mark.parametrize("confidence, sampled", [(0.6, True), (0.5999, False), (np.nan, False), (None, True)])
def test_confidence_gate_uses_inclusive_cutoff_and_repository_default(confidence: float | None, sampled: bool) -> None:
    encoder = _Encoder()
    joiner = LongTermReID(encoder)
    observations = _observations([20])
    # Trackers treats absent confidence as 1.0 through default_confidences.
    observations.confidence = None if confidence is None else np.array([confidence])

    joiner.update(observations, _frame())

    assert len(encoder.calls) == int(sampled)


@pytest.mark.parametrize("width, height, sampled", [(25, 50, True), (24, 50, False), (25, 49, False)])
def test_box_width_and_height_gates_are_inclusive(width: float, height: float, sampled: bool) -> None:
    encoder = _Encoder()
    joiner = LongTermReID(encoder)

    joiner.update(_observations([20], [(10, 10, 10 + width, 10 + height)]), _frame())

    assert len(encoder.calls) == int(sampled)


def test_box_area_gate_is_independent_of_width_and_height_minima() -> None:
    encoder = _Encoder()
    joiner = LongTermReID(encoder, LongTermReIDConfig(minimum_box_width=1, minimum_box_height=1))
    boxes = [(10, 10, 42, 42), (70, 10, 101, 42)]

    joiner.update(_observations([20, 30], boxes), _frame())

    assert len(encoder.calls) == 1
    np.testing.assert_array_equal(encoder.calls[0], [boxes[0]])


@pytest.mark.parametrize(
    "touching, inside",
    [
        ((4, 10, 40, 70), (4.01, 10, 40, 70)),
        ((10, 4, 40, 70), (10, 4.01, 40, 70)),
        ((350, 10, 396, 70), (350, 10, 395.99, 70)),
        ((10, 170, 40, 236), (10, 170, 40, 235.99)),
    ],
)
def test_all_image_border_margins_are_strict(
    touching: tuple[float, float, float, float], inside: tuple[float, float, float, float]
) -> None:
    encoder = _Encoder()
    joiner = LongTermReID(encoder)

    joiner.update(_observations([20], [touching]), _frame())
    assert encoder.calls == []
    joiner.update(_observations([20], [inside]), _frame())

    assert len(encoder.calls) == 1


def test_feature_gap_counts_missing_images_but_not_duplicate_steps() -> None:
    encoder = _Encoder()
    joiner = LongTermReID(encoder, LongTermReIDConfig(minimum_feature_frame_gap=3))
    joiner.update(_observations([20]), _frame())
    joiner.update(_observations([20]), _frame(), step=1)
    assert len(encoder.calls) == 1
    assert joiner.frame_number == 2
    joiner.update(_observations([20]))
    joiner.update(_observations([20]), _frame())
    assert len(encoder.calls) == 1

    joiner.update(_observations([20]), _frame())
    assert len(encoder.calls) == 2
    joiner.update(_observations([20]), _frame(), step=4)
    assert len(encoder.calls) == 2
    assert joiner.frame_number == 6


@pytest.mark.parametrize("confirmation_step, reconnects", [(4, True), (5, False)])
def test_reconnection_age_starts_at_birth_with_inclusive_deadline(confirmation_step: int, reconnects: bool) -> None:
    joiner = LongTermReID(_Encoder(), LongTermReIDConfig(maximum_reconnection_age=2))
    old = joiner.update(_observations([100]), _frame())

    live = joiner.update(
        _observations([200]),
        _frame(),
        tracklet_starts={200: (2, 2)},
        removed_tracklet_ids=[100],
        frame_number=confirmation_step,
        step=confirmation_step,
    )

    assert live.tracker_id[0] >= 0
    assert (live.tracker_id[0] == old.tracker_id[0]) == reconnects
    assert joiner.frame_number == confirmation_step
    assert joiner.pop_finalized_results() == {}


def test_delayed_confirmation_of_first_frame_birth_finalizes_without_features() -> None:
    joiner = LongTermReID(_Encoder())
    joiner.update(_observations([]))
    joiner.update(_observations([]))

    live = joiner.update(_observations([20]), tracklet_starts={20: (1, 1)})

    assert live.tracker_id[0] >= 0
    assert not live.data["reid_pending"][0]
    assert joiner.finish() == {}


@pytest.mark.parametrize("clock", ["frame_number", "step"])
def test_invalid_clocks_do_not_consume_a_processing_reference(clock: str) -> None:
    joiner = LongTermReID(_Encoder())
    joiner.update(_observations([20]))

    with pytest.raises(ValueError, match=clock):
        if clock == "frame_number":
            joiner.update(_observations([20]), frame_number=0)
        else:
            joiner.update(_observations([20]), step=0)

    assert joiner.frame_number == 1
    joiner.update(_observations([20]))
    assert joiner.frame_number == 2


@pytest.mark.parametrize("invalid", ["image_dtype", "image_shape", "reference_boxes", "birth", "pause_seed"])
def test_invalid_update_arguments_leave_identity_state_available_for_retry(invalid: str) -> None:
    encoder = _Encoder()
    joiner = LongTermReID(encoder)
    joiner.update(_observations([100]), _frame())
    joiner.update(_observations([200]), _frame())
    options: dict[str, Any] = {
        "frame": _frame(),
        "frame_number": 3,
        "step": 3,
        "removed_tracklet_ids": [100],
        "pause_reconnection": True,
    }
    if invalid == "image_dtype":
        options["frame"] = _frame().astype(np.float32)
    elif invalid == "image_shape":
        options["frame"] = _frame()[:, :, 0]
    elif invalid == "reference_boxes":
        options["reference_boxes"] = np.zeros((2, 3))
    elif invalid == "birth":
        options["tracklet_starts"] = {400: (4, 4)}
    else:
        options["tracklet_paused_steps"] = {400: 1}  # Raw age is still zero.

    # A late invalid registration must not retire the old owner, age the
    # pending track or register the earlier valid row from the same batch.
    with pytest.raises(ValueError):
        joiner.update(_observations([200, 300, 400]), **options)
    assert (joiner.frame_number, joiner._step) == (2, 2)
    assert joiner.public_ids == {100: 0, 200: -1}
    assert joiner._owners == {0: 100}
    assert joiner._retired == set()
    assert joiner._gallery == {}
    assert joiner._tracks[200].paused_steps == 0
    assert joiner._tracks[200].last_frame == 2
    assert len(encoder.calls) == 2
    assert joiner.pop_finalized_results() == {}

    current = joiner.update(
        _observations([200, 300, 400]),
        _frame(),
        frame_number=3,
        step=3,
        removed_tracklet_ids=[100],
        pause_reconnection=True,
    )
    assert current.tracker_id.tolist() == [0, -1, -1]
    history = joiner.pop_finalized_results()
    assert list(history) == [2]
    assert history[2].tracker_id.tolist() == [0]
