# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""CBIoU-specific tracker tests.

Generic lifecycle contracts are covered in test_trackers.py via ALL_TRACKER_IDS.
This file covers C-BIoU-specific invariants (Yang et al., WACV 2023):
  - Cascaded BIoU with per-step buffer scales (b1, b2)
  - CMC disabled; frame argument triggers UserWarning
  - BIoU association more tolerant than standard IoU
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import supervision as sv

from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.botsort.tracklet import BoTSORTTracklet
from trackers.core.cbiou.tracker import CBIoUTracker
from trackers.utils.iou import BIoU


def _detection(xyxy: tuple[float, float, float, float], conf: float = 0.9) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([conf], dtype=np.float32),
    )


def _make_frame(h: int = 480, w: int = 640, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


class TestCBIoUConstruction:
    def test_default_construction(self) -> None:
        tracker = CBIoUTracker()
        assert tracker is not None

    def test_per_step_biou_instances(self) -> None:
        tracker = CBIoUTracker(
            buffer_ratio_first=0.1,
            buffer_ratio_second=0.3,
        )
        assert isinstance(tracker.iou_first, BIoU)
        assert isinstance(tracker.iou_second, BIoU)
        assert not hasattr(tracker, "iou_unconfirmed")

    def test_buffer_ratios_forwarded_to_biou(self) -> None:
        tracker = CBIoUTracker(
            buffer_ratio_first=0.1,
            buffer_ratio_second=0.3,
        )
        assert tracker.iou_first.buffer_ratio == pytest.approx(0.1)
        assert tracker.iou_second.buffer_ratio == pytest.approx(0.3)

    def test_cmc_disabled(self) -> None:
        tracker = CBIoUTracker()
        assert tracker.enable_cmc is False
        assert tracker.cmc is None

    def test_tracker_id(self) -> None:
        assert CBIoUTracker.tracker_id == "cbiou"

    def test_invalid_buffer_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="buffer_ratio"):
            CBIoUTracker(buffer_ratio_first=-0.01)


class TestCBIoUFrameWarning:
    def test_frame_triggers_warning(self) -> None:
        tracker = CBIoUTracker()
        with pytest.warns(UserWarning):
            tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame=_make_frame())

    def test_no_warning_without_frame(self) -> None:
        tracker = CBIoUTracker()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tracker.update(_detection((100.0, 100.0, 200.0, 200.0)))


class TestCBIoUAssociationTolerance:
    """BIoU should associate near-miss detections that plain IoU would miss."""

    def test_near_miss_associated_with_buffer(self) -> None:
        """
        A track initialized at box A, then a detection at box B just outside
        should be associated by CBIoU (buffer expands boxes) but not by
        BoTSORT with standard IoU (tight threshold).

        Box A: [0, 0, 100, 100]  (100x100)
        Box B: [110, 0, 210, 100]  (gap of 10px = 10% of width)
        With buffer_ratio=0.15 each side expands by 15px, so A becomes
        [-15, -15, 115, 115] and B becomes [95, -15, 225, 115] —
        they now overlap.
        """
        # Frame 1: spawn a track at box A with high confidence
        cbiou = CBIoUTracker(
            buffer_ratio_first=0.15,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.5,
            minimum_iou_threshold_first_assoc=0.05,
        )
        botsort = BoTSORTTracker(
            enable_cmc=False,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.5,
            minimum_iou_threshold_first_assoc=0.05,
        )

        box_a = (0.0, 0.0, 100.0, 100.0)
        box_b = (110.0, 0.0, 210.0, 100.0)

        cbiou.update(_detection(box_a))
        botsort.update(_detection(box_a))
        botsort_frame1_track_id = next((t.tracker_id for t in botsort.tracks), None)

        # Frame 2: detection slightly outside A — CBIoU buffer closes the gap
        cbiou_result = cbiou.update(_detection(box_b))
        botsort_result = botsort.update(_detection(box_b))

        assert cbiou_result.tracker_id is not None and len(cbiou_result.tracker_id) == 1
        assert cbiou_result.tracker_id[0] >= 0
        cbiou_frame1_id = cbiou.tracks[0].tracker_id
        assert cbiou_result.tracker_id[0] == cbiou_frame1_id

        botsort_ids = botsort_result.tracker_id
        if botsort_ids is not None and len(botsort_ids) > 0 and botsort_frame1_track_id is not None:
            assert botsort_ids[0] != botsort_frame1_track_id


class TestCBIoUZeroBufferEquivalence:
    """With buffer_ratio=0, BIoU recovers IoU; C-BIoU should match BoT-SORT (no CMC)."""

    def test_zero_buffer_matches_botsort_without_cmc(self) -> None:
        """CBIoU(buffer=0) and BoTSORT(no CMC) stay equivalent, including across a miss/gap frame."""
        detections = [
            _detection((0.0, 0.0, 50.0, 50.0)),
            _detection((5.0, 5.0, 55.0, 55.0)),
            sv.Detections.empty(),  # miss/gap frame: both trackers should prune identically
            _detection((100.0, 100.0, 150.0, 150.0)),
            _detection((105.0, 105.0, 155.0, 155.0)),
            _detection((8.0, 8.0, 58.0, 58.0)),
        ]

        def run_tracker(tracker: CBIoUTracker | BoTSORTTracker) -> list[sv.Detections]:
            tracker.reset()
            return [tracker.update(det) for det in detections]

        cbiou = CBIoUTracker(
            buffer_ratio_first=0.0,
            buffer_ratio_second=0.0,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.5,
            minimum_iou_threshold_first_assoc=0.3,
            minimum_iou_threshold_second_assoc=0.3,
            minimum_iou_threshold_unconfirmed_assoc=0.3,
            high_conf_det_threshold=0.6,
        )
        botsort = BoTSORTTracker(
            enable_cmc=False,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.5,
            minimum_iou_threshold_first_assoc=0.3,
            minimum_iou_threshold_second_assoc=0.3,
            minimum_iou_threshold_unconfirmed_assoc=0.3,
            high_conf_det_threshold=0.6,
        )

        cbiou_results = run_tracker(cbiou)
        botsort_results = run_tracker(botsort)

        for frame_idx, (r_cbiou, r_botsort) in enumerate(zip(cbiou_results, botsort_results)):
            assert len(r_cbiou) == len(r_botsort), (
                f"frame {frame_idx}: CBIoU(buffer=0) and BoTSORT(no CMC) returned different "
                f"detection counts ({len(r_cbiou)} vs {len(r_botsort)})"
            )
            np.testing.assert_array_equal(
                r_cbiou.tracker_id,
                r_botsort.tracker_id,
                err_msg=f"frame {frame_idx}: different tracker IDs",
            )
            if len(r_cbiou) > 0:
                np.testing.assert_allclose(
                    r_cbiou.xyxy.astype(np.float32),
                    r_botsort.xyxy.astype(np.float32),
                    err_msg=f"frame {frame_idx}: different boxes",
                )


class TestCBIoUSearchSpace:
    def test_cascade_buffer_params_in_search_space(self) -> None:
        ss = CBIoUTracker.search_space
        assert "buffer_ratio_first" in ss
        assert "buffer_ratio_second" in ss
        assert "buffer_ratio_unconfirmed" not in ss

    def test_no_cmc_in_search_space(self) -> None:
        ss = CBIoUTracker.search_space
        assert "enable_cmc" not in ss
        assert "cmc_method" not in ss


class TestCBIoUUnmatchedLowConfidence:
    def test_unmatched_low_conf_detection_has_minus_one_tracker_id(self) -> None:
        """Unmatched low-confidence detection appears in update() output with tracker_id=-1."""
        tracker = CBIoUTracker(
            minimum_consecutive_frames=1,
            high_conf_det_threshold=0.6,
            buffer_ratio_first=0.1,
            buffer_ratio_second=0.3,
            minimum_iou_threshold_second_assoc=0.1,
        )
        # Frame 1: establish a confirmed track near origin
        tracker.update(
            sv.Detections(
                xyxy=np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
                confidence=np.array([0.9], dtype=np.float32),
            )
        )
        # Frame 2: low-confidence detection far from any track (no IoU overlap)
        result = tracker.update(
            sv.Detections(
                xyxy=np.array([[500.0, 500.0, 510.0, 510.0]], dtype=np.float32),
                confidence=np.array([0.3], dtype=np.float32),
            )
        )
        assert len(result) == 1
        assert result.tracker_id is not None
        assert result.tracker_id[0] == -1


class TestCBIoUStickyMaturity:
    def test_instant_activation_off_track_pruned_on_miss(self) -> None:
        """With instant activation off, an unmatured track is pruned on a miss."""
        tracker = CBIoUTracker(instant_first_frame_activation=False)
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
        tracker = CBIoUTracker()
        obj = (10.0, 10.0, 50.0, 50.0)

        first = tracker.update(_detection(obj))
        assert first.tracker_id is not None
        track_id = int(first.tracker_id[0])

        tracker.update(sv.Detections.empty())  # miss 1: time_since_update=1 → confirmed
        tracker.update(sv.Detections.empty())  # miss 2: time_since_update=2 → lost_tracks

        assert any(t.tracker_id == track_id for t in tracker.tracks)

        returned = tracker.update(_detection(obj))
        assert returned.tracker_id is not None
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
        tracker = CBIoUTracker(minimum_consecutive_frames=mcf)
        obj = (10.0, 10.0, 50.0, 50.0)

        first = tracker.update(_detection(obj))
        assert first.tracker_id is not None
        track_id = int(first.tracker_id[0])

        tracker.update(sv.Detections.empty())

        assert any(t.tracker_id == track_id for t in tracker.tracks)

        returned = tracker.update(_detection(obj))
        assert returned.tracker_id is not None
        assert track_id in returned.tracker_id.tolist()

    def test_instant_activated_track_survives_a_miss_with_shifted_return_box(self) -> None:
        """Sticky track keeps its ID after a miss even when it reappears shifted.

        Box A (spawn): [0, 0, 100, 100]. Box B (return, after the miss):
        [100, 0, 200, 100] — flush against A's right edge, so plain IoU is
        exactly 0 (no overlap: Step 1's raw box overlap would miss it). With
        the tracker's default ``buffer_ratio_first=0.3``, BIoU expands both
        boxes by 30px on every side before Step 1 association, producing
        enough overlap to re-associate the returning detection with the
        sticky (instant-activated) track. Mirrors the box-A/box-B
        construction in ``TestCBIoUAssociationTolerance``.
        """
        tracker = CBIoUTracker()
        box_a = (0.0, 0.0, 100.0, 100.0)
        box_b = (100.0, 0.0, 200.0, 100.0)

        first = tracker.update(_detection(box_a, conf=1.0))
        assert first.tracker_id is not None
        track_id = int(first.tracker_id[0])

        tracker.update(sv.Detections.empty())  # no detections: object missed this frame
        assert any(t.tracker_id == track_id for t in tracker.tracks)

        returned = tracker.update(_detection(box_b, conf=1.0))  # object reappears, shifted
        assert returned.tracker_id is not None
        assert track_id in returned.tracker_id.tolist()

    def test_sticky_track_pruned_once_time_since_update_exceeds_lost_track_buffer(self) -> None:
        """Sticky maturity delays deletion; it does not grant permanent immunity.

        With ``lost_track_buffer=1`` (and the default 30 FPS ``frame_rate``),
        ``maximum_frames_without_update`` scales to 1. An instant-activated
        (sticky) track survives the first miss (``time_since_update == 1``,
        within budget) but must be pruned once a second consecutive miss
        pushes ``time_since_update`` to 2, past the budget.
        """
        tracker = CBIoUTracker(lost_track_buffer=1)
        obj = (10.0, 10.0, 50.0, 50.0)

        first = tracker.update(_detection(obj))
        assert first.tracker_id is not None
        track_id = int(first.tracker_id[0])

        tracker.update(sv.Detections.empty())  # miss 1: time_since_update=1, within budget
        assert any(t.tracker_id == track_id for t in tracker.tracks)

        tracker.update(sv.Detections.empty())  # miss 2: time_since_update=2, exceeds budget
        assert not any(t.tracker_id == track_id for t in tracker.tracks)

    def test_sticky_and_immature_tracks_on_shared_miss_frame_no_cross_contamination(self) -> None:
        """Sticky and immature tracks on the same miss frame are pruned independently.

        Track A is instant-activated on frame 1 (sticky: holds a real
        ``tracker_id`` immediately). Track B spawns on frame 2 — after frame 1,
        so it is not instant-activated and, with the default
        ``minimum_consecutive_frames=2``, remains unconfirmed (``tracker_id ==
        -1``) after a single update. Frame 3 is a miss for both: the sticky
        guard must keep A alive while the unconfirmed-track removal step
        prunes B, with no ID cross-contamination between the two.
        """
        tracker = CBIoUTracker()
        box_a = (10.0, 10.0, 50.0, 50.0)
        box_b = (300.0, 300.0, 340.0, 340.0)

        first = tracker.update(_detection(box_a))  # frame 1: A instant-activated
        assert first.tracker_id is not None
        track_a_id = int(first.tracker_id[0])
        assert track_a_id >= 0

        second = tracker.update(_detection(box_b))  # frame 2: B spawns, unconfirmed
        assert second.tracker_id is not None
        assert -1 in second.tracker_id.tolist()

        tracker.update(sv.Detections.empty())  # frame 3: shared miss for A and B

        remaining_ids = {t.tracker_id for t in tracker.tracks}
        assert track_a_id in remaining_ids
        assert -1 not in remaining_ids
        assert len(tracker.tracks) == 1


def test_biou_matrix_reads_cache_without_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """_biou_matrix reads predicted boxes from the per-frame cache, never re-decoding.

    P4-2 decode-once: each tracklet's box is decoded a single time per ``update()``
    and passed to all three association stages via a map keyed by ``id()``. The
    helper must read that map and never call ``get_state_bbox`` itself — doing so
    would reintroduce the per-stage redundant decode the cache exists to remove.
    """
    tracker = CBIoUTracker()
    tracklet = BoTSORTTracklet(np.array([0.0, 0.0, 10.0, 10.0]))

    def _fail() -> np.ndarray:
        raise AssertionError("get_state_bbox must not be called; boxes come from the cache")

    monkeypatch.setattr(tracklet, "get_state_bbox", _fail)
    cached_box = np.array([5.0, 5.0, 15.0, 15.0])
    boxes = np.array([[5.0, 5.0, 15.0, 15.0]])  # identical to the cached box -> IoU 1.0

    result = tracker._biou_matrix([tracklet], boxes, BIoU(buffer_ratio=0.0), {id(tracklet): cached_box})

    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(1.0)
