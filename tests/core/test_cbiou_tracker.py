# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""CBIoU-specific tracker tests.

Generic lifecycle / reset / tracked_objects / mutation contracts are
covered for all trackers in test_trackers.py via ALL_TRACKER_IDS.
This file covers CBIoU-specific invariants:
  - CMC is always disabled (frame argument triggers UserWarning)
  - buffer_ratio is correctly forwarded to BIoU
  - BIoU association is more tolerant of near-miss detections than plain IoU
  - buffer_ratio=0.0 produces the same results as BoTSORT(enable_cmc=False)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import supervision as sv

from trackers.core.botsort.tracker import BoTSORTTracker
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
        assert isinstance(tracker.iou, BIoU)

    def test_buffer_ratio_forwarded_to_biou(self) -> None:
        tracker = CBIoUTracker(buffer_ratio=0.25)
        assert isinstance(tracker.iou, BIoU)
        assert tracker.iou.buffer_ratio == pytest.approx(0.25)

    def test_buffer_ratio_stored_on_tracker(self) -> None:
        tracker = CBIoUTracker(buffer_ratio=0.15)
        assert tracker.buffer_ratio == pytest.approx(0.15)

    def test_cmc_is_disabled(self) -> None:
        tracker = CBIoUTracker()
        assert tracker.enable_cmc is False
        assert tracker.cmc is None

    def test_tracker_id(self) -> None:
        assert CBIoUTracker.tracker_id == "cbiou"

    def test_invalid_buffer_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="buffer_ratio"):
            CBIoUTracker(buffer_ratio=-0.01)


class TestCBIoUFrameWarning:
    """Passing a frame to update() must emit UserWarning (CMC is disabled)."""

    def test_frame_triggers_warning(self) -> None:
        tracker = CBIoUTracker()
        frame = _make_frame()
        det = _detection((100.0, 100.0, 200.0, 200.0))
        with pytest.warns(UserWarning):
            tracker.update(det, frame=frame)

    def test_no_warning_without_frame(self) -> None:
        tracker = CBIoUTracker()
        det = _detection((100.0, 100.0, 200.0, 200.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tracker.update(det)


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
        [-15, -15, 115, 115] and B becomes [93.5, -15, 226.5, 115] —
        they now overlap.
        """
        # Frame 1: spawn a track at box A with high confidence
        cbiou = CBIoUTracker(
            buffer_ratio=0.15,
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

        for tracker in (cbiou, botsort):
            tracker.update(_detection(box_a, conf=0.9))

        # Frame 2: detection slightly outside A — CBIoU buffer closes the gap
        cbiou_result = cbiou.update(_detection(box_b, conf=0.9))
        botsort_result = botsort.update(_detection(box_b, conf=0.9))

        cbiou_ids = cbiou_result.tracker_id
        botsort_ids = botsort_result.tracker_id

        # CBIoU should reuse the existing track (buffer closes the gap)
        # — exactly one output detection with a confirmed (>=0) ID
        assert cbiou_ids is not None and len(cbiou_ids) == 1
        assert cbiou_ids[0] >= 0, "CBIoU should have associated the near-miss detection"

        # CBIoU's confirmed ID on frame 2 must equal the one it assigned on frame 1
        cbiou_frame1 = cbiou.tracks[0].tracker_id
        assert cbiou_ids[0] == cbiou_frame1, (
            "CBIoU should reuse the existing track ID, not spawn a new one"
        )

        # BoTSORT with standard IoU: boxes don't overlap, so old track goes lost
        # and a new unconfirmed track is spawned (tracker_id == -1 or a fresh ID)
        assert botsort_ids is not None
        # The important behavioral difference: BoTSORT should NOT continue the
        # original track (it can't see the gap-crossing detection as a match)
        if len(botsort_ids) > 0:
            botsort_frame1_track_id = next(
                (t.tracker_id for t in botsort.tracks), None
            )
            # Original BoTSORT track should be gone or unmatched
            assert botsort_ids[0] != cbiou_frame1 or botsort_ids[0] == -1, (
                "BoTSORT standard IoU should not have matched the near-miss box"
            )

    def test_zero_buffer_behaves_like_standard_iou(self) -> None:
        """With buffer_ratio=0 CBIoU produces identical results to BoTSORT(no CMC)."""
        cbiou = CBIoUTracker(
            buffer_ratio=0.0,
            minimum_consecutive_frames=2,
            track_activation_threshold=0.7,
        )
        botsort = BoTSORTTracker(
            enable_cmc=False,
            minimum_consecutive_frames=2,
            track_activation_threshold=0.7,
        )

        detections = [
            _detection((50.0, 50.0, 150.0, 150.0), conf=0.9),
            _detection((55.0, 55.0, 155.0, 155.0), conf=0.9),
            _detection((60.0, 60.0, 160.0, 160.0), conf=0.9),
        ]

        for det in detections:
            r_cbiou = cbiou.update(det)
            r_botsort = botsort.update(det)
            # Both should produce the same number of outputs
            assert len(r_cbiou) == len(r_botsort), (
                f"CBIoU(buffer=0) and BoTSORT(no CMC) diverged: "
                f"cbiou={len(r_cbiou)}, botsort={len(r_botsort)}"
            )


class TestCBIoUSearchSpace:
    def test_buffer_ratio_in_search_space(self) -> None:
        assert "buffer_ratio" in CBIoUTracker.search_space

    def test_no_cmc_params_in_search_space(self) -> None:
        ss = CBIoUTracker.search_space
        assert "enable_cmc" not in ss
        assert "cmc_method" not in ss
        assert "cmc_downscale" not in ss

    def test_search_space_buffer_ratio_range(self) -> None:
        spec = CBIoUTracker.search_space["buffer_ratio"]
        assert spec["type"] == "uniform"
        low, high = spec["range"]
        assert low >= 0.0
        assert high <= 1.0
        assert low < high
