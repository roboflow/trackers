# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import argparse

import numpy as np
import pytest
import supervision as sv

from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.scripts.track import _apply_reid_tracker_params, add_track_subparser


def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)


def _det(xyxy: tuple[float, float, float, float], conf: float = 0.9) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([conf], dtype=np.float32),
    )


def _norm(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    return vec / np.linalg.norm(vec)


class _KeyedReIDEncoder:
    """Return fixed embeddings keyed by detection top-left corner."""

    def __init__(self, table: dict[tuple[int, int], np.ndarray]) -> None:
        self.table = table
        self.calls = 0

    def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        self.calls += 1
        if len(detections) == 0:
            return np.empty((0, 0), dtype=np.float32)
        rows = []
        for box in detections.xyxy:
            key = (int(round(float(box[0]))), int(round(float(box[1]))))
            rows.append(self.table.get(key, _norm(np.array([float(box[0]), float(box[1]), 1.0, 0.0]))))
        return np.stack(rows)


class TestBoTSORTReIDE2E:
    def test_requires_frame_when_reid_enabled(self) -> None:
        tracker = BoTSORTTracker(enable_cmc=False, reid_model=_KeyedReIDEncoder({}))
        with pytest.raises(ValueError, match="requires frame"):
            tracker.update(_det((10.0, 10.0, 30.0, 30.0)))

    def test_feature_bank_initializes_on_spawn(self) -> None:
        model = _KeyedReIDEncoder({})
        tracker = BoTSORTTracker(enable_cmc=False, reid_model=model)
        frame = _frame()
        tracker.update(_det((10.0, 10.0, 30.0, 30.0)), frame=frame)
        bank = tracker.tracks[0].feature_bank
        assert bank is not None and bank.is_initialized

    def test_appearance_changes_assignment_vs_geometry_only(self) -> None:
        """Near-equal IoU: ReID keeps the appearance match; geometry-only takes max IoU."""
        identity = _norm(np.array([1.0, 0.0, 0.0, 0.0]))
        impostor = _norm(np.array([0.0, 1.0, 0.0, 0.0]))

        class _PhaseEncoder:
            phase = 1

            def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
                rows = []
                for box in detections.xyxy:
                    x1 = int(round(float(box[0])))
                    y1 = int(round(float(box[1])))
                    if self.phase == 1:
                        rows.append(identity)
                    elif (x1, y1) == (10, 10):
                        rows.append(impostor)
                    elif (x1, y1) == (14, 14):
                        rows.append(identity)
                return np.stack(rows)

        encoder = _PhaseEncoder()
        frame = _frame(1)
        shared = dict(
            enable_cmc=False,
            minimum_iou_threshold_first_assoc=0.01,
            appearance_threshold=0.6,
            proximity_threshold=0.99,
        )

        geo = BoTSORTTracker(**shared)
        geo.update(_det((10.0, 10.0, 30.0, 30.0)), frame=frame)

        reid = BoTSORTTracker(**shared, reid_model=encoder)
        reid.update(_det((10.0, 10.0, 30.0, 30.0)), frame=frame)
        track_id = int(reid.tracks[0].tracker_id)

        competitors = sv.Detections(
            xyxy=np.array(
                [
                    [10.0, 10.0, 30.0, 30.0],
                    [14.0, 14.0, 34.0, 34.0],
                ],
                dtype=np.float32,
            ),
            confidence=np.array([0.9, 0.9], dtype=np.float32),
        )

        encoder.phase = 2
        geo_out = geo.update(competitors, frame=frame)
        reid_out = reid.update(competitors, frame=frame)

        def _matched_top_left(out: sv.Detections, tid: int) -> tuple[float, float]:
            mask = out.tracker_id == tid
            assert mask.any(), f"track {tid} not found in output"
            box = out.xyxy[mask][0]
            return float(box[0]), float(box[1])

        assert _matched_top_left(geo_out, int(geo.tracks[0].tracker_id)) == (10.0, 10.0)
        assert _matched_top_left(reid_out, track_id) == (14.0, 14.0)

    def test_lost_track_recovers_with_appearance(self) -> None:
        identity = _norm(np.array([1.0, 0.0, 0.0, 0.0]))
        table = {(10, 10): identity, (12, 12): identity}
        frame = _frame(2)
        tracker = BoTSORTTracker(
            enable_cmc=False,
            reid_model=_KeyedReIDEncoder(table),
            minimum_iou_threshold_first_assoc=0.01,
            proximity_threshold=0.99,
        )
        first = tracker.update(_det((10.0, 10.0, 30.0, 30.0)), frame=frame)
        track_id = int(first.tracker_id[0])

        tracker.update(sv.Detections.empty(), frame=frame)
        assert any(t.tracker_id == track_id for t in tracker.tracks)

        recovered = tracker.update(_det((12.0, 12.0, 32.0, 32.0)), frame=frame)
        assert track_id in recovered.tracker_id.tolist()

    def test_no_reid_parity_unchanged(self) -> None:
        boxes = sv.Detections(
            xyxy=np.array([[10.0, 10.0, 40.0, 40.0], [100.0, 100.0, 130.0, 130.0]], dtype=np.float32),
            confidence=np.array([0.9, 0.9], dtype=np.float32),
        )
        frame = _frame(3)
        baseline = BoTSORTTracker(enable_cmc=False)
        reid_off = BoTSORTTracker(enable_cmc=False, reid_model=None)
        np.testing.assert_array_equal(
            baseline.update(boxes, frame=frame).tracker_id,
            reid_off.update(boxes, frame=frame).tracker_id,
        )

    def test_low_confidence_stage_does_not_update_feature_bank(self) -> None:
        model = _KeyedReIDEncoder({(10, 10): _norm(np.array([1.0, 0.0, 0.0, 0.0]))})
        tracker = BoTSORTTracker(
            enable_cmc=False,
            reid_model=model,
            high_conf_det_threshold=0.8,
            minimum_iou_threshold_second_assoc=0.01,
        )
        frame = _frame(4)
        tracker.update(_det((10.0, 10.0, 30.0, 30.0)), frame=frame)
        bank = tracker.tracks[0].feature_bank
        assert bank is not None
        before = bank.feature
        assert before is not None

        calls_after_high = model.calls
        tracker.update(_det((12.0, 12.0, 32.0, 32.0), conf=0.5), frame=frame)
        assert model.calls == calls_after_high
        after = bank.feature
        assert after is not None
        np.testing.assert_allclose(before, after)

    def test_reset_clears_tracks(self) -> None:
        tracker = BoTSORTTracker(enable_cmc=False, reid_model=_KeyedReIDEncoder({}))
        tracker.update(_det((10.0, 10.0, 30.0, 30.0)), frame=_frame(5))
        tracker.reset()
        assert tracker.tracks == []


class TestTrackCLIReID:
    def test_help_lists_reid_flags(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_track_subparser(subparsers)
        help_text = subparsers.choices["track"].format_help()
        assert "--tracker.reid.enable" in help_text
        assert "--tracker.reid.model" in help_text
        assert "--tracker.reid.architecture" in help_text

    def test_apply_reid_rejects_non_botsort_tracker(self) -> None:
        args = argparse.Namespace(
            tracker_reid_enable=True,
            tracker_reid_model=None,
            tracker_reid_device="cpu",
            tracker_reid_architecture=None,
        )
        _, error = _apply_reid_tracker_params("bytetrack", args, {})
        assert error is not None and "botsort" in error
