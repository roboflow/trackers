# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""BoT-SORT ReID association and fusion tests."""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from trackers.core.botsort.fusion import fuse_botsort_reid_association
from trackers.core.botsort.tracker import BoTSORTTracker


def _detection(xyxy: tuple[float, float, float, float], conf: float = 0.9) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([conf], dtype=np.float32),
    )


def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)


def _norm(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    return vec / np.linalg.norm(vec)


class _KeyedReIDEncoder:
    """Deterministic embeddings keyed by detection top-left corner."""

    def __init__(self, table: dict[tuple[int, int], np.ndarray] | None = None) -> None:
        self.table = table or {}
        self.calls = 0

    def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        self.calls += 1
        if len(detections) == 0:
            return np.empty((0, 0), dtype=np.float32)
        rows = []
        for box in detections.xyxy:
            key = (round(float(box[0])), round(float(box[1])))
            rows.append(self.table.get(key, _norm(np.array([float(box[0]), float(box[1]), 1.0, 0.0]))))
        return np.stack(rows)


class TestFuseBotsortReidAssociation:
    def test_appearance_can_win_when_proximity_passes(self) -> None:
        fused = fuse_botsort_reid_association(
            np.array([[0.7]], dtype=np.float32),
            np.array([[0.63]], dtype=np.float32),
            np.array([[0.8]], dtype=np.float32),
            proximity_iou_similarity=np.array([[0.7]], dtype=np.float32),
            proximity_threshold=0.5,
            appearance_threshold=0.25,
        )
        assert fused[0, 0] == pytest.approx(0.9)

    def test_low_proximity_zeros_appearance(self) -> None:
        fused = fuse_botsort_reid_association(
            np.array([[0.4]], dtype=np.float32),
            np.array([[0.36]], dtype=np.float32),
            np.array([[0.9]], dtype=np.float32),
            proximity_iou_similarity=np.array([[0.4]], dtype=np.float32),
            proximity_threshold=0.5,
            appearance_threshold=0.25,
        )
        assert fused[0, 0] == pytest.approx(0.36)

    def test_proximity_uses_standard_iou_not_giou(self) -> None:
        fused = fuse_botsort_reid_association(
            np.array([[0.85]], dtype=np.float32),
            np.array([[0.80]], dtype=np.float32),
            np.array([[0.95]], dtype=np.float32),
            proximity_iou_similarity=np.array([[0.35]], dtype=np.float32),
            proximity_threshold=0.5,
            appearance_threshold=0.25,
        )
        assert fused[0, 0] == pytest.approx(0.80)


class TestBoTSORTTrackerReID:
    def test_rejects_invalid_reid_ema_alpha(self) -> None:
        with pytest.raises(ValueError, match="reid_ema_alpha"):
            BoTSORTTracker(enable_cmc=False, reid_model=_KeyedReIDEncoder(), reid_ema_alpha=1.5)

    def test_requires_frame_when_reid_enabled(self) -> None:
        tracker = BoTSORTTracker(enable_cmc=False, reid_model=_KeyedReIDEncoder())
        with pytest.raises(ValueError, match="requires frame"):
            tracker.update(_detection((10.0, 10.0, 30.0, 30.0)))

    def test_feature_bank_initializes_on_spawn(self) -> None:
        tracker = BoTSORTTracker(enable_cmc=False, reid_model=_KeyedReIDEncoder())
        tracker.update(_detection((10.0, 10.0, 30.0, 30.0)), frame=_frame())
        bank = tracker.tracks[0].feature_bank
        assert bank is not None and bank.is_initialized

    def test_appearance_changes_assignment_vs_geometry_only(self) -> None:
        identity = _norm(np.array([1.0, 0.0, 0.0, 0.0]))
        impostor = _norm(np.array([0.0, 1.0, 0.0, 0.0]))

        class _PhaseEncoder:
            phase = 1

            def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
                rows = []
                for box in detections.xyxy:
                    key = (round(float(box[0])), round(float(box[1])))
                    if self.phase == 1:
                        rows.append(identity)
                    elif key == (10, 10):
                        rows.append(impostor)
                    else:
                        rows.append(identity)
                return np.stack(rows)

        encoder = _PhaseEncoder()
        frame = _frame(1)
        kwargs = dict(
            enable_cmc=False,
            minimum_iou_threshold_first_assoc=0.01,
            appearance_threshold=0.6,
            proximity_threshold=0.99,
        )

        geo = BoTSORTTracker(**kwargs)
        geo.update(_detection((10.0, 10.0, 30.0, 30.0)), frame=frame)

        reid = BoTSORTTracker(**kwargs, reid_model=encoder)
        reid.update(_detection((10.0, 10.0, 30.0, 30.0)), frame=frame)
        track_id = int(reid.tracks[0].tracker_id)

        competitors = sv.Detections(
            xyxy=np.array([[10.0, 10.0, 30.0, 30.0], [14.0, 14.0, 34.0, 34.0]], dtype=np.float32),
            confidence=np.array([0.9, 0.9], dtype=np.float32),
        )
        encoder.phase = 2
        geo_out = geo.update(competitors, frame=frame)
        reid_out = reid.update(competitors, frame=frame)

        def _matched_xy(out: sv.Detections, tid: int) -> tuple[float, float]:
            assert out.tracker_id is not None
            box = out.xyxy[out.tracker_id == tid][0]
            return float(box[0]), float(box[1])

        assert _matched_xy(geo_out, int(geo.tracks[0].tracker_id)) == (10.0, 10.0)
        assert _matched_xy(reid_out, track_id) == (14.0, 14.0)

    def test_low_confidence_stage_does_not_update_feature_bank(self) -> None:
        model = _KeyedReIDEncoder({(10, 10): _norm(np.array([1.0, 0.0, 0.0, 0.0]))})
        tracker = BoTSORTTracker(
            enable_cmc=False,
            reid_model=model,
            high_conf_det_threshold=0.8,
            minimum_iou_threshold_second_assoc=0.01,
        )
        tracker.update(_detection((10.0, 10.0, 30.0, 30.0)), frame=_frame(4))
        bank = tracker.tracks[0].feature_bank
        assert bank is not None
        before = bank.feature
        assert before is not None

        calls_after_high = model.calls
        tracker.update(_detection((12.0, 12.0, 32.0, 32.0), conf=0.5), frame=_frame(4))
        assert model.calls == calls_after_high
        after = bank.feature
        assert after is not None
        np.testing.assert_allclose(before, after)
