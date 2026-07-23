# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""BoT-SORT ReID association and fusion tests."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import supervision as sv

from trackers.core.botsort.fusion import fuse_botsort_reid_association
from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.reid.appearance import (
    appearance_similarity,
    extract_detection_embeddings,
)
from trackers.core.reid.feature_bank import FeatureBank


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


def test_botsort_import_does_not_load_reid_model_stack() -> None:
    """Importing BoT-SORT must not pull the heavy ``reid`` package (torch etc.)."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import trackers.core.botsort.tracker; "
                "assert 'reid' not in sys.modules; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


class TestFeatureBank:
    """Unit tests for ``FeatureBank`` L2 + EMA behavior."""

    def test_first_update_normalizes_embedding(self) -> None:
        # BoT-SORT STrack.update_features: L2-normalize before storage.
        bank = FeatureBank(alpha=0.9)
        bank.update(np.array([3.0, 4.0], dtype=np.float32))
        feature = bank.feature
        assert feature is not None
        np.testing.assert_allclose(feature, [0.6, 0.8], atol=1e-6)

    def test_blends_on_unit_sphere(self) -> None:
        # BoT-SORT: EMA on unit vectors, then L2-normalize the blend again.
        bank = FeatureBank(alpha=0.75)
        bank.update(np.array([1.0, 0.0], dtype=np.float32))
        bank.update(np.array([0.0, 1.0], dtype=np.float32))
        feature = bank.feature
        assert feature is not None
        # 0.75*[1,0] + 0.25*[0,1] = [0.75, 0.25], then / ||.||
        expected = np.array([0.75, 0.25], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(feature, expected, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(feature), 1.0, atol=1e-6)

    def test_zero_embedding_is_accepted(self) -> None:
        bank = FeatureBank()
        bank.update(np.zeros(8, dtype=np.float32))
        feature = bank.feature
        assert feature is not None
        np.testing.assert_allclose(feature, 0.0)

    def test_non_finite_embedding_raises(self) -> None:
        bank = FeatureBank()
        with pytest.raises(ValueError, match="finite"):
            bank.update(np.array([1.0, np.nan], dtype=np.float32))
        assert not bank.is_initialized

    def test_shape_change_raises(self) -> None:
        bank = FeatureBank()
        bank.update(np.array([1.0, 0.0], dtype=np.float32))
        before = bank.feature
        assert before is not None
        with pytest.raises(ValueError, match="shape"):
            bank.update(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        after = bank.feature
        assert after is not None
        np.testing.assert_allclose(after, before)


class TestAppearanceSimilarity:
    """Unit tests for cosine ``appearance_similarity`` and embedding extraction."""

    def test_identical_vectors_are_one(self) -> None:
        similarity = appearance_similarity(
            [np.array([1.0, 0.0], dtype=np.float32)],
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(similarity, [[1.0]], atol=1e-6)

    def test_orthogonal_vectors_are_zero(self) -> None:
        similarity = appearance_similarity(
            [np.array([1.0, 0.0], dtype=np.float32)],
            np.array([[0.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(similarity, [[0.0]], atol=1e-6)

    def test_normalizes_both_sides(self) -> None:
        similarity = appearance_similarity(
            [np.array([3.0, 4.0], dtype=np.float32)],
            np.array([[6.0, 8.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(similarity, [[1.0]], atol=1e-6)

    def test_none_track_yields_zero_row(self) -> None:
        similarity = appearance_similarity(
            [None, np.array([1.0, 0.0], dtype=np.float32)],
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(similarity, [[0.0], [1.0]], atol=1e-6)

    def test_empty_inputs_return_empty_matrix(self) -> None:
        assert appearance_similarity([], np.empty((0, 4), dtype=np.float32)).shape == (0, 0)
        assert appearance_similarity(
            [np.array([1.0, 0.0], dtype=np.float32)],
            np.empty((0, 2), dtype=np.float32),
        ).shape == (1, 0)

    def test_non_finite_detection_rows_raise(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            appearance_similarity(
                [np.array([1.0, 0.0], dtype=np.float32)],
                np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32),
            )

    def test_incompatible_track_dimensions_raise(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            appearance_similarity(
                [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
                np.array([[1.0, 0.0]], dtype=np.float32),
            )

    def test_extract_detection_embeddings_requires_one_row_per_box(self) -> None:
        # Encoder must return embeddings.shape[0] == len(boxes).
        class _WrongLengthEncoder:
            def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
                return np.empty((0, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="rows"):
            extract_detection_embeddings(
                _WrongLengthEncoder(),
                _frame(),
                np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
            )


class TestFuseBotsortReidAssociation:
    """Unit tests for BoT-SORT IoU/appearance fusion gates."""

    def test_appearance_can_win_when_proximity_passes(self) -> None:
        # Association IoU 0.63 clears the proximity gate (needs IoU > 1 - 0.5 = 0.5),
        # so a strong appearance score can beat it (0.63 → 0.9).
        fused = fuse_botsort_reid_association(
            np.array([[0.63]], dtype=np.float32),
            np.array([[0.8]], dtype=np.float32),
            proximity_threshold=0.5,
            appearance_threshold=0.25,
        )
        assert fused[0, 0] == pytest.approx(0.9)

    def test_low_proximity_ignores_appearance(self) -> None:
        # Association IoU 0.36 fails the proximity gate (needs IoU > 1 - 0.5 = 0.5),
        # so appearance is discarded even though it is strong (0.9). Score stays IoU-only.
        iou_only = np.array([[0.36]], dtype=np.float32)
        fused = fuse_botsort_reid_association(
            iou_only,
            np.array([[0.9]], dtype=np.float32),
            proximity_threshold=0.5,
            appearance_threshold=0.25,
        )
        assert fused[0, 0] == pytest.approx(float(iou_only[0, 0]))

    def test_proximity_uses_standard_iou_not_giou(self) -> None:
        # Association score uses a high GIoU-like value (0.80), but standard IoU is
        # only 0.35 and fails the proximity gate, so appearance must not be used.
        association_iou = np.array([[0.80]], dtype=np.float32)
        fused = fuse_botsort_reid_association(
            association_iou,
            np.array([[0.95]], dtype=np.float32),
            proximity_threshold=0.5,
            appearance_threshold=0.25,
            proximity_iou_similarity=np.array([[0.35]], dtype=np.float32),
        )
        assert fused[0, 0] == pytest.approx(float(association_iou[0, 0]))


class TestBoTSORTTrackerReID:
    """Integration-style tests for BoT-SORT tracker appearance association."""

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
        geo = BoTSORTTracker(
            enable_cmc=False,
            minimum_iou_threshold_first_assoc=0.01,
            appearance_threshold=0.6,
            proximity_threshold=0.99,
        )
        geo.update(_detection((10.0, 10.0, 30.0, 30.0)), frame=frame)

        reid = BoTSORTTracker(
            enable_cmc=False,
            minimum_iou_threshold_first_assoc=0.01,
            appearance_threshold=0.6,
            proximity_threshold=0.99,
            reid_model=encoder,
        )
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

    @pytest.mark.integration
    def test_real_reid_model_runs_over_frames(self) -> None:
        """Smoke the ``trackers`` → ``reid`` boundary with a real encoder."""
        import reid

        reid_model = reid.ReIDModel.from_pretrained(architecture="osnet_x0_25", device="cpu")
        tracker = BoTSORTTracker(enable_cmc=False, reid_model=reid_model)

        rng = np.random.default_rng(0)
        box = np.array([30.0, 30.0, 70.0, 90.0], dtype=np.float32)
        for _ in range(3):
            frame = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
            detections = sv.Detections(
                xyxy=box[None, :].copy(),
                confidence=np.array([0.9], dtype=np.float32),
            )
            result = tracker.update(detections, frame=frame)
            assert result.tracker_id is not None
            box = box + np.array([2.0, 1.0, 2.0, 1.0], dtype=np.float32)

        assert len(tracker.tracks) == 1
        bank = tracker.tracks[0].feature_bank
        assert bank is not None and bank.is_initialized
