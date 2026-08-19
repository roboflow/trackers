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

from trackers.core.botsort.tracker import BoTSORTTracker
from trackers.core.reid.fusion import (
    ReidFusionMethod,
    fuse_adaptive_reid_association,
    fuse_botsort_reid_association,
)


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


class TestFuseBotsortReidAssociation:
    """Unit tests for BoT-SORT IoU/appearance fusion gates."""

    def test_appearance_can_win_when_proximity_passes(self) -> None:
        # Association IoU 0.63 clears the proximity gate (needs IoU > 1 - 0.5 = 0.5),
        # so a strong appearance score can beat it (0.63 → 0.9).
        fused = fuse_botsort_reid_association(
            np.array([[0.63]], dtype=np.float32),
            np.array([[0.8]], dtype=np.float32),
            reid_proximity_threshold=0.5,
            reid_appearance_threshold=0.25,
        )
        assert fused[0, 0] == pytest.approx(0.9)

    def test_low_proximity_ignores_appearance(self) -> None:
        # Association IoU 0.36 fails the proximity gate (needs IoU > 1 - 0.5 = 0.5),
        # so appearance is discarded even though it is strong (0.9). Score stays IoU-only.
        iou_only = np.array([[0.36]], dtype=np.float32)
        fused = fuse_botsort_reid_association(
            iou_only,
            np.array([[0.9]], dtype=np.float32),
            reid_proximity_threshold=0.5,
            reid_appearance_threshold=0.25,
        )
        assert fused[0, 0] == pytest.approx(float(iou_only[0, 0]))

    def test_proximity_uses_standard_iou_not_giou(self) -> None:
        # Association score uses a high GIoU-like value (0.80), but standard IoU is
        # only 0.35 and fails the proximity gate, so appearance must not be used.
        association_iou = np.array([[0.80]], dtype=np.float32)
        fused = fuse_botsort_reid_association(
            association_iou,
            np.array([[0.95]], dtype=np.float32),
            reid_proximity_threshold=0.5,
            reid_appearance_threshold=0.25,
            proximity_iou_similarity=np.array([[0.35]], dtype=np.float32),
        )
        assert fused[0, 0] == pytest.approx(float(association_iou[0, 0]))


class TestBoTSORTTrackerReID:
    """Integration-style tests for BoT-SORT tracker appearance association."""

    def test_rejects_invalid_reid_ema_alpha(self) -> None:
        with pytest.raises(ValueError, match="reid_ema_alpha"):
            BoTSORTTracker(enable_cmc=False, reid_model=_KeyedReIDEncoder(), reid_ema_alpha=1.5)

    @pytest.mark.parametrize("parameter", ["reid_appearance_threshold", "reid_proximity_threshold"])
    @pytest.mark.parametrize("value", [-0.01, 1.01])
    def test_rejects_invalid_association_thresholds(self, parameter: str, value: float) -> None:
        with pytest.raises(ValueError, match=parameter):
            # mypy cannot narrow a dynamic dict against typed kwargs.
            BoTSORTTracker(enable_cmc=False, **{parameter: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize("parameter", ["reid_appearance_threshold", "reid_proximity_threshold"])
    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_accepts_association_threshold_boundaries(self, parameter: str, value: float) -> None:
        # mypy cannot narrow a dynamic dict against typed kwargs.
        tracker = BoTSORTTracker(enable_cmc=False, **{parameter: value})  # type: ignore[arg-type]

        assert getattr(tracker, parameter) == value

    def test_requires_frame_when_reid_enabled(self) -> None:
        tracker = BoTSORTTracker(enable_cmc=False, reid_model=_KeyedReIDEncoder())
        with pytest.raises(ValueError, match="requires frame"):
            tracker.update(_detection((10.0, 10.0, 30.0, 30.0)))

        assert tracker.frame_id == 0
        tracked = tracker.update(_detection((10.0, 10.0, 30.0, 30.0)), frame=_frame())
        np.testing.assert_array_equal(tracked.tracker_id, [0])

    def test_feature_bank_initializes_on_spawn(self) -> None:
        tracker = BoTSORTTracker(enable_cmc=False, reid_model=_KeyedReIDEncoder())
        tracker.update(_detection((10.0, 10.0, 30.0, 30.0)), frame=_frame())
        bank = tracker.tracks[0].feature_bank
        assert bank is not None and bank.feature is not None

    def test_unconfirmed_appearance_match_updates_feature_bank(self) -> None:
        """An appearance-only unconfirmed match blends its detection feature."""
        initial_feature = _norm(np.array([1.0, 0.0, 0.0, 0.0]))
        matched_feature = _norm(np.array([0.8, 0.6, 0.0, 0.0]))
        encoder = _KeyedReIDEncoder({(10, 10): initial_feature})
        tracker = BoTSORTTracker(
            enable_cmc=False,
            instant_first_frame_activation=False,
            reid_model=encoder,
            reid_ema_alpha=0.5,
            minimum_iou_threshold_unconfirmed_assoc=0.85,
        )
        detection = _detection((10.0, 10.0, 30.0, 30.0), conf=0.8)

        tracker.update(detection, frame=_frame(2))
        track = tracker.tracks[0]
        assert track.tracker_id == -1
        bank = track.feature_bank
        assert bank is not None
        feature_after_spawn = bank.feature
        assert feature_after_spawn is not None
        np.testing.assert_allclose(feature_after_spawn, initial_feature)

        # Geometry fused with confidence is only 0.8, below the 0.85 gate;
        # cosine appearance raises the fused similarity to 0.9 and permits the match.
        encoder.table[(10, 10)] = matched_feature
        result = tracker.update(detection, frame=_frame(3))

        assert len(tracker.tracks) == 1
        assert tracker.tracks[0] is track
        assert result.tracker_id is not None
        assert result.tracker_id.tolist() == [track.tracker_id]
        expected_feature = _norm(0.5 * initial_feature + 0.5 * matched_feature)
        feature_after_match = bank.feature
        assert feature_after_match is not None
        np.testing.assert_allclose(feature_after_match, expected_feature, rtol=1e-6, atol=1e-7)

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
            reid_appearance_threshold=0.6,
            reid_proximity_threshold=0.99,
        )
        geo.update(_detection((10.0, 10.0, 30.0, 30.0)), frame=frame)

        reid = BoTSORTTracker(
            enable_cmc=False,
            minimum_iou_threshold_first_assoc=0.01,
            reid_appearance_threshold=0.6,
            reid_proximity_threshold=0.99,
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
        assert bank is not None and bank.feature is not None


class TestFuseAdaptiveReidAssociation:
    """Unit tests for Deep OC-SORT adaptive appearance weighting."""

    def test_dominant_match_outweighs_tied_match(self) -> None:
        # Same top appearance (0.9) and same IoU (0.6) in both cases. When the runner-up
        # is far behind, the adaptive bonus lifts the appearance weight above the base
        # 0.75; when the two candidates tie, no bonus is granted.
        iou = np.array([[0.6, 0.6]], dtype=np.float32)
        dominant = fuse_adaptive_reid_association(
            iou,
            np.array([[0.9, 0.1]], dtype=np.float32),
            reid_appearance_weight=0.75,
            reid_adaptive_weight_cap=0.5,
            reid_proximity_threshold=0.5,
        )
        tied = fuse_adaptive_reid_association(
            iou,
            np.array([[0.9, 0.9]], dtype=np.float32),
            reid_appearance_weight=0.75,
            reid_adaptive_weight_cap=0.5,
            reid_proximity_threshold=0.5,
        )
        # Gap 0.8 is capped at 0.5, halved to a 0.25 bonus: 0.6 + (0.75 + 0.25) * 0.9.
        assert dominant[0, 0] == pytest.approx(1.5)
        # No gap, so the base weight alone applies: 0.6 + 0.75 * 0.9.
        assert tied[0, 0] == pytest.approx(1.275)
        assert dominant[0, 0] > tied[0, 0]

    def test_low_proximity_ignores_appearance(self) -> None:
        # IoU 0.36 fails the proximity gate (needs IoU > 1 - 0.5), so the score stays
        # IoU-only despite a strong appearance match.
        fused = fuse_adaptive_reid_association(
            np.array([[0.36]], dtype=np.float32),
            np.array([[0.9]], dtype=np.float32),
            reid_appearance_weight=0.75,
            reid_adaptive_weight_cap=0.5,
            reid_proximity_threshold=0.5,
        )
        assert fused[0, 0] == pytest.approx(0.36)

    def test_gated_competitor_does_not_inflate_the_bonus(self) -> None:
        # Both detections look identical (0.9), so appearance cannot tell them apart and
        # no bonus is owed. Gating one out on proximity must not change that: the bonus
        # measures appearance discriminativeness, not geometric isolation.
        appearance = np.array([[0.9, 0.9]], dtype=np.float32)
        both_close = fuse_adaptive_reid_association(
            np.array([[0.6, 0.6]], dtype=np.float32),
            appearance,
            reid_appearance_weight=0.75,
            reid_adaptive_weight_cap=0.5,
            reid_proximity_threshold=0.5,
        )
        one_gated = fuse_adaptive_reid_association(
            np.array([[0.6, 0.4]], dtype=np.float32),
            appearance,
            reid_appearance_weight=0.75,
            reid_adaptive_weight_cap=0.5,
            reid_proximity_threshold=0.5,
        )

        # Base weight only in both cases: 0.6 + 0.75 * 0.9.
        assert both_close[0, 0] == pytest.approx(1.275)
        assert one_gated[0, 0] == pytest.approx(1.275)
        # The gated pair keeps its geometry and gains nothing from appearance.
        assert one_gated[0, 1] == pytest.approx(0.4)

    def test_bonus_combines_track_and_detection_gaps(self) -> None:
        # A non-square matrix so the per-track (axis 1) and per-detection (axis 0) gaps
        # differ and both contribute. Track 0 leads its row by 0.6 and column 0 leads by
        # 0.5, each capped at 0.4, giving a bonus of (0.4 + 0.4) / 2 = 0.4.
        fused = fuse_adaptive_reid_association(
            np.full((2, 3), 0.6, dtype=np.float32),
            np.array([[0.9, 0.3, 0.2], [0.4, 0.3, 0.2]], dtype=np.float32),
            reid_appearance_weight=0.5,
            reid_adaptive_weight_cap=0.4,
            reid_proximity_threshold=0.5,
        )
        assert fused[0, 0] == pytest.approx(0.6 + (0.5 + 0.4) * 0.9)
        # Track 1 leads its own row by only 0.1, so its bonus is smaller.
        assert fused[1, 0] == pytest.approx(0.6 + (0.5 + (0.1 + 0.4) / 2) * 0.4)

    def test_negative_appearance_never_lowers_similarity(self) -> None:
        # A negative cosine similarity is clamped to zero so appearance can only ever
        # raise a score, never push a geometrically sound pair below the threshold.
        fused = fuse_adaptive_reid_association(
            np.array([[0.8]], dtype=np.float32),
            np.array([[-0.5]], dtype=np.float32),
            reid_appearance_weight=0.75,
            reid_adaptive_weight_cap=0.5,
            reid_proximity_threshold=0.5,
        )
        assert fused[0, 0] == pytest.approx(0.8)

    def test_single_candidate_gets_no_adaptive_bonus(self) -> None:
        # With one track and one detection there is no runner-up to compare against,
        # so only the base weight applies: 0.7 + 0.75 * 0.8.
        fused = fuse_adaptive_reid_association(
            np.array([[0.7]], dtype=np.float32),
            np.array([[0.8]], dtype=np.float32),
            reid_appearance_weight=0.75,
            reid_adaptive_weight_cap=0.5,
            reid_proximity_threshold=0.5,
        )
        assert fused[0, 0] == pytest.approx(1.3)

    def test_proximity_uses_standard_iou_not_giou(self) -> None:
        # The tracker always passes two different matrices: association similarity is
        # IoU fused with detection confidence, while the gate is raw IoU. A version
        # that ignores the caller's gate matrix would keep the appearance term here.
        association_iou = np.array([[0.80]], dtype=np.float32)
        fused = fuse_adaptive_reid_association(
            association_iou,
            np.array([[0.95]], dtype=np.float32),
            reid_appearance_weight=0.75,
            reid_adaptive_weight_cap=0.5,
            reid_proximity_threshold=0.5,
            proximity_iou_similarity=np.array([[0.35]], dtype=np.float32),
        )
        assert fused[0, 0] == pytest.approx(float(association_iou[0, 0]))


class TestReidFusionSelection:
    """The fusion selector on the tracker."""

    def test_botsort_fusion_is_the_default(self) -> None:
        assert BoTSORTTracker().reid_fusion == "botsort"

    def test_unknown_fusion_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown reid_fusion"):
            BoTSORTTracker(reid_fusion="deepocsort")  # type: ignore[arg-type]

    @pytest.mark.parametrize("parameter", ["reid_appearance_weight", "reid_adaptive_weight_cap"])
    def test_rejects_negative_adaptive_weights(self, parameter: str) -> None:
        with pytest.raises(ValueError, match=parameter):
            # mypy cannot narrow a dynamic dict against typed kwargs.
            BoTSORTTracker(enable_cmc=False, **{parameter: -0.1})  # type: ignore[arg-type]

    def test_adaptive_parameters_reach_fusion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import trackers.core.botsort.tracker as tracker_module

        received: dict[str, float] = {}

        def spy(association, appearance, **kwargs):
            received.update(kwargs)
            return tracker_module.fuse_botsort_reid_association(
                association, appearance, reid_proximity_threshold=0.5, reid_appearance_threshold=0.25
            )

        monkeypatch.setattr(tracker_module, "fuse_adaptive_reid_association", spy)
        tracker = BoTSORTTracker(
            enable_cmc=False,
            reid_model=_KeyedReIDEncoder({(10, 10): _norm(np.array([1.0, 0.0, 0.0, 0.0]))}),
            reid_fusion="adaptive",
            reid_appearance_weight=0.5,
            reid_adaptive_weight_cap=0.4,
            reid_proximity_threshold=0.9,
        )
        tracker.update(_detection((10.0, 10.0, 30.0, 30.0)), frame=_frame(5))
        tracker.update(_detection((10.0, 10.0, 30.0, 30.0)), frame=_frame(6))

        assert received["reid_appearance_weight"] == 0.5
        assert received["reid_adaptive_weight_cap"] == 0.4
        assert received["reid_proximity_threshold"] == 0.9

    def test_adaptive_fusion_changes_association(self) -> None:
        # Non-default weights: at 0.75 with the cap binding the effective weight is exactly
        # 1.0, which makes the fusion symmetric in its inputs and would hide wiring mistakes.
        far = (10.0, 10.0, 30.0, 30.0)
        near = (11.0, 11.0, 31.0, 31.0)
        feature = _norm(np.array([1.0, 0.0, 0.0, 0.0]))

        def run(fusion: ReidFusionMethod) -> np.ndarray:
            tracker = BoTSORTTracker(
                enable_cmc=False,
                reid_model=_KeyedReIDEncoder({(10, 10): feature, (11, 11): feature}),
                reid_fusion=fusion,
                reid_appearance_weight=0.3,
                reid_adaptive_weight_cap=0.1,
                minimum_iou_threshold_first_assoc=1.02,
            )
            tracker.update(_detection(far), frame=_frame(5))
            return tracker.update(_detection(near), frame=_frame(6)).tracker_id

        # The gate sits in the window that separates the two fusions. Taking a minimum
        # of costs can never exceed a similarity of 1, while adding a weighted
        # appearance term reaches 0.74 + 0.3 * 1.0, so only the latter clears 1.02.
        np.testing.assert_array_equal(run("botsort"), [-1])
        np.testing.assert_array_equal(run("adaptive"), [0])
