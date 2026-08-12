# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for evaluation result serialization and table formatting."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trackers.eval import (
    AggregationIncompatibleError,
    BenchmarkResult,
    HOTAMetrics,
    SequenceResult,
    aggregate_hota_metrics,
    compute_hota_metrics,
    evaluate_multicamera_scenes,
)
from trackers.eval.results import _reject_nonfinite_json_constant

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "multicamera"


def _hota_metrics(**overrides: float | int | None) -> HOTAMetrics:
    base: dict[str, float | int | None] = {
        "HOTA": 0.5,
        "DetA": 0.4,
        "AssA": 0.6,
        "DetRe": 0.5,
        "DetPr": 0.5,
        "AssRe": 0.5,
        "AssPr": 0.5,
        "LocA": 0.9,
        "OWTA": 0.5,
        "HOTA_TP": 10,
        "HOTA_FN": 2,
        "HOTA_FP": 2,
    }
    base.update(overrides)
    return HOTAMetrics(**base)  # type: ignore[arg-type]


class TestBenchmarkResultAggregation:
    """Aggregation discriminator serialization and table labels."""

    def test_aggregation_round_trip(self) -> None:
        result = BenchmarkResult(
            sequences={
                "s1": SequenceResult(sequence="s1", HOTA=_hota_metrics()),
            },
            aggregate=SequenceResult(sequence="SCENE_MEAN", HOTA=_hota_metrics()),
            aggregation="scene_mean",
        )
        restored = BenchmarkResult.from_dict(json.loads(result.json(), parse_constant=_reject_nonfinite_json_constant))
        assert restored.aggregation == "scene_mean"
        assert restored.aggregate.sequence == "SCENE_MEAN"

    def test_legacy_json_defaults_to_tp_weighted(self) -> None:
        """Legacy payloads retain defaults and have no benchmark coverage."""
        legacy = {
            "sequences": {
                "s1": {
                    "sequence": "s1",
                    "HOTA": _hota_metrics().to_dict(),
                }
            },
            "aggregate": {
                "sequence": "COMBINED",
                "HOTA": _hota_metrics().to_dict(),
            },
        }
        restored = BenchmarkResult.from_dict(legacy)
        assert restored.aggregation == "tp_weighted"
        assert restored.coverage is None

    def test_table_uses_aggregate_sequence_label(self) -> None:
        result = BenchmarkResult(
            sequences={
                "scene_a": SequenceResult(sequence="scene_a", HOTA=_hota_metrics(HOTA=0.5)),
            },
            aggregate=SequenceResult(sequence="SCENE_MEAN", HOTA=_hota_metrics(HOTA=0.5)),
            aggregation="scene_mean",
        )
        table = result.table(columns=["HOTA"])
        assert "SCENE_MEAN" in table
        assert "COMBINED" not in table

    def test_tp_weighted_hota_rejects_scene_mean(self) -> None:
        result = evaluate_multicamera_scenes(
            gt_dir=FIXTURE_DIR / "gt",
            tracker_dir=FIXTURE_DIR / "pred",
            scene_camera_map=FIXTURE_DIR / "scene_camera_map.json",
            allow_partial=True,
        )
        with pytest.raises(AggregationIncompatibleError, match="scene_mean"):
            result.tp_weighted_hota()

        # Reloaded scene-mean payloads keep aggregation='scene_mean'.
        reloaded = BenchmarkResult.from_dict(json.loads(result.json(), parse_constant=_reject_nonfinite_json_constant))
        with pytest.raises(AggregationIncompatibleError, match="scene_mean"):
            reloaded.tp_weighted_hota()

    def test_tp_weighted_hota_rejects_missing_arrays(self) -> None:
        """Typed error when tp_weighted results lack per-alpha arrays."""
        result = BenchmarkResult(
            sequences={
                "s1": SequenceResult(sequence="s1", HOTA=_hota_metrics()),
            },
            aggregate=SequenceResult(sequence="COMBINED", HOTA=_hota_metrics()),
            aggregation="tp_weighted",
        )
        with pytest.raises(AggregationIncompatibleError, match="HOTA_TP_array"):
            result.tp_weighted_hota()

    def test_mot_tp_weighted_regression(self) -> None:
        """MOT path's TP-weighted aggregate behaviour remains unchanged."""
        high_quality = compute_hota_metrics(
            gt_ids=[np.array([0, 1, 2, 3])],
            tracker_ids=[np.array([10, 20, 30, 40])],
            similarity_scores=[np.diag([0.9, 0.9, 0.9, 0.9])],
        )
        low_quality = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([10])],
            similarity_scores=[np.array([[0.3]])],
        )
        agg = aggregate_hota_metrics([high_quality, low_quality])
        assert agg["HOTA"] > low_quality["HOTA"]
        arithmetic = 0.5 * (high_quality["HOTA"] + low_quality["HOTA"])
        assert agg["HOTA"] != pytest.approx(arithmetic, rel=1e-3)

        result = BenchmarkResult(
            sequences={
                "high": SequenceResult(sequence="high", HOTA=HOTAMetrics.from_dict(high_quality)),
                "low": SequenceResult(sequence="low", HOTA=HOTAMetrics.from_dict(low_quality)),
            },
            aggregate=SequenceResult(sequence="COMBINED", HOTA=HOTAMetrics.from_dict(agg)),
            aggregation="tp_weighted",
        )
        via_public = result.tp_weighted_hota()
        assert via_public.HOTA == pytest.approx(agg["HOTA"], rel=1e-4, abs=1e-4)

    def test_tp_weighted_hota_does_not_import_evaluator(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Result reaggregation does not depend on the private evaluator module."""
        metrics = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([0])],
            similarity_scores=[np.array([[1.0]])],
        )
        result = BenchmarkResult(
            sequences={
                "s1": SequenceResult(sequence="s1", HOTA=HOTAMetrics.from_dict(metrics)),
            },
            aggregate=SequenceResult(sequence="COMBINED", HOTA=HOTAMetrics.from_dict(metrics)),
        )
        original_import = builtins.__import__

        def rejecting_import(name: str, *args: Any, **kwargs: Any) -> object:
            if name == "trackers.eval.evaluate":
                raise AssertionError("BenchmarkResult must not import trackers.eval.evaluate")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", rejecting_import)

        aggregated = result.tp_weighted_hota()

        assert aggregated.HOTA == pytest.approx(1.0)


class TestBenchmarkResultLoad:
    """Strict JSON persistence behavior."""

    @pytest.mark.parametrize(
        "constant",
        [
            pytest.param("NaN", id="nan"),
            pytest.param("Infinity", id="positive-infinity"),
            pytest.param("-Infinity", id="negative-infinity"),
        ],
    )
    def test_nonfinite_json_constant_rejected(
        self,
        tmp_path: Path,
        constant: str,
    ) -> None:
        """Non-standard JSON numeric constants are rejected while loading."""
        path = tmp_path / "result.json"
        path.write_text(
            '{"sequences": {}, "aggregate": {"sequence": "COMBINED"}, '
            '"aggregation": "tp_weighted", "extra": ' + constant + "}"
        )

        with pytest.raises(ValueError, match="Non-finite JSON constant"):
            BenchmarkResult.load(path)

    def test_valid_legacy_payload_still_loads(self, tmp_path: Path) -> None:
        """Strict parsing preserves the backward-compatible legacy defaults."""
        path = tmp_path / "legacy.json"
        path.write_text('{"sequences": {}, "aggregate": {"sequence": "COMBINED"}}')

        restored = BenchmarkResult.load(path)

        assert restored.aggregation == "tp_weighted"
        assert restored.coverage is None

    @pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "inf", "+inf", "-inf"])
    def test_nonfinite_numeric_strings_rejected(self, tmp_path: Path, value: str) -> None:
        payload: dict[str, Any] = {
            "sequences": {},
            "aggregate": {"sequence": "COMBINED", "HOTA": _hota_metrics().to_dict()},
        }
        payload["aggregate"]["HOTA"]["HOTA"] = value
        path = tmp_path / "string-constant.json"
        path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="Non-finite"):
            BenchmarkResult.load(path)

    def test_unknown_aggregation_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "unknown-aggregation.json"
        path.write_text(
            json.dumps(
                {
                    "sequences": {},
                    "aggregate": {"sequence": "COMBINED"},
                    "aggregation": "mystery",
                }
            )
        )
        with pytest.raises(ValueError, match="aggregation"):
            BenchmarkResult.load(path)

    def test_overflowing_json_number_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "overflow.json"
        path.write_text(
            '{"sequences": {}, "aggregate": {"sequence": "COMBINED", '
            '"HOTA": {"HOTA": 1e9999}}, "aggregation": "tp_weighted"}'
        )

        with pytest.raises(ValueError, match="Non-finite"):
            BenchmarkResult.load(path)

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param('{"sequences": {},}', id="trailing-comma"),
            pytest.param('{"sequences": {}', id="truncated-object"),
            pytest.param('{"sequences": {}}{"aggregate": {}}', id="concatenated-documents"),
        ],
    )
    def test_malformed_near_valid_json_rejected(self, tmp_path: Path, payload: str) -> None:
        path = tmp_path / "malformed.json"
        path.write_text(payload)

        with pytest.raises(json.JSONDecodeError):
            BenchmarkResult.load(path)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            pytest.param("complete", "false", id="non-boolean-complete"),
            pytest.param("evaluated_scenes", [], id="sequence-coverage-mismatch"),
        ],
    )
    def test_inconsistent_coverage_rejected(self, field: str, value: object) -> None:
        payload: dict[str, Any] = {
            "sequences": {"scene_061": {"sequence": "scene_061"}},
            "aggregate": {"sequence": "SCENE_MEAN"},
            "aggregation": "scene_mean",
            "coverage": {
                "benchmark": "AI City Challenge 2024",
                "split": "test",
                "protocol": "MTMC_Tracking_2024",
                "file_format": "aicity-2024",
                "canonical_scene_camera_map_sha256": "a" * 64,
                "scene_camera_map_sha256": "a" * 64,
                "expected_scenes": ["scene_061"],
                "evaluated_scenes": ["scene_061"],
                "missing_scenes": [],
                "complete": True,
            },
        }
        payload["coverage"][field] = value

        with pytest.raises(ValueError, match=r"coverage|complete|evaluated"):
            BenchmarkResult.from_dict(payload)


class TestAggregateHotaValidation:
    """Public HOTA aggregation reports incompatible payloads consistently."""

    @pytest.mark.parametrize(
        "missing_field",
        [
            pytest.param("HOTA_TP_array", id="true-positive-array"),
            pytest.param("HOTA_FN_array", id="false-negative-array"),
            pytest.param("HOTA_FP_array", id="false-positive-array"),
            pytest.param("AssA_array", id="association-accuracy-array"),
            pytest.param("AssRe_array", id="association-recall-array"),
            pytest.param("AssPr_array", id="association-precision-array"),
            pytest.param("LocA_array", id="localization-array"),
        ],
    )
    def test_missing_required_array_raises_typed_error(self, missing_field: str) -> None:
        """Every required per-alpha array has the same typed failure contract."""
        complete = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([0])],
            similarity_scores=[np.array([[1.0]])],
        )
        del complete[missing_field]

        with pytest.raises(AggregationIncompatibleError, match=missing_field):
            aggregate_hota_metrics([complete])

    @pytest.mark.parametrize(
        "invalid_array",
        [
            pytest.param(None, id="none"),
            pytest.param([1.0], id="wrong-shape"),
            pytest.param([float("nan")] * 19, id="non-finite"),
        ],
    )
    def test_invalid_required_array_raises_typed_error(self, invalid_array: object) -> None:
        complete = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([0])],
            similarity_scores=[np.array([[1.0]])],
        )
        complete["AssA_array"] = invalid_array

        with pytest.raises(AggregationIncompatibleError, match="AssA_array"):
            aggregate_hota_metrics([complete])

    def test_json_list_arrays_aggregate_after_validation(self) -> None:
        complete = compute_hota_metrics(
            gt_ids=[np.array([0])],
            tracker_ids=[np.array([0])],
            similarity_scores=[np.array([[1.0]])],
        )
        for field_name in (
            "HOTA_TP_array",
            "HOTA_FN_array",
            "HOTA_FP_array",
            "AssA_array",
            "AssRe_array",
            "AssPr_array",
            "LocA_array",
        ):
            complete[field_name] = complete[field_name].tolist()

        aggregated = aggregate_hota_metrics([complete])

        assert aggregated["HOTA"] == pytest.approx(1.0)


class TestSceneMeanStrictJson:
    def test_scene_mean_json_is_strict_and_null_for_undefined(self) -> None:
        result = evaluate_multicamera_scenes(
            gt_dir=FIXTURE_DIR / "gt",
            tracker_dir=FIXTURE_DIR / "pred",
            scene_camera_map=FIXTURE_DIR / "scene_camera_map.json",
            allow_partial=True,
        )
        payload = result.json()
        assert "NaN" not in payload
        assert "Infinity" not in payload
        data = json.loads(payload, parse_constant=_reject_nonfinite_json_constant)
        aggregate_hota = data["aggregate"]["HOTA"]
        for field in ("DetRe", "DetPr", "AssRe", "AssPr", "OWTA", "HOTA_TP", "HOTA_FN", "HOTA_FP"):
            assert aggregate_hota[field] is None, field
        for field in ("HOTA", "DetA", "AssA", "LocA"):
            assert isinstance(aggregate_hota[field], float), field

        restored = BenchmarkResult.from_dict(data)
        assert restored.aggregate.HOTA is not None
        assert restored.aggregate.HOTA.DetRe is None
        assert restored.aggregate.HOTA.HOTA_TP is None
