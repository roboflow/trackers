# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for evaluation result serialization and table formatting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trackers.eval import (
    BenchmarkResult,
    evaluate_multicamera_scenes,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "multicamera"


@pytest.fixture(scope="module")
def scene_mean_result() -> BenchmarkResult:
    return evaluate_multicamera_scenes(
        gt_dir=FIXTURE_DIR / "gt",
        tracker_dir=FIXTURE_DIR / "pred",
        scene_camera_map=FIXTURE_DIR / "scene_camera_map.json",
    )


class TestBenchmarkResultAggregation:
    """Aggregation discriminator serialization and table labels."""

    def test_aggregation_round_trip(self, scene_mean_result: BenchmarkResult) -> None:
        restored = BenchmarkResult.from_dict(json.loads(scene_mean_result.json()))
        assert restored.aggregation == "scene_mean"
        assert restored.aggregate.sequence == "SCENE_MEAN"

    def test_legacy_json_defaults_to_tp_weighted(self) -> None:
        legacy = {"sequences": {}, "aggregate": {"sequence": "COMBINED"}}
        restored = BenchmarkResult.from_dict(legacy)
        assert restored.aggregation == "tp_weighted"

    def test_table_uses_aggregate_sequence_label(self, scene_mean_result: BenchmarkResult) -> None:
        table = scene_mean_result.table(columns=["HOTA"])
        assert "SCENE_MEAN" in table
        assert "COMBINED" not in table


class TestSceneMeanJson:
    def test_undefined_aggregate_fields_serialize_as_null(self, scene_mean_result: BenchmarkResult) -> None:
        aggregate_hota = json.loads(scene_mean_result.json())["aggregate"]["HOTA"]

        undefined = ("DetRe", "DetPr", "AssRe", "AssPr", "OWTA", "HOTA_TP", "HOTA_FN", "HOTA_FP")
        assert {field: aggregate_hota[field] for field in undefined} == dict.fromkeys(undefined)
