# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Integration tests comparing our metrics against TrackEval on real data.

These tests download SportsMOT and DanceTrack test datasets and verify that our
benchmark evaluation produces identical results to TrackEval.
Numerical parity is the key requirement.

Tests use auto-detection - no explicit format/benchmark/split/tracker_name
parameters are passed, verifying the smart detection logic works correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from trackers.eval import evaluate_benchmark

if TYPE_CHECKING:
    pass


@pytest.mark.integration
def test_evaluate_benchmark_sportsmot_flat(
    sportsmot_flat_data: tuple[Path, dict[str, Any]],
) -> None:
    """Test evaluate_benchmark with SportsMOT flat format (auto-detected)."""
    data_path, expected_results = sportsmot_flat_data

    # Auto-detection should detect flat format from *.txt files
    result = evaluate_benchmark(
        gt_dir=data_path / "gt",
        tracker_dir=data_path / "trackers",
        seqmap=data_path / "seqmap.txt",
        metrics=["CLEAR", "HOTA", "Identity"],
    )

    # Verify all sequences are evaluated
    assert len(result.sequences) == len(expected_results), (
        f"Sequence count mismatch: got {len(result.sequences)}, "
        f"expected {len(expected_results)}"
    )

    # Verify each sequence matches expected results
    for seq_name, seq_result in result.sequences.items():
        assert seq_name in expected_results, f"Unexpected sequence: {seq_name}"
        expected_clear = expected_results[seq_name]["CLEAR"]
        _verify_clear_metrics(seq_result.CLEAR, expected_clear, f"sportsmot/{seq_name}")
        expected_hota = expected_results[seq_name]["HOTA"]
        _verify_hota_metrics(seq_result.HOTA, expected_hota, f"sportsmot/{seq_name}")
        expected_identity = expected_results[seq_name]["Identity"]
        _verify_identity_metrics(
            seq_result.Identity, expected_identity, f"sportsmot/{seq_name}"
        )

    # Verify aggregate metrics are computed correctly
    assert result.aggregate.sequence == "COMBINED"


@pytest.mark.integration
def test_evaluate_benchmark_sportsmot_mot17(
    sportsmot_mot17_data: tuple[Path, dict[str, Any]],
) -> None:
    """Test evaluate_benchmark with SportsMOT MOT17 format (auto-detected)."""
    data_path, expected_results = sportsmot_mot17_data

    # Auto-detection should detect MOT17 format, benchmark, split, and tracker
    result = evaluate_benchmark(
        gt_dir=data_path / "gt",
        tracker_dir=data_path / "trackers",
        metrics=["CLEAR", "HOTA", "Identity"],
    )

    # Verify all sequences are evaluated
    assert len(result.sequences) == len(expected_results), (
        f"Sequence count mismatch: got {len(result.sequences)}, "
        f"expected {len(expected_results)}"
    )

    # Verify each sequence matches expected results
    for seq_name, seq_result in result.sequences.items():
        assert seq_name in expected_results, f"Unexpected sequence: {seq_name}"
        expected_clear = expected_results[seq_name]["CLEAR"]
        _verify_clear_metrics(
            seq_result.CLEAR, expected_clear, f"sportsmot_mot17/{seq_name}"
        )
        expected_hota = expected_results[seq_name]["HOTA"]
        _verify_hota_metrics(
            seq_result.HOTA, expected_hota, f"sportsmot_mot17/{seq_name}"
        )
        expected_identity = expected_results[seq_name]["Identity"]
        _verify_identity_metrics(
            seq_result.Identity, expected_identity, f"sportsmot_mot17/{seq_name}"
        )


@pytest.mark.integration
def test_evaluate_benchmark_dancetrack_flat(
    dancetrack_flat_data: tuple[Path, dict[str, Any]],
) -> None:
    """Test evaluate_benchmark with DanceTrack flat format (auto-detected)."""
    data_path, expected_results = dancetrack_flat_data

    # Auto-detection should detect flat format from *.txt files
    result = evaluate_benchmark(
        gt_dir=data_path / "gt",
        tracker_dir=data_path / "trackers",
        seqmap=data_path / "seqmap.txt",
        metrics=["CLEAR", "HOTA", "Identity"],
    )

    # Verify all sequences are evaluated
    assert len(result.sequences) == len(expected_results), (
        f"Sequence count mismatch: got {len(result.sequences)}, "
        f"expected {len(expected_results)}"
    )

    # Verify each sequence matches expected results
    for seq_name, seq_result in result.sequences.items():
        assert seq_name in expected_results, f"Unexpected sequence: {seq_name}"
        expected_clear = expected_results[seq_name]["CLEAR"]
        _verify_clear_metrics(
            seq_result.CLEAR, expected_clear, f"dancetrack/{seq_name}"
        )
        expected_hota = expected_results[seq_name]["HOTA"]
        _verify_hota_metrics(seq_result.HOTA, expected_hota, f"dancetrack/{seq_name}")
        expected_identity = expected_results[seq_name]["Identity"]
        _verify_identity_metrics(
            seq_result.Identity, expected_identity, f"dancetrack/{seq_name}"
        )


@pytest.mark.integration
def test_evaluate_benchmark_dancetrack_mot17(
    dancetrack_mot17_data: tuple[Path, dict[str, Any]],
) -> None:
    """Test evaluate_benchmark with DanceTrack MOT17 format (auto-detected)."""
    data_path, expected_results = dancetrack_mot17_data

    # Auto-detection should detect MOT17 format, benchmark, split, and tracker
    result = evaluate_benchmark(
        gt_dir=data_path / "gt",
        tracker_dir=data_path / "trackers",
        metrics=["CLEAR", "HOTA", "Identity"],
    )

    # Verify all sequences are evaluated
    assert len(result.sequences) == len(expected_results), (
        f"Sequence count mismatch: got {len(result.sequences)}, "
        f"expected {len(expected_results)}"
    )

    # Verify each sequence matches expected results
    for seq_name, seq_result in result.sequences.items():
        assert seq_name in expected_results, f"Unexpected sequence: {seq_name}"
        expected_clear = expected_results[seq_name]["CLEAR"]
        _verify_clear_metrics(
            seq_result.CLEAR, expected_clear, f"dancetrack_mot17/{seq_name}"
        )
        expected_hota = expected_results[seq_name]["HOTA"]
        _verify_hota_metrics(
            seq_result.HOTA, expected_hota, f"dancetrack_mot17/{seq_name}"
        )
        expected_identity = expected_results[seq_name]["Identity"]
        _verify_identity_metrics(
            seq_result.Identity, expected_identity, f"dancetrack_mot17/{seq_name}"
        )


def _verify_clear_metrics(
    computed: Any,
    expected: dict[str, Any],
    context: str,
) -> None:
    """Verify CLEAR metrics match expected values.

    Args:
        computed: CLEARMetrics object from our computation.
        expected: Expected metrics dict from TrackEval.
        context: Context string for error messages (e.g., "sportsmot/seq1").
    """
    # Integer metrics must match exactly
    integer_metrics = ["CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag"]
    for metric in integer_metrics:
        if metric not in expected:
            continue
        computed_val = getattr(computed, metric)
        expected_val = expected[metric]
        assert computed_val == expected_val, (
            f"{context}: {metric} mismatch - "
            f"got {computed_val}, expected {expected_val}"
        )

    # Float metrics: both values should be fractions (0-1)
    float_metrics = ["MOTA", "MOTP"]
    for metric in float_metrics:
        if metric not in expected:
            continue
        computed_val = getattr(computed, metric)
        expected_val = expected[metric]
        assert computed_val == pytest.approx(expected_val, rel=1e-4, abs=1e-4), (
            f"{context}: {metric} mismatch - "
            f"got {computed_val}, expected {expected_val}"
        )


def _verify_hota_metrics(
    computed: Any,
    expected: dict[str, Any],
    context: str,
) -> None:
    """Verify HOTA metrics match expected values.

    Args:
        computed: HOTAMetrics object from our computation.
        expected: Expected metrics dict from TrackEval.
        context: Context string for error messages (e.g., "sportsmot/seq1").
    """
    float_metrics = ["HOTA", "DetA", "AssA", "LocA"]
    for metric in float_metrics:
        if metric not in expected:
            continue
        computed_val = getattr(computed, metric)
        expected_val = expected[metric]
        assert computed_val == pytest.approx(expected_val, rel=1e-4, abs=1e-4), (
            f"{context}: {metric} mismatch - "
            f"got {computed_val}, expected {expected_val}"
        )


def _verify_identity_metrics(
    computed: Any,
    expected: dict[str, Any],
    context: str,
) -> None:
    """Verify Identity metrics match expected values.

    Args:
        computed: IdentityMetrics object from our computation.
        expected: Expected metrics dict from TrackEval.
        context: Context string for error messages (e.g., "sportsmot/seq1").
    """
    # Integer metrics must match exactly
    int_metrics = ["IDTP", "IDFN", "IDFP"]
    for metric in int_metrics:
        if metric not in expected:
            continue
        computed_val = getattr(computed, metric)
        expected_val = expected[metric]
        assert computed_val == expected_val, (
            f"{context}: {metric} mismatch - "
            f"got {computed_val}, expected {expected_val}"
        )

    # Float metrics with tolerance
    float_metrics = ["IDF1", "IDR", "IDP"]
    for metric in float_metrics:
        if metric not in expected:
            continue
        computed_val = getattr(computed, metric)
        expected_val = expected[metric]
        assert computed_val == pytest.approx(expected_val, rel=1e-4, abs=1e-4), (
            f"{context}: {metric} mismatch - "
            f"got {computed_val}, expected {expected_val}"
        )
