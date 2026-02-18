# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Integration tests comparing our metrics against TrackEval on real data.

These tests download SportsMOT and DanceTrack test datasets and verify that our
benchmark evaluation produces identical results to TrackEval.
Numerical parity is the key requirement.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from trackers.eval import evaluate_mot_sequences

if TYPE_CHECKING:
    pass


@pytest.mark.integration
def test_evaluate_mot_sequences_sportsmot_flat(
    sportsmot_flat_data: tuple[Path, dict[str, Any]],
) -> None:
    """Test evaluate_mot_sequences with SportsMOT flat format (auto-detected)."""
    data_path, expected_results = sportsmot_flat_data

    # Auto-detection should detect flat format from *.txt files
    result = evaluate_mot_sequences(
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
def test_evaluate_mot_sequences_sportsmot_mot17(
    sportsmot_mot17_data: tuple[Path, dict[str, Any]],
) -> None:
    """Test evaluate_mot_sequences with SportsMOT MOT17 format (auto-detected)."""
    data_path, expected_results = sportsmot_mot17_data

    # Point directly at the split-level directories
    gt_dir, tracker_dir = _resolve_mot17_dirs(data_path)

    result = evaluate_mot_sequences(
        gt_dir=gt_dir,
        tracker_dir=tracker_dir,
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
def test_evaluate_mot_sequences_dancetrack_flat(
    dancetrack_flat_data: tuple[Path, dict[str, Any]],
) -> None:
    """Test evaluate_mot_sequences with DanceTrack flat format (auto-detected)."""
    data_path, expected_results = dancetrack_flat_data

    # Auto-detection should detect flat format from *.txt files
    result = evaluate_mot_sequences(
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
def test_evaluate_mot_sequences_dancetrack_mot17(
    dancetrack_mot17_data: tuple[Path, dict[str, Any]],
) -> None:
    """Test evaluate_mot_sequences with DanceTrack MOT17 format (auto-detected)."""
    data_path, expected_results = dancetrack_mot17_data

    # Point directly at the split-level directories
    gt_dir, tracker_dir = _resolve_mot17_dirs(data_path)

    result = evaluate_mot_sequences(
        gt_dir=gt_dir,
        tracker_dir=tracker_dir,
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


def _resolve_mot17_dirs(data_path: Path) -> tuple[Path, Path]:
    """Resolve gt_dir and tracker_dir for MOT17 layout test data.

    The MOT17 test data has structure:
        data_path/gt/{Benchmark}-{split}/{seq}/gt/gt.txt
        data_path/trackers/{Benchmark}-{split}/{tracker}/data/{seq}.txt

    This function discovers the split directory and tracker data directory
    so tests can pass them directly.
    """
    gt_root = data_path / "gt"
    tracker_root = data_path / "trackers"

    # Find the single {Benchmark}-{split} directory under gt/
    gt_splits = [d for d in gt_root.iterdir() if d.is_dir()]
    assert len(gt_splits) == 1, f"Expected one split dir in {gt_root}, got {gt_splits}"
    gt_dir = gt_splits[0]

    # Find the matching tracker data directory
    tracker_split = tracker_root / gt_splits[0].name
    tracker_names = [d for d in tracker_split.iterdir() if d.is_dir()]
    assert len(tracker_names) == 1, (
        f"Expected one tracker dir in {tracker_split}, got {tracker_names}"
    )
    tracker_dir = tracker_names[0] / "data"

    return gt_dir, tracker_dir


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
