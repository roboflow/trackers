# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Integration tests comparing our metrics against TrackEval on real data.

These tests download the SoccerNet MOT test dataset and verify that our
CLEAR metrics implementation produces identical results to TrackEval.
Numerical parity is the key requirement.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from test.conftest import SOCCERNET_SEQUENCES
from trackers.eval.clear import compute_clear_metrics
from trackers.eval.io import load_mot_file, prepare_mot_sequence

if TYPE_CHECKING:
    pass


@pytest.mark.integration
@pytest.mark.parametrize("sequence_name", SOCCERNET_SEQUENCES)
def test_clear_metrics_match_trackeval(
    sequence_name: str,
    soccernet_test_data: tuple[Path, dict[str, Any]],
) -> None:
    """Verify CLEAR metrics match TrackEval for each sequence.

    This test loads GT and tracker data for a sequence, runs our CLEAR
    metrics pipeline, and compares results against TrackEval's output.

    Args:
        sequence_name: Name of the sequence (e.g., "SNMOT-116").
        soccernet_test_data: Fixture providing data path and expected results.
    """
    data_path, expected_results = soccernet_test_data

    # Load GT and tracker data
    gt_path = data_path / "gt" / f"{sequence_name}.txt"
    tracker_path = data_path / "trackers" / f"{sequence_name}.txt"

    gt_data = load_mot_file(gt_path)
    tracker_data = load_mot_file(tracker_path)

    # Prepare sequence data (computes IoU, remaps IDs)
    seq_data = prepare_mot_sequence(gt_data, tracker_data)

    # Compute CLEAR metrics
    result = compute_clear_metrics(
        seq_data.gt_ids,
        seq_data.tracker_ids,
        seq_data.similarity_scores,
        threshold=0.5,
    )

    # Get expected results from TrackEval
    expected_clear = expected_results["sequences"][sequence_name]["CLEAR"]

    # Integer metrics - must match exactly
    integer_metrics = [
        "CLR_TP",
        "CLR_FN",
        "CLR_FP",
        "IDSW",
        "MT",
        "PT",
        "ML",
        "Frag",
    ]
    for metric in integer_metrics:
        assert result[metric] == expected_clear[metric], (
            f"{sequence_name}: {metric} mismatch - "
            f"got {result[metric]}, expected {expected_clear[metric]}"
        )

    # Float metrics - allow small tolerance for percentage rounding
    # TrackEval displays 5 decimal places, we use rel=1e-4 for safety
    # Note: Our implementation outputs fractions (0-1), TrackEval outputs % (0-100)
    float_metrics_ours = [
        "MOTA",
        "MOTP",
        "MTR",
        "PTR",
        "MLR",
    ]
    for metric in float_metrics_ours:
        # TrackEval outputs percentages (0-100), our implementation uses fractions (0-1)
        # Convert our result to percentage for comparison
        expected_val = expected_clear[metric]
        result_val = result[metric] * 100  # Convert fraction to percentage

        assert result_val == pytest.approx(expected_val, rel=1e-4, abs=1e-2), (
            f"{sequence_name}: {metric} mismatch - "
            f"got {result_val}, expected {expected_val}"
        )
