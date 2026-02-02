# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Integration tests comparing our metrics against TrackEval on real data.

These tests download SportsMOT and DanceTrack test datasets and verify that our
metrics implementation produces identical results to TrackEval.
Numerical parity is the key requirement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from trackers.eval.clear import compute_clear_metrics
from trackers.eval.io import load_mot_file, prepare_mot_sequence

if TYPE_CHECKING:
    pass

# Import the helper function from conftest
from test.conftest import _get_test_data


@pytest.mark.integration
def test_clear_metrics_match_trackeval(
    dataset_name: str,
    sequence_name: str,
) -> None:
    """Verify CLEAR metrics match TrackEval for each sequence."""
    data_path, expected_results = _get_test_data(dataset_name)

    gt_path = data_path / "gt" / f"{sequence_name}.txt"
    tracker_path = data_path / "trackers" / f"{sequence_name}.txt"

    gt_data = load_mot_file(gt_path)
    tracker_data = load_mot_file(tracker_path)
    seq_data = prepare_mot_sequence(gt_data, tracker_data)

    result = compute_clear_metrics(
        seq_data.gt_ids,
        seq_data.tracker_ids,
        seq_data.similarity_scores,
        threshold=0.5,
    )

    # New format: sequences are top-level keys (no "sequences" wrapper)
    expected_clear = expected_results[sequence_name]["CLEAR"]

    # Integer metrics must match exactly
    integer_metrics = ["CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag"]
    for metric in integer_metrics:
        if metric not in expected_clear:
            continue
        assert result[metric] == expected_clear[metric], (
            f"{dataset_name}/{sequence_name}: {metric} mismatch - "
            f"got {result[metric]}, expected {expected_clear[metric]}"
        )

    # Float metrics: both our implementation and expected results are fractions (0-1)
    float_metrics = ["MOTA", "MOTP"]
    for metric in float_metrics:
        expected_val = expected_clear[metric]
        result_val = result[metric]

        assert result_val == pytest.approx(expected_val, rel=1e-4, abs=1e-4), (
            f"{dataset_name}/{sequence_name}: {metric} mismatch - "
            f"got {result_val}, expected {expected_val}"
        )
