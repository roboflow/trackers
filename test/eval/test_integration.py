# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Integration tests comparing our metrics against TrackEval on real data.

These tests download the SoccerNet MOT test dataset and verify that our
metrics implementation produces identical results to TrackEval.
Numerical parity is the key requirement.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from trackers.eval.clear import compute_clear_metrics
from trackers.eval.io import load_mot_file, prepare_mot_sequence

if TYPE_CHECKING:
    pass


@pytest.mark.integration
def test_clear_metrics_match_trackeval(
    sequence_name: str,
    soccernet_test_data: tuple[Path, dict[str, Any]],
) -> None:
    """Verify CLEAR metrics match TrackEval for each sequence."""
    data_path, expected_results = soccernet_test_data

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

    expected_clear = expected_results["sequences"][sequence_name]["CLEAR"]

    # Integer metrics must match exactly
    integer_metrics = ["CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag"]
    for metric in integer_metrics:
        assert result[metric] == expected_clear[metric], (
            f"{sequence_name}: {metric} mismatch - "
            f"got {result[metric]}, expected {expected_clear[metric]}"
        )

    # Float metrics: our implementation outputs fractions (0-1),
    # TrackEval outputs percentages (0-100)
    float_metrics = ["MOTA", "MOTP", "MTR", "PTR", "MLR"]
    for metric in float_metrics:
        expected_val = expected_clear[metric]
        result_val = result[metric] * 100

        assert result_val == pytest.approx(expected_val, rel=1e-4, abs=1e-2), (
            f"{sequence_name}: {metric} mismatch - "
            f"got {result_val}, expected {expected_val}"
        )
