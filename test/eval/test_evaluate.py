# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for high-level evaluation functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from trackers.eval import evaluate_mot_sequence


@pytest.fixture
def sample_mot_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create sample GT and tracker MOT files for testing."""
    gt_content = "1,1,100,200,50,60,1,1\n1,2,150,250,40,50,1,1\n2,1,105,205,50,60,1,1\n"
    tracker_content = (
        "1,10,102,202,50,60,0.9,1\n1,20,152,252,40,50,0.8,1\n2,10,107,207,50,60,0.9,1\n"
    )

    gt_path = tmp_path / "gt.txt"
    tracker_path = tmp_path / "tracker.txt"
    gt_path.write_text(gt_content)
    tracker_path.write_text(tracker_content)

    return gt_path, tracker_path


def test_evaluate_mot_sequence_hota_only(sample_mot_files: tuple[Path, Path]) -> None:
    """Test evaluate_mot_sequence with only HOTA metrics (no CLEAR)."""
    gt_path, tracker_path = sample_mot_files

    result = evaluate_mot_sequence(
        gt_path=gt_path,
        tracker_path=tracker_path,
        metrics=["HOTA"],
    )

    # HOTA should be present
    assert result.HOTA is not None
    assert result.HOTA.HOTA >= 0
    assert result.HOTA.DetA >= 0
    assert result.HOTA.AssA >= 0

    # CLEAR should be None when not requested
    assert result.CLEAR is None

    # Identity should be None when not requested
    assert result.Identity is None


def test_evaluate_mot_sequence_identity_only(
    sample_mot_files: tuple[Path, Path],
) -> None:
    """Test evaluate_mot_sequence with only Identity metrics."""
    gt_path, tracker_path = sample_mot_files

    result = evaluate_mot_sequence(
        gt_path=gt_path,
        tracker_path=tracker_path,
        metrics=["Identity"],
    )

    # Identity should be present
    assert result.Identity is not None
    assert result.Identity.IDF1 >= 0

    # CLEAR and HOTA should be None
    assert result.CLEAR is None
    assert result.HOTA is None


def test_evaluate_mot_sequence_clear_only(sample_mot_files: tuple[Path, Path]) -> None:
    """Test evaluate_mot_sequence with only CLEAR metrics (default)."""
    gt_path, tracker_path = sample_mot_files

    result = evaluate_mot_sequence(
        gt_path=gt_path,
        tracker_path=tracker_path,
        metrics=["CLEAR"],
    )

    # CLEAR should be present
    assert result.CLEAR is not None
    assert result.CLEAR.MOTA is not None

    # HOTA and Identity should be None
    assert result.HOTA is None
    assert result.Identity is None


def test_evaluate_mot_sequence_all_metrics(sample_mot_files: tuple[Path, Path]) -> None:
    """Test evaluate_mot_sequence with all metrics."""
    gt_path, tracker_path = sample_mot_files

    result = evaluate_mot_sequence(
        gt_path=gt_path,
        tracker_path=tracker_path,
        metrics=["CLEAR", "HOTA", "Identity"],
    )

    # All should be present
    assert result.CLEAR is not None
    assert result.HOTA is not None
    assert result.Identity is not None


def test_evaluate_mot_sequence_table_hota_only(
    sample_mot_files: tuple[Path, Path],
) -> None:
    """Test that table() works when only HOTA is computed."""
    gt_path, tracker_path = sample_mot_files

    result = evaluate_mot_sequence(
        gt_path=gt_path,
        tracker_path=tracker_path,
        metrics=["HOTA"],
    )

    # table() should work without errors
    table_str = result.table()
    assert "HOTA" in table_str
    assert "DetA" in table_str
    # CLEAR columns should not be present
    assert "MOTA" not in table_str


def test_evaluate_mot_sequence_json_hota_only(
    sample_mot_files: tuple[Path, Path],
) -> None:
    """Test that json() works when only HOTA is computed."""
    gt_path, tracker_path = sample_mot_files

    result = evaluate_mot_sequence(
        gt_path=gt_path,
        tracker_path=tracker_path,
        metrics=["HOTA"],
    )

    # json() should work without errors
    json_str = result.json()
    assert "HOTA" in json_str
    assert "DetA" in json_str
