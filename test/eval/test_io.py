# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trackers.eval.io import (
    MOTFrameData,
    load_mot_file,
    prepare_mot_sequence,
)


class TestLoadMotFile:
    """Tests for load_mot_file function."""

    def test_valid_single_frame(self, tmp_path: Path) -> None:
        """Load a valid MOT file with a single frame."""
        content = "1,1,100,200,50,60,0.9,1\n1,2,150,250,40,50,0.8,1\n"
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        result = load_mot_file(file_path)

        assert len(result) == 1
        assert 1 in result
        assert len(result[1].ids) == 2
        assert np.array_equal(result[1].ids, [1, 2])
        assert result[1].boxes.shape == (2, 4)
        assert np.allclose(result[1].boxes[0], [100, 200, 50, 60])

    def test_valid_multi_frame(self, tmp_path: Path) -> None:
        """Load a valid MOT file with multiple frames."""
        content = (
            "1,1,100,200,50,60,0.9,1\n"
            "1,2,150,250,40,50,0.8,1\n"
            "2,1,105,205,50,60,0.9,1\n"
            "3,1,110,210,50,60,0.9,1\n"
            "3,3,200,300,30,40,0.7,1\n"
        )
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        result = load_mot_file(file_path)

        assert len(result) == 3
        assert set(result.keys()) == {1, 2, 3}
        assert len(result[1].ids) == 2
        assert len(result[2].ids) == 1
        assert len(result[3].ids) == 2

    def test_file_not_found(self) -> None:
        """Raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="MOT file not found"):
            load_mot_file("/nonexistent/path/to/file.txt")

    def test_empty_file(self, tmp_path: Path) -> None:
        """Raise ValueError for empty file."""
        file_path = tmp_path / "empty.txt"
        file_path.write_text("")

        with pytest.raises(ValueError, match="MOT file is empty"):
            load_mot_file(file_path)

    def test_missing_columns(self, tmp_path: Path) -> None:
        """Raise ValueError for file with too few columns."""
        content = "1,1,100,200\n"  # Only 4 columns, need 6
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        with pytest.raises(ValueError, match="expected at least 6 columns"):
            load_mot_file(file_path)

    def test_extra_columns_ignored(self, tmp_path: Path) -> None:
        """Extra columns beyond required are ignored."""
        content = "1,1,100,200,50,60,0.9,1,0.5,-1,-1,-1\n"
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        result = load_mot_file(file_path)

        assert len(result) == 1
        assert len(result[1].ids) == 1
        assert result[1].boxes.shape == (1, 4)

    def test_minimal_columns(self, tmp_path: Path) -> None:
        """Load file with exactly 6 columns (no conf, no class)."""
        content = "1,1,100,200,50,60\n"
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        result = load_mot_file(file_path)

        assert len(result) == 1
        assert result[1].confidences[0] == 1.0  # Default confidence
        assert result[1].classes[0] == 1  # Default class

    def test_with_confidence_no_class(self, tmp_path: Path) -> None:
        """Load file with 7 columns (conf but no class)."""
        content = "1,1,100,200,50,60,0.75\n"
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        result = load_mot_file(file_path)

        assert result[1].confidences[0] == 0.75
        assert result[1].classes[0] == 1  # Default class

    def test_spaces_as_delimiter(self, tmp_path: Path) -> None:
        """Load file with space-separated values."""
        content = "1 1 100 200 50 60 0.9 1\n"
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        result = load_mot_file(file_path)

        assert len(result) == 1
        assert np.array_equal(result[1].ids, [1])

    def test_non_sequential_frames(self, tmp_path: Path) -> None:
        """Load file with non-sequential frame numbers."""
        content = (
            "1,1,100,200,50,60,0.9,1\n"
            "5,1,100,200,50,60,0.9,1\n"
            "10,1,100,200,50,60,0.9,1\n"
        )
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        result = load_mot_file(file_path)

        assert set(result.keys()) == {1, 5, 10}


class TestPrepareMotSequence:
    """Tests for prepare_mot_sequence function."""

    def test_perfect_tracking(self) -> None:
        """GT and tracker match perfectly."""
        gt_data = {
            1: MOTFrameData(
                ids=np.array([1, 2]),
                boxes=np.array([[100, 100, 50, 50], [200, 200, 50, 50]]),
                confidences=np.array([1.0, 1.0]),
                classes=np.array([1, 1]),
            ),
            2: MOTFrameData(
                ids=np.array([1, 2]),
                boxes=np.array([[105, 105, 50, 50], [205, 205, 50, 50]]),
                confidences=np.array([1.0, 1.0]),
                classes=np.array([1, 1]),
            ),
        }
        tracker_data = {
            1: MOTFrameData(
                ids=np.array([10, 20]),
                boxes=np.array([[100, 100, 50, 50], [200, 200, 50, 50]]),
                confidences=np.array([0.9, 0.9]),
                classes=np.array([1, 1]),
            ),
            2: MOTFrameData(
                ids=np.array([10, 20]),
                boxes=np.array([[105, 105, 50, 50], [205, 205, 50, 50]]),
                confidences=np.array([0.9, 0.9]),
                classes=np.array([1, 1]),
            ),
        }

        result = prepare_mot_sequence(gt_data, tracker_data)

        assert result.num_frames == 2
        assert result.num_gt_ids == 2
        assert result.num_tracker_ids == 2
        assert result.num_gt_dets == 4
        assert result.num_tracker_dets == 4
        # Check IoU is 1.0 for matching boxes
        assert np.allclose(result.similarity_scores[0].diagonal(), [1.0, 1.0])

    def test_missing_frames_in_tracker(self) -> None:
        """Tracker has missing frames."""
        gt_data = {
            1: MOTFrameData(
                ids=np.array([1]),
                boxes=np.array([[100, 100, 50, 50]]),
                confidences=np.array([1.0]),
                classes=np.array([1]),
            ),
            2: MOTFrameData(
                ids=np.array([1]),
                boxes=np.array([[105, 105, 50, 50]]),
                confidences=np.array([1.0]),
                classes=np.array([1]),
            ),
        }
        tracker_data = {
            1: MOTFrameData(
                ids=np.array([10]),
                boxes=np.array([[100, 100, 50, 50]]),
                confidences=np.array([0.9]),
                classes=np.array([1]),
            ),
            # Frame 2 missing
        }

        result = prepare_mot_sequence(gt_data, tracker_data)

        assert result.num_frames == 2
        assert result.num_gt_dets == 2
        assert result.num_tracker_dets == 1
        # Frame 2 should have empty tracker IDs
        assert len(result.tracker_ids[1]) == 0
        assert result.similarity_scores[1].shape == (1, 0)

    def test_missing_frames_in_gt(self) -> None:
        """GT has missing frames."""
        gt_data = {
            1: MOTFrameData(
                ids=np.array([1]),
                boxes=np.array([[100, 100, 50, 50]]),
                confidences=np.array([1.0]),
                classes=np.array([1]),
            ),
            # Frame 2 missing
        }
        tracker_data = {
            1: MOTFrameData(
                ids=np.array([10]),
                boxes=np.array([[100, 100, 50, 50]]),
                confidences=np.array([0.9]),
                classes=np.array([1]),
            ),
            2: MOTFrameData(
                ids=np.array([10]),
                boxes=np.array([[105, 105, 50, 50]]),
                confidences=np.array([0.9]),
                classes=np.array([1]),
            ),
        }

        result = prepare_mot_sequence(gt_data, tracker_data)

        assert result.num_frames == 2
        assert result.num_gt_dets == 1
        assert result.num_tracker_dets == 2
        # Frame 2 should have empty GT IDs
        assert len(result.gt_ids[1]) == 0
        assert result.similarity_scores[1].shape == (0, 1)

    def test_empty_gt(self) -> None:
        """Empty GT data."""
        gt_data: dict[int, MOTFrameData] = {}
        tracker_data = {
            1: MOTFrameData(
                ids=np.array([10]),
                boxes=np.array([[100, 100, 50, 50]]),
                confidences=np.array([0.9]),
                classes=np.array([1]),
            ),
        }

        result = prepare_mot_sequence(gt_data, tracker_data)

        assert result.num_frames == 1
        assert result.num_gt_ids == 0
        assert result.num_gt_dets == 0
        assert result.num_tracker_dets == 1

    def test_empty_tracker(self) -> None:
        """Empty tracker data."""
        gt_data = {
            1: MOTFrameData(
                ids=np.array([1]),
                boxes=np.array([[100, 100, 50, 50]]),
                confidences=np.array([1.0]),
                classes=np.array([1]),
            ),
        }
        tracker_data: dict[int, MOTFrameData] = {}

        result = prepare_mot_sequence(gt_data, tracker_data)

        assert result.num_frames == 1
        assert result.num_tracker_ids == 0
        assert result.num_tracker_dets == 0
        assert result.num_gt_dets == 1

    def test_id_remapping_non_sequential(self) -> None:
        """Non-sequential IDs like 100, 500 are remapped to 0, 1."""
        gt_data = {
            1: MOTFrameData(
                ids=np.array([100, 500]),
                boxes=np.array([[100, 100, 50, 50], [200, 200, 50, 50]]),
                confidences=np.array([1.0, 1.0]),
                classes=np.array([1, 1]),
            ),
        }
        tracker_data = {
            1: MOTFrameData(
                ids=np.array([999, 1000]),
                boxes=np.array([[100, 100, 50, 50], [200, 200, 50, 50]]),
                confidences=np.array([0.9, 0.9]),
                classes=np.array([1, 1]),
            ),
        }

        result = prepare_mot_sequence(gt_data, tracker_data)

        # IDs should be remapped to 0-indexed
        assert result.gt_id_mapping == {100: 0, 500: 1}
        assert result.tracker_id_mapping == {999: 0, 1000: 1}
        # Remapped IDs should be used
        assert np.array_equal(result.gt_ids[0], [0, 1])
        assert np.array_equal(result.tracker_ids[0], [0, 1])

    def test_num_frames_auto_detection(self) -> None:
        """num_frames is auto-detected from max frame number."""
        gt_data = {
            1: MOTFrameData(
                ids=np.array([1]),
                boxes=np.array([[100, 100, 50, 50]]),
                confidences=np.array([1.0]),
                classes=np.array([1]),
            ),
            5: MOTFrameData(
                ids=np.array([1]),
                boxes=np.array([[100, 100, 50, 50]]),
                confidences=np.array([1.0]),
                classes=np.array([1]),
            ),
        }
        tracker_data = {
            3: MOTFrameData(
                ids=np.array([10]),
                boxes=np.array([[100, 100, 50, 50]]),
                confidences=np.array([0.9]),
                classes=np.array([1]),
            ),
        }

        result = prepare_mot_sequence(gt_data, tracker_data)

        assert result.num_frames == 5
        assert len(result.gt_ids) == 5
        assert len(result.tracker_ids) == 5

    def test_num_frames_explicit(self) -> None:
        """Explicit num_frames overrides auto-detection."""
        gt_data = {
            1: MOTFrameData(
                ids=np.array([1]),
                boxes=np.array([[100, 100, 50, 50]]),
                confidences=np.array([1.0]),
                classes=np.array([1]),
            ),
        }
        tracker_data = {
            1: MOTFrameData(
                ids=np.array([10]),
                boxes=np.array([[100, 100, 50, 50]]),
                confidences=np.array([0.9]),
                classes=np.array([1]),
            ),
        }

        result = prepare_mot_sequence(gt_data, tracker_data, num_frames=10)

        assert result.num_frames == 10
        assert len(result.gt_ids) == 10

    def test_both_empty(self) -> None:
        """Both GT and tracker are empty."""
        result = prepare_mot_sequence({}, {})

        assert result.num_frames == 0
        assert result.num_gt_ids == 0
        assert result.num_tracker_ids == 0
        assert result.num_gt_dets == 0
        assert result.num_tracker_dets == 0

    def test_similarity_scores_shape(self) -> None:
        """Similarity matrices have correct shape for each frame."""
        gt_data = {
            1: MOTFrameData(
                ids=np.array([1, 2, 3]),
                boxes=np.array(
                    [[100, 100, 50, 50], [200, 200, 50, 50], [300, 300, 50, 50]]
                ),
                confidences=np.array([1.0, 1.0, 1.0]),
                classes=np.array([1, 1, 1]),
            ),
            2: MOTFrameData(
                ids=np.array([1]),
                boxes=np.array([[105, 105, 50, 50]]),
                confidences=np.array([1.0]),
                classes=np.array([1]),
            ),
        }
        tracker_data = {
            1: MOTFrameData(
                ids=np.array([10, 20]),
                boxes=np.array([[100, 100, 50, 50], [200, 200, 50, 50]]),
                confidences=np.array([0.9, 0.9]),
                classes=np.array([1, 1]),
            ),
            2: MOTFrameData(
                ids=np.array([10, 20, 30, 40]),
                boxes=np.array(
                    [
                        [105, 105, 50, 50],
                        [205, 205, 50, 50],
                        [305, 305, 50, 50],
                        [405, 405, 50, 50],
                    ]
                ),
                confidences=np.array([0.9, 0.9, 0.9, 0.9]),
                classes=np.array([1, 1, 1, 1]),
            ),
        }

        result = prepare_mot_sequence(gt_data, tracker_data)

        # Frame 1: 3 GT x 2 tracker
        assert result.similarity_scores[0].shape == (3, 2)
        # Frame 2: 1 GT x 4 tracker
        assert result.similarity_scores[1].shape == (1, 4)
