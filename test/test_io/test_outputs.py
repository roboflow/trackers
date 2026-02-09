# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for output handlers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from trackers.io import MOTOutput, validate_output_path


class TestMOTOutput:
    """Tests for MOTOutput context manager."""

    def test_writes_mot_format(self, tmp_path: Path) -> None:
        output_file = tmp_path / "output.txt"

        detections = sv.Detections(
            xyxy=np.array([[100.0, 100.0, 150.0, 180.0]]),
            confidence=np.array([0.95]),
            tracker_id=np.array([1]),
        )

        with MOTOutput(output_file) as mot:
            mot.write(1, detections)

        content = output_file.read_text()
        assert content.startswith("1,1,100.00,100.00,50.00,80.00,0.9500")

    def test_handles_empty_detections(self, tmp_path: Path) -> None:
        output_file = tmp_path / "output.txt"

        with MOTOutput(output_file) as mot:
            mot.write(1, sv.Detections.empty())

        content = output_file.read_text()
        assert content == ""

    def test_handles_none_path(self) -> None:
        with MOTOutput(None) as mot:
            mot.write(1, sv.Detections.empty())  # should not raise


class TestValidateOutputPath:
    """Tests for validate_output_path function."""

    def test_raises_when_path_exists(self, tmp_path: Path) -> None:
        existing_file = tmp_path / "existing.mp4"
        existing_file.touch()

        with pytest.raises(FileExistsError, match="already exists"):
            validate_output_path(existing_file)

    def test_allows_overwrite(self, tmp_path: Path) -> None:
        existing_file = tmp_path / "existing.mp4"
        existing_file.touch()

        validate_output_path(existing_file, overwrite=True)

    def test_allows_nonexistent_path(self, tmp_path: Path) -> None:
        validate_output_path(tmp_path / "new.mp4")

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        nested_path = tmp_path / "a" / "b" / "c" / "output.mp4"
        assert not nested_path.parent.exists()

        validate_output_path(nested_path)

        assert nested_path.parent.exists()
