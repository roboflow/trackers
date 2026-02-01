# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for MOT file I/O functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trackers.eval.io import (
    load_mot_file,
)

# ============================================================================
# load_mot_file tests
# ============================================================================


def test_load_mot_file_valid(tmp_path: Path) -> None:
    """Load a valid MOT file with multiple frames."""
    content = (
        "1,1,100,200,50,60,0.9,1\n"
        "1,2,150,250,40,50,0.8,1\n"
        "2,1,105,205,50,60,0.9,1\n"
    )
    file_path = tmp_path / "test.txt"
    file_path.write_text(content)

    result = load_mot_file(file_path)

    assert set(result.keys()) == {1, 2}
    assert np.array_equal(result[1].ids, [1, 2])
    assert np.array_equal(result[2].ids, [1])
    assert result[1].boxes.shape == (2, 4)
    assert np.allclose(result[1].boxes[0], [100, 200, 50, 60])


@pytest.mark.parametrize(
    "content,expected_conf,expected_class",
    [
        pytest.param("1,1,100,200,50,60\n", 1.0, 1, id="minimal_6_columns"),
        pytest.param("1,1,100,200,50,60,0.75\n", 0.75, 1, id="7_columns_conf_only"),
        pytest.param("1,1,100,200,50,60,0.9,2\n", 0.9, 2, id="8_columns_full"),
    ],
)
def test_load_mot_file_column_defaults(
    tmp_path: Path,
    content: str,
    expected_conf: float,
    expected_class: int,
) -> None:
    """Test default values for optional columns."""
    file_path = tmp_path / "test.txt"
    file_path.write_text(content)

    result = load_mot_file(file_path)

    assert result[1].confidences[0] == pytest.approx(expected_conf)
    assert result[1].classes[0] == expected_class


def test_load_mot_file_not_found() -> None:
    """Raise FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError, match="MOT file not found"):
        load_mot_file("/nonexistent/path/to/file.txt")


def test_load_mot_file_empty(tmp_path: Path) -> None:
    """Raise ValueError for empty file."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")

    with pytest.raises(ValueError, match="MOT file is empty"):
        load_mot_file(file_path)


def test_load_mot_file_invalid_columns(tmp_path: Path) -> None:
    """Raise ValueError for file with too few columns."""
    content = "1,1,100,200\n"
    file_path = tmp_path / "test.txt"
    file_path.write_text(content)

    with pytest.raises(ValueError, match="expected at least 6 columns"):
        load_mot_file(file_path)
