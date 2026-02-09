# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for video utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest

from trackers import frames_from_source

FRAME_WIDTH = 96
FRAME_HEIGHT = 96
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)
VALUE_MULTIPLIER = 40
VIDEO_COMPRESSION_TOLERANCE = 5


def create_frame(index: int) -> np.ndarray:
    """Create a test frame with deterministic pixel values for verification.

    Each frame has all pixels set to the same value derived from the index.
    The value is calculated as index * VALUE_MULTIPLIER (clamped to 255).

    We use VALUE_MULTIPLIER=40 to spread values apart (0, 40, 80, 120, ...)
    because video codecs like mp4v use lossy compression that can alter
    pixel values by small amounts. Adjacent values like 0, 1, 2, 3 would
    become indistinguishable after compression, but 0, 40, 80, 120 remain
    clearly distinguishable even with compression artifacts.

    For lossless formats (PNG, JPG with quality=100), exact matching works.
    For video files, use expected_frame_value() with a tolerance check.
    """
    pixel_value = min(index * VALUE_MULTIPLIER, 255)
    return np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), pixel_value, dtype=np.uint8)


def expected_frame_value(index: int) -> int:
    """Return the expected pixel value for a frame created with create_frame(index)."""
    return min(index * VALUE_MULTIPLIER, 255)


@pytest.fixture
def video_factory(tmp_path: Path) -> Callable[[int], Path]:
    """Factory for creating test videos with specified number of frames."""

    def _create(n_frames: int) -> Path:
        video_path = tmp_path / f"video_{n_frames}_frames.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 25.0, FRAME_SIZE)

        for index in range(n_frames):
            writer.write(create_frame(index))
        writer.release()

        return video_path

    return _create


@pytest.fixture
def image_directory_factory(tmp_path: Path) -> Callable[[int, str], Path]:
    """Factory for creating image directories with specified number of frames."""

    def _create(n_frames: int, filename_pattern: str = "{:04d}.png") -> Path:
        directory = tmp_path / f"imgdir_{n_frames}_frames"
        directory.mkdir(exist_ok=True)

        for index in range(n_frames):
            filename = filename_pattern.format(index)
            cv2.imwrite(str(directory / filename), create_frame(index))

        return directory

    return _create


@pytest.fixture
def empty_directory(tmp_path: Path) -> Path:
    """Empty directory with no files."""
    directory = tmp_path / "empty"
    directory.mkdir()
    return directory


@pytest.fixture
def directory_with_non_image_files(tmp_path: Path) -> Path:
    """Directory containing only non-image files."""
    directory = tmp_path / "non_images"
    directory.mkdir()
    for index in range(4):
        (directory / f"note_{index}.txt").write_text(f"placeholder {index}")
    return directory


@pytest.fixture
def directory_with_corrupted_image(tmp_path: Path) -> Path:
    """Directory with valid images followed by one corrupted image file."""
    directory = tmp_path / "with_corrupt"
    directory.mkdir()

    num_valid_images = 3
    for index in range(num_valid_images):
        cv2.imwrite(str(directory / f"{index:04d}.png"), create_frame(index))

    corrupted_image_path = directory / f"{num_valid_images:04d}.png"
    corrupted_image_path.write_bytes(b"not a valid image")

    return directory


class TestFramesFromSourceVideo:
    """Tests for frames_from_source with video files."""

    def test_reads_video_frames_in_order(self, video_factory) -> None:
        num_frames = 5
        video_path = video_factory(n_frames=num_frames)
        frames = list(frames_from_source(video_path))

        assert len(frames) == num_frames

        for frame_id, frame in frames:
            frame_index = frame_id - 1
            expected = expected_frame_value(frame_index)

            assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
            assert frame.dtype == np.uint8

            mean_pixel_value = frame.mean()
            assert abs(mean_pixel_value - expected) < VIDEO_COMPRESSION_TOLERANCE, (
                f"Frame {frame_id}: expected ~{expected}, "
                f"got mean {mean_pixel_value:.1f}"
            )

    def test_reads_single_frame_video(self, video_factory) -> None:
        video_path = video_factory(n_frames=1)
        frames = list(frames_from_source(video_path))

        assert len(frames) == 1
        assert frames[0][0] == 1

    def test_nonexistent_video_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Cannot open"):
            list(frames_from_source("/nonexistent/video.mp4"))


class TestFramesFromSourceImageDirectory:
    """Tests for frames_from_source with image directories."""

    def test_reads_images_in_alphabetical_order(self, image_directory_factory) -> None:
        num_frames = 7
        directory = image_directory_factory(
            n_frames=num_frames, filename_pattern="{:04d}.png"
        )
        frames = list(frames_from_source(directory))

        assert len(frames) == num_frames

        for frame_id, frame in frames:
            frame_index = frame_id - 1
            expected = expected_frame_value(frame_index)

            assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
            assert np.all(frame == expected)

    def test_reads_prefixed_filenames(self, image_directory_factory) -> None:
        num_frames = 4
        directory = image_directory_factory(
            n_frames=num_frames, filename_pattern="frame_{:05d}.png"
        )
        frames = list(frames_from_source(directory))

        assert len(frames) == num_frames

        for frame_id, frame in frames:
            frame_index = frame_id - 1
            expected = expected_frame_value(frame_index)
            assert np.all(frame == expected)

    def test_accepts_path_object(self, image_directory_factory) -> None:
        num_frames = 3
        directory = image_directory_factory(n_frames=num_frames)
        frames = list(frames_from_source(directory))
        assert len(frames) == num_frames

    def test_accepts_string_path(self, image_directory_factory) -> None:
        num_frames = 3
        directory = image_directory_factory(n_frames=num_frames)
        frames = list(frames_from_source(str(directory)))
        assert len(frames) == num_frames


class TestFramesFromSourceErrors:
    """Tests for frames_from_source error handling."""

    def test_empty_directory_raises_value_error(self, empty_directory) -> None:
        with pytest.raises(ValueError, match="No supported image files"):
            list(frames_from_source(empty_directory))

    def test_non_image_files_raises_value_error(
        self, directory_with_non_image_files
    ) -> None:
        with pytest.raises(ValueError, match="No supported image files"):
            list(frames_from_source(directory_with_non_image_files))

    def test_corrupted_image_raises_os_error(
        self, directory_with_corrupted_image
    ) -> None:
        with pytest.raises(OSError, match="Failed to read image"):
            list(frames_from_source(directory_with_corrupted_image))
