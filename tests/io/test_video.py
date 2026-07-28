# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from trackers import frames_from_source
from trackers.io.video import _DEFAULT_OUTPUT_FPS, _VideoOutput

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
                f"Frame {frame_id}: expected ~{expected}, got mean {mean_pixel_value:.1f}"
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
    def test_reads_images_in_alphabetical_order(self, image_directory_factory) -> None:
        num_frames = 7
        directory = image_directory_factory(n_frames=num_frames, filename_pattern="{:04d}.png")
        frames = list(frames_from_source(directory))

        assert len(frames) == num_frames

        for frame_id, frame in frames:
            frame_index = frame_id - 1
            expected = expected_frame_value(frame_index)

            assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
            assert np.all(frame == expected)

    def test_reads_prefixed_filenames(self, image_directory_factory) -> None:
        num_frames = 4
        directory = image_directory_factory(n_frames=num_frames, filename_pattern="frame_{:05d}.png")
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
    def test_empty_directory_raises_value_error(self, empty_directory) -> None:
        with pytest.raises(ValueError, match="No supported image files"):
            list(frames_from_source(empty_directory))

    def test_non_image_files_raises_value_error(self, directory_with_non_image_files) -> None:
        with pytest.raises(ValueError, match="No supported image files"):
            list(frames_from_source(directory_with_non_image_files))

    def test_corrupted_image_raises_os_error(self, directory_with_corrupted_image) -> None:
        with pytest.raises(OSError, match="Failed to read image"):
            list(frames_from_source(directory_with_corrupted_image))


class TestVideoOutputFPS:
    def test_uses_source_fps_when_provided(self, tmp_path: Path) -> None:
        output_path = tmp_path / "out.mp4"
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        with _VideoOutput(output_path, fps=24.0) as video:
            video.write(frame)

        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert actual_fps == pytest.approx(24.0, abs=0.1)

    def test_falls_back_to_default_fps(self, tmp_path: Path) -> None:
        output_path = tmp_path / "out.mp4"
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        with _VideoOutput(output_path) as video:
            video.write(frame)

        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert actual_fps == pytest.approx(_DEFAULT_OUTPUT_FPS, abs=0.1)


def _count_written_frames(path: Path) -> int:
    """Count the frames actually persisted in a written video file."""
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"Failed to open written video for verification: {path}"
    try:
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        return count
    finally:
        cap.release()


class TestVideoOutputResolutionChange:
    def test_all_same_size_frames_are_written(self, tmp_path: Path) -> None:
        output_path = tmp_path / "same_size.mp4"
        num_frames = 3

        with _VideoOutput(output_path) as video:
            for index in range(num_frames):
                assert video.write(create_frame(index)) is True

        assert _count_written_frames(output_path) == num_frames

    def test_mismatched_frame_is_kept_not_dropped(self, tmp_path: Path) -> None:
        # cv2.VideoWriter silently discards frames whose size differs from the
        # writer's; without resizing the middle frame vanishes from the output.
        output_path = tmp_path / "resized.mp4"
        first = create_frame(1)
        odd = np.full((FRAME_HEIGHT // 2, FRAME_WIDTH * 2, 3), 80, dtype=np.uint8)
        last = create_frame(3)

        with _VideoOutput(output_path) as video:
            assert video.write(first) is True
            assert video.write(odd) is True
            assert video.write(last) is True

        assert _count_written_frames(output_path) == 3

    def test_three_distinct_sizes_are_all_resized_and_kept(self, tmp_path: Path) -> None:
        """Writer resizes every subsequent distinct size, not only the first mismatch."""
        output_path = tmp_path / "three_sizes.mp4"
        size_a_frame = create_frame(1)  # opens writer at FRAME_SIZE (96x96)
        size_b_frame = np.full((FRAME_HEIGHT // 2, FRAME_WIDTH * 2, 3), expected_frame_value(2), dtype=np.uint8)
        size_c_frame = np.full((FRAME_HEIGHT * 2, FRAME_WIDTH // 2, 3), expected_frame_value(3), dtype=np.uint8)

        with _VideoOutput(output_path) as video:
            assert video.write(size_a_frame) is True
            assert video.write(size_b_frame) is True
            assert video.write(size_c_frame) is True

        # cv2.VideoWriter silently drops any frame whose size differs from the
        # writer's; a count of 3 proves both B and C were resized to match A's
        # size before being written, not just the first mismatch (B).
        assert _count_written_frames(output_path) == 3

    def test_mismatched_frame_is_resized_to_writer_size(self, tmp_path: Path) -> None:
        output_path = tmp_path / "check_size.mp4"
        with _VideoOutput(output_path) as video:
            video.write(create_frame(1))
            odd = np.full((FRAME_HEIGHT // 2, FRAME_WIDTH * 2, 3), 80, dtype=np.uint8)
            resized = video._match_writer_size(odd)

        assert resized.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
        # `odd` is a uniform-value array, so resizing (any interpolation method)
        # preserves the constant pixel value exactly, in-memory, with no codec
        # involved — an exact equality check is appropriate here.
        assert np.all(resized == 80)

    def test_size_mismatch_is_warned_once(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        output_path = tmp_path / "warn_once.mp4"
        odd = np.full((FRAME_HEIGHT // 2, FRAME_WIDTH * 2, 3), 80, dtype=np.uint8)

        with caplog.at_level(logging.WARNING, logger="trackers.io.video"):
            with _VideoOutput(output_path) as video:
                video.write(create_frame(1))
                video.write(odd)
                video.write(odd)

        mismatch_warnings = [r for r in caplog.records if "differs from the writer" in r.message]
        assert len(mismatch_warnings) == 1

    def test_revert_to_original_size_is_not_resized_or_rewarned(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """After a mismatch resize+warn, a frame reverting to the writer's size is a no-op."""
        output_path = tmp_path / "revert_size.mp4"
        original = create_frame(1)
        odd = np.full((FRAME_HEIGHT // 2, FRAME_WIDTH * 2, 3), 80, dtype=np.uint8)
        reverted = create_frame(2)  # same size as `original`, distinct pixel value

        with caplog.at_level(logging.WARNING, logger="trackers.io.video"):
            with _VideoOutput(output_path) as video:
                video.write(original)  # opens writer at FRAME_SIZE
                video.write(odd)  # mismatch: resized once, warning logged
                matched = video._match_writer_size(reverted)  # revert to writer's size

        # No resize needed: same object is returned unchanged.
        assert matched is reverted
        # The one-time warning must not fire again for the revert.
        mismatch_warnings = [r for r in caplog.records if "differs from the writer" in r.message]
        assert len(mismatch_warnings) == 1

    def test_none_path_ignores_frame_size(self, caplog: pytest.LogCaptureFixture) -> None:
        # A no-op sink must not track sizes or warn on a resolution change.
        with caplog.at_level(logging.WARNING, logger="trackers.io.video"):
            with _VideoOutput(None) as video:
                assert video.write(create_frame(1)) is True
                odd = np.full((FRAME_HEIGHT // 2, FRAME_WIDTH * 2, 3), 80, dtype=np.uint8)
                assert video.write(odd) is True
        assert video._frame_size is None
        mismatch_warnings = [r for r in caplog.records if "differs from the writer" in r.message]
        assert len(mismatch_warnings) == 0
