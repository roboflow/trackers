# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import time
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from rich.console import Console

from trackers.scripts.progress import (
    _classify_source,
    _format_time,
    _SourceInfo,
    _TrackingProgress,
)

FRAME_WIDTH = 64
FRAME_HEIGHT = 64
FRAME_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)


@pytest.fixture
def video_factory(tmp_path: Path) -> Callable[[int], Path]:
    """Create a small test video with *n* frames."""

    def _create(n_frames: int) -> Path:
        video_path = tmp_path / f"video_{n_frames}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 25.0, FRAME_SIZE)
        for _ in range(n_frames):
            writer.write(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
        writer.release()
        return video_path

    return _create


@pytest.fixture
def image_directory_factory(tmp_path: Path) -> Callable[[int], Path]:
    """Create a directory with *n* PNG images."""

    def _create(n_frames: int) -> Path:
        directory = tmp_path / f"imgdir_{n_frames}"
        directory.mkdir(exist_ok=True)
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        for i in range(n_frames):
            cv2.imwrite(str(directory / f"{i:04d}.png"), frame)
        return directory

    return _create


def _make_console() -> tuple[Console, StringIO]:
    """Return a Console that writes to a StringIO buffer."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=200)
    return console, buf


class TestClassifySource:
    def test_video_file(self, video_factory: Callable[[int], Path]) -> None:
        video_path = video_factory(10)
        info = _classify_source(str(video_path))

        assert info.source_type == "video"
        assert info.total_frames is not None
        assert info.total_frames > 0
        assert info.fps is not None
        assert info.fps > 0

    def test_image_directory(
        self, image_directory_factory: Callable[[int], Path]
    ) -> None:
        directory = image_directory_factory(7)
        info = _classify_source(str(directory))

        assert info.source_type == "image_dir"
        assert info.total_frames == 7
        assert info.fps is None

    def test_image_directory_path_object(
        self, image_directory_factory: Callable[[int], Path]
    ) -> None:
        directory = image_directory_factory(3)
        info = _classify_source(directory)

        assert info.source_type == "image_dir"
        assert info.total_frames == 3

    def test_webcam_from_int(self) -> None:
        info = _classify_source(0)

        assert info.source_type == "webcam"
        assert info.total_frames is None
        assert info.fps is None

    def test_webcam_from_str(self) -> None:
        info = _classify_source("0")

        assert info.source_type == "webcam"
        assert info.total_frames is None

    @pytest.mark.parametrize(
        "url",
        [
            "rtsp://192.168.1.10:554/stream",
            "http://example.com/stream.mjpg",
            "https://example.com/stream.mjpg",
        ],
    )
    def test_stream_url(self, url: str) -> None:
        info = _classify_source(url)

        assert info.source_type == "stream"
        assert info.total_frames is None
        assert info.fps is None

    def test_video_with_zero_frame_count(self) -> None:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 0.0,
            cv2.CAP_PROP_FPS: 30.0,
        }.get(prop, 0.0)

        with patch("trackers.scripts.progress.cv2.VideoCapture", return_value=mock_cap):
            info = _classify_source("some_video.mp4")

        assert info.source_type == "video"
        assert info.total_frames is None
        mock_cap.release.assert_called_once()

    def test_nonexistent_file(self) -> None:
        info = _classify_source("/nonexistent/video.mp4")

        assert info.source_type == "video"
        assert info.total_frames is None

    def test_empty_image_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        info = _classify_source(str(empty_dir))

        assert info.source_type == "image_dir"
        assert info.total_frames is None


class TestFormatTime:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "0:00"),
            (5, "0:05"),
            (65, "1:05"),
            (3661, "1:01:01"),
            (-1, "--"),
        ],
    )
    def test_format_time(self, seconds: float, expected: str) -> None:
        assert _format_time(seconds) == expected


class TestBuildLine:
    def test_bounded_format(self) -> None:
        console, _ = _make_console()
        source_info = _SourceInfo(source_type="video", total_frames=100)
        progress = _TrackingProgress(source_info, console=console)
        progress._start_time = time.monotonic() - 5.0
        progress._frames_processed = 50

        line = progress._build_line("⠹")
        text = line.plain

        assert "50 / 100" in text
        assert "frames" in text
        assert "50%" in text
        assert "fps" in text
        assert "elapsed" in text
        assert "eta" in text

    def test_unbounded_format(self) -> None:
        console, _ = _make_console()
        source_info = _SourceInfo(source_type="webcam")
        progress = _TrackingProgress(source_info, console=console)
        progress._start_time = time.monotonic() - 5.0
        progress._frames_processed = 50

        line = progress._build_line("⠹")
        text = line.plain

        assert "50 / --" in text
        assert "frames" in text
        assert "--" in text
        assert "fps" in text
        assert "elapsed" in text
        assert "eta --" in text

    def test_final_no_eta(self) -> None:
        console, _ = _make_console()
        source_info = _SourceInfo(source_type="video", total_frames=100)
        progress = _TrackingProgress(source_info, console=console)
        progress._start_time = time.monotonic() - 5.0
        progress._frames_processed = 100

        line = progress._build_line("✓", show_eta=False)
        text = line.plain

        assert "eta" not in text
        assert "✓" in text

    def test_suffix_appended(self) -> None:
        console, _ = _make_console()
        source_info = _SourceInfo(source_type="video", total_frames=100)
        progress = _TrackingProgress(source_info, console=console)
        progress._start_time = time.monotonic() - 5.0
        progress._frames_processed = 50

        line = progress._build_line("✗", show_eta=False, suffix="(interrupted)")
        text = line.plain

        assert text.endswith("(interrupted)")

    def test_zero_elapsed_no_crash(self) -> None:
        console, _ = _make_console()
        source_info = _SourceInfo(source_type="video", total_frames=100)
        progress = _TrackingProgress(source_info, console=console)
        progress._start_time = time.monotonic()
        progress._frames_processed = 0

        # Should not raise ZeroDivisionError
        line = progress._build_line("⠹")
        text = line.plain

        assert "fps" in text


class TestTrackingProgressLifecycle:
    def test_bounded_completed(self) -> None:
        console, buf = _make_console()
        source_info = _SourceInfo(source_type="video", total_frames=5)

        with _TrackingProgress(source_info, console=console) as progress:
            for _ in range(5):
                progress.update()
            progress.complete()

        output = buf.getvalue()
        assert "✓" in output
        assert "(interrupted)" not in output

    def test_bounded_interrupted_by_display_quit(self) -> None:
        console, buf = _make_console()
        source_info = _SourceInfo(source_type="video", total_frames=10)

        with _TrackingProgress(source_info, console=console) as progress:
            for _ in range(5):
                progress.update()
            progress.complete(interrupted=True)

        output = buf.getvalue()
        assert "✗" in output
        assert "(interrupted)" in output

    def test_bounded_keyboard_interrupt(self) -> None:
        console, buf = _make_console()
        source_info = _SourceInfo(source_type="video", total_frames=10)

        progress = _TrackingProgress(source_info, console=console)
        progress.__enter__()
        for _ in range(3):
            progress.update()

        # Simulate KeyboardInterrupt in __exit__
        progress.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None)

        output = buf.getvalue()
        assert "✗" in output
        assert "(interrupted)" in output

    def test_unbounded_completed(self) -> None:
        console, buf = _make_console()
        source_info = _SourceInfo(source_type="webcam")

        with _TrackingProgress(source_info, console=console) as progress:
            for _ in range(20):
                progress.update()
            progress.complete()

        output = buf.getvalue()
        assert "✓" in output

    def test_unbounded_keyboard_interrupt(self) -> None:
        console, buf = _make_console()
        source_info = _SourceInfo(source_type="stream")

        progress = _TrackingProgress(source_info, console=console)
        progress.__enter__()
        for _ in range(10):
            progress.update()

        progress.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None)

        output = buf.getvalue()
        assert "✓" in output
        assert "(interrupted)" not in output

    def test_error_shows_source_lost(self) -> None:
        console, buf = _make_console()
        source_info = _SourceInfo(source_type="stream")

        progress = _TrackingProgress(source_info, console=console)
        progress.__enter__()
        for _ in range(5):
            progress.update()

        err = RuntimeError("connection lost")
        progress.__exit__(RuntimeError, err, None)

        output = buf.getvalue()
        assert "✗" in output
        assert "(source lost)" in output

    def test_frames_count_in_output(self) -> None:
        console, buf = _make_console()
        source_info = _SourceInfo(source_type="image_dir", total_frames=30)

        with _TrackingProgress(source_info, console=console) as progress:
            for _ in range(30):
                progress.update()
            progress.complete()

        output = buf.getvalue()
        assert "30 / 30" in output

    def test_unbounded_frames_count_in_output(self) -> None:
        console, buf = _make_console()
        source_info = _SourceInfo(source_type="webcam")

        with _TrackingProgress(source_info, console=console) as progress:
            for _ in range(42):
                progress.update()
            progress.complete()

        output = buf.getvalue()
        assert "42 / --" in output
