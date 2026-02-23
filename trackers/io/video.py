# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from trackers.io.paths import _resolve_video_output_path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"})
_DEFAULT_OUTPUT_FPS = 30.0


def frames_from_source(
    source: str | Path | int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield numbered BGR frames from video files, webcams, network streams, or image
    directories.

    Args:
        source: Video file path, RTSP/HTTP stream URL, webcam index, or path to a
            directory containing images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`,
            `.tiff`).

    Returns:
        Iterator of `(frame_id, frame)` tuples where `frame_id` is 1-based and `frame`
        is a `np.ndarray` in BGR format.

    Raises:
        ValueError: Source cannot be opened or directory contains no supported images.
        OSError: Image file exists but cannot be decoded / read.
        RuntimeError: Capture read failure after successful open.
    """
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        yield from _iter_image_folder_frames(Path(source))
    else:
        yield from _iter_capture_frames(source)


def _iter_capture_frames(
    src: str | int | Path,
) -> Iterator[tuple[int, np.ndarray]]:
    # Convert numeric strings to int for webcam indices
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(str(src) if isinstance(src, Path) else src)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video/capture source: {src!r}")

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            yield frame_id, frame
    except Exception as e:
        raise RuntimeError(f"Failed while reading from capture source {src!r}") from e
    finally:
        cap.release()


def _iter_image_folder_frames(
    folder: Path,
    *,
    extensions: frozenset[str] = IMAGE_EXTENSIONS,
) -> Iterator[tuple[int, np.ndarray]]:
    images = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )

    if not images:
        raise ValueError(f"No supported image files found in directory: {folder}")

    for frame_id, path in enumerate(images, start=1):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"Failed to read image: {path}")
        yield frame_id, frame


class _VideoOutput:
    """Context manager for lazy video file writing."""

    def __init__(self, path: Path | None, *, fps: float = _DEFAULT_OUTPUT_FPS):
        self.path = path
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray) -> bool:
        """Write a frame to the video file. Initializes writer on first call.

        Returns:
            True if write succeeded or path is None, False on failure.
        """
        if self.path is None:
            return True
        if self._writer is None:
            self._writer = self._create_writer(frame)
            if self._writer is None:
                return False
        self._writer.write(frame)
        return True

    def _create_writer(self, frame: np.ndarray) -> cv2.VideoWriter | None:
        if self.path is None:
            return None

        resolved = _resolve_video_output_path(self.path)
        resolved.parent.mkdir(parents=True, exist_ok=True)

        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(resolved), fourcc, self.fps, (width, height))

        if not writer.isOpened():
            raise OSError(f"Failed to open video writer for '{resolved}'")

        return writer

    def __enter__(self) -> _VideoOutput:
        return self

    def __exit__(self, *_: object) -> None:
        if self._writer is not None:
            self._writer.release()


class _DisplayWindow:
    """Context manager for OpenCV display window with resizable output."""

    def __init__(self, window_name: str = "Tracking"):
        self.window_name = window_name
        self._quit_requested = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    def show(self, frame: np.ndarray) -> bool:
        """Display a frame and check for quit key (q or ESC).

        Returns:
            True if quit was requested, False otherwise.
        """
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            self._quit_requested = True
        return self._quit_requested

    @property
    def quit_requested(self) -> bool:
        """Return True if user pressed quit key."""
        return self._quit_requested

    def __enter__(self):
        return self

    def __exit__(self, *_):
        cv2.destroyWindow(self.window_name)
