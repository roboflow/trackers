# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from trackers.io.paths import _resolve_video_output_path

logger = logging.getLogger("trackers.io.video")

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
    images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions)

    if not images:
        raise ValueError(f"No supported image files found in directory: {folder}")

    for frame_id, path in enumerate(images, start=1):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"Failed to read image: {path}")
        yield frame_id, frame


class _VideoOutput:
    """Context manager for lazy video file writing."""

    def __init__(self, path: Path | None, *, fps: float = _DEFAULT_OUTPUT_FPS) -> None:
        self.path = path
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._frame_size: tuple[int, int] | None = None
        self._size_mismatch_logged = False

    def write(self, frame: np.ndarray) -> bool:
        """Write a frame to the video file. Initializes writer on first call.

        The writer is bound to the first frame's resolution, and a video file
        cannot hold frames of mixed sizes. A later frame of a different size
        (e.g. a mid-stream resolution change from an RTSP renegotiation) is
        resized to the writer's resolution so it is kept in the output stream
        rather than being silently dropped by the codec. This guarantee only
        covers height/width mismatches: the writer is opened with
        `isColor=True` and channel count is not reconciled, so a mid-stream
        frame with a different number of channels (e.g. single-channel
        grayscale) can still be silently dropped by the codec.

        Returns:
            True if write succeeded or path is None, False on failure.
        """
        if self.path is None:
            return True
        if self._writer is None:
            self._writer = self._create_writer(frame)
            if self._writer is None:
                return False
        frame = self._match_writer_size(frame)
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

        self._frame_size = (width, height)
        return writer

    def _match_writer_size(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame to the writer's resolution, warning once on mismatch.

        `cv2.VideoWriter.write` silently discards frames whose size differs from
        the one the writer was opened with, so a resolution change mid-stream
        would drop frames from the output without any error. Resizing keeps the
        stream contiguous; the mismatch is logged once to avoid per-frame spam.
        The resize stretches the frame to the writer's exact dimensions without
        preserving the source aspect ratio (aspect ratio may be distorted).
        """
        height, width = frame.shape[:2]
        if self._frame_size is None or self._frame_size == (width, height):
            return frame
        if not self._size_mismatch_logged:
            logger.warning(
                "Video frame size %s differs from the writer size %s; resizing "
                "to keep the output stream contiguous (aspect ratio may be "
                "distorted).",
                (width, height),
                self._frame_size,
            )
            self._size_mismatch_logged = True
        return cv2.resize(frame, self._frame_size)

    def __enter__(self) -> _VideoOutput:
        return self

    def __exit__(self, *_: object) -> None:
        if self._writer is not None:
            self._writer.release()


class _DisplayWindow:
    """Context manager for OpenCV display window with resizable output."""

    def __init__(self, window_name: str = "Tracking") -> None:
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

    def __enter__(self) -> _DisplayWindow:
        return self

    def __exit__(self, *_: object) -> None:
        cv2.destroyWindow(self.window_name)
