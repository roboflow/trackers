# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Video and frame I/O utilities."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from trackers.io.paths import resolve_video_output_path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"})


def frames_from_source(
    source: Union[str, Path, int],
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

    Examples:
        >>> from trackers import frames_from_source
        >>> import cv2
        >>>
        >>> # Video file
        >>> for frame_id, frame in frames_from_source("video.mp4"):  # doctest: +SKIP
        ...     cv2.imshow("frame", frame)
        ...     if cv2.waitKey(1) & 0xFF == ord("q"):
        ...         break
        >>>
        >>> # Webcam
        >>> for frame_id, frame in frames_from_source(0):  # doctest: +SKIP
        ...     cv2.imshow("frame", frame)
        ...     if cv2.waitKey(1) & 0xFF == ord("q"):
        ...         break
        >>>
        >>> # Network stream
        >>> for frame_id, frame in frames_from_source("rtsp://192.168.1.100:554/stream1"):  # doctest: +SKIP
        ...     cv2.imshow("frame", frame)
        ...     if cv2.waitKey(1) & 0xFF == ord("q"):
        ...         break
        >>>
        >>> # MOT17-style image sequence
        >>> for frame_id, frame in frames_from_source("MOT17/train/MOT17-02-FRCNN/img1"):  # doctest: +SKIP
        ...     cv2.imshow("frame", frame)
        ...     if cv2.waitKey(1) & 0xFF == ord("q"):
        ...         break
    """  # noqa: E501
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        yield from _iter_image_folder_frames(Path(source))
    else:
        yield from _iter_capture_frames(source)


def _iter_capture_frames(
    src: Union[str, int, Path],
) -> Iterator[tuple[int, np.ndarray]]:
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


class VideoOutput:
    """Context manager for lazy video file writing."""

    def __init__(self, path: Path | None):
        self.path = path
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray) -> None:
        """Write a frame to the video file. Initializes writer on first call."""
        if self.path is None:
            return
        if self._writer is None:
            self._writer = self._create_writer(frame, self.path)
        self._writer.write(frame)

    def _create_writer(self, frame: np.ndarray, path: Path) -> cv2.VideoWriter:
        resolved = resolve_video_output_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        return cv2.VideoWriter(str(resolved), fourcc, 30.0, (w, h))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self._writer is not None:
            self._writer.release()


class DisplayWindow:
    """Context manager for OpenCV display window."""

    def __init__(self, window_name: str = "Tracking"):
        self.window_name = window_name
        self._quit_requested = False

    def show(self, frame: np.ndarray) -> None:
        """Display a frame and check for quit key."""
        cv2.imshow(self.window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._quit_requested = True

    @property
    def quit_requested(self) -> bool:
        """Return True if user pressed 'q' to quit."""
        return self._quit_requested

    def __enter__(self):
        return self

    def __exit__(self, *_):
        cv2.destroyAllWindows()
