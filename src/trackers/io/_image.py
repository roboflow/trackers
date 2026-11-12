# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Image file reading and writing that also works for non-ASCII paths on Windows."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def _is_non_ascii_windows_path(path: str | Path) -> bool:
    """Return whether OpenCV would read the path in the ANSI code page instead of Unicode."""
    return os.name == "nt" and not str(path).isascii()


def _read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image like ``cv2.imread``, also from non-ASCII paths on Windows.

    Args:
        path: Image file to read.
        flags: OpenCV read flags.

    Returns:
        The decoded image, or ``None`` when the file is missing or cannot be decoded.
    """
    if not _is_non_ascii_windows_path(path):
        return cv2.imread(str(path), flags)
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, flags)


def _write_image(path: str | Path, image: np.ndarray) -> bool:
    """Write an image like ``cv2.imwrite``, also to non-ASCII paths on Windows.

    Args:
        path: Destination file. Its extension selects the format.
        image: Image to encode.

    Returns:
        Whether the file was written.
    """
    if not _is_non_ascii_windows_path(path):
        return bool(cv2.imwrite(str(path), image))
    success, buffer = cv2.imencode(Path(path).suffix, image)
    if not success:
        return False
    try:
        buffer.tofile(str(path))
    except OSError:
        return False
    return True
