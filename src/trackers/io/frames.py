# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Load MOT benchmark frames from ``img1/`` directories."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MOT_FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# MOT Challenge uses 6-digit stems; DanceTrack uses 8-digit. Try both, then plain int.
MOT_FRAME_STEM_WIDTHS = (6, 8)


def mot_frame_stems(frame_idx: int) -> tuple[str, ...]:
    """Return filename stems to probe for ``frame_idx`` (most common layouts first)."""
    stems = [f"{frame_idx:0{width}d}" for width in MOT_FRAME_STEM_WIDTHS]
    plain = str(frame_idx)
    if plain not in stems:
        stems.append(plain)
    return tuple(stems)


def resolve_mot_frame_path(frame_dir: Path, frame_idx: int) -> Path | None:
    """Return the first existing image path for ``frame_idx`` under ``frame_dir``.

    Args:
        frame_dir: Directory containing frame images (typically ``…/img1/``).
        frame_idx: 1-based frame index from MOT detection files.

    Returns:
        Resolved path, or ``None`` when no matching file exists.
    """
    for stem in mot_frame_stems(frame_idx):
        for ext in MOT_FRAME_EXTENSIONS:
            path = frame_dir / f"{stem}{ext}"
            if path.is_file():
                return path
    return None


def load_mot_frame_image(frame_dir: Path, frame_idx: int) -> np.ndarray:
    """Load one BGR frame from a MOT ``img1`` directory.

    Raises:
        FileNotFoundError: When no file matches any supported stem/extension.
        OSError: When a matching file exists but cannot be decoded.
    """
    path = resolve_mot_frame_path(frame_dir, frame_idx)
    if path is None:
        stems = ", ".join(mot_frame_stems(frame_idx))
        extensions = ", ".join(MOT_FRAME_EXTENSIONS)
        raise FileNotFoundError(
            f"MOT frame image not found for frame {frame_idx} under {frame_dir}: "
            f"tried stems ({stems}) with extensions ({extensions})"
        )
    frame = cv2.imread(str(path))
    if frame is None:
        raise OSError(f"Failed to decode image: {path}")
    return frame
