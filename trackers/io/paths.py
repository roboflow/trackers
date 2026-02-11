# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Path validation and resolution utilities."""

from __future__ import annotations

from pathlib import Path


def _resolve_video_output_path(path: Path) -> Path:
    """Resolve video output path, handling directories.

    If path is an existing directory, generates 'output.mp4' inside it.
    If path has no extension, adds '.mp4'.
    """
    if path.is_dir():
        return path / "output.mp4"
    if not path.suffix:
        return path.with_suffix(".mp4")
    return path


def _validate_output_path(path: Path, *, overwrite: bool = False) -> None:
    """Validate output path and create parent directories if needed.

    Args:
        path: Output file path to validate.
        overwrite: If True, allow overwriting existing files.

    Raises:
        FileExistsError: If path exists and overwrite is False.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file '{path}' already exists. Use --overwrite to replace."
        )
