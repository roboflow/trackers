# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Helpers shared by every ``trackers inspect`` component.

Only helpers that were byte-identical across the component modules live here.
Rendering helpers such as ``overlay_masks``, ``draw_boxes``, and the text-panel
drawers stay local to each component: they share names but not implementations,
and each draws the state its own component actually has. Unifying them would
silently change what the visualizations look like.

``torch`` is imported inside the functions that need it rather than at module
scope, matching :mod:`trackers.cli.mcbyte`, so that importing the CLI does not
pull in the deep-learning stack.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch

__all__ = [
    "IMAGE_EXTENSIONS",
    "INSPECT_OUTPUT_ROOT",
    "list_selected_frame_paths",
    "load_rgb_image",
    "parse_xyxy_box",
    "print_device_info",
    "save_rgb_image",
    "timestamped_run_dir",
    "validate_device",
]

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
"""Frame file suffixes recognised when listing an image directory."""

INSPECT_OUTPUT_ROOT = Path("outputs/inspect")
"""Base output directory, resolved against the current working directory.

Every component defaults to ``INSPECT_OUTPUT_ROOT / <component>``. The path is
deliberately relative to the working directory rather than to the repository, so
an installed ``trackers`` writes somewhere the caller chose.
"""

_MASK_EXTRA_HINT = "trackers inspect requires the mask extra: pip install 'trackers[mask]'"


def require_torch() -> Any:
    """Import :mod:`torch` on demand and raise a CLI-friendly error if it is missing.

    Returns:
        The imported :mod:`torch` module.

    Raises:
        ImportError: If the ``mask`` extra is not installed.

    Examples:
        >>> module = require_torch()
        >>> module.__name__
        'torch'
    """
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on install extras
        raise ImportError(_MASK_EXTRA_HINT) from error
    return torch


def parse_xyxy_box(box: str) -> tuple[float, float, float, float]:
    """Parse one command-line bounding box in ``x1,y1,x2,y2`` format.

    Args:
        box: Comma-separated box string.

    Returns:
        The box as an ``(x1, y1, x2, y2)`` tuple.

    Raises:
        ValueError: If the string does not hold exactly four values.

    Examples:
        >>> parse_xyxy_box("10,20,110,220")
        (10.0, 20.0, 110.0, 220.0)
    """
    values = [float(value) for value in box.split(",")]
    if len(values) != 4:
        raise ValueError("Each box must contain exactly 4 comma-separated values: x1,y1,x2,y2.")
    return values[0], values[1], values[2], values[3]


def list_selected_frame_paths(
    image_dir: Path,
    start_file: str,
    end_file: str,
) -> list[Path]:
    """List sorted frame paths from ``start_file`` to ``end_file`` inclusive.

    Args:
        image_dir: Directory holding the frames.
        start_file: Filename of the first frame to select.
        end_file: Filename of the last frame to select.

    Returns:
        Frame paths in sorted order, inclusive of both endpoints.

    Raises:
        FileNotFoundError: If either endpoint is absent from ``image_dir``.
        ValueError: If ``end_file`` sorts before ``start_file``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     root = Path(directory)
        ...     _ = [(root / f"{index}.jpg").write_bytes(b"") for index in (1, 2, 3)]
        ...     [path.name for path in list_selected_frame_paths(root, "1.jpg", "2.jpg")]
        ['1.jpg', '2.jpg']
    """
    frame_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    filenames = [path.name for path in frame_paths]

    if start_file not in filenames:
        raise FileNotFoundError(f"Start file not found in {image_dir}: {start_file}")
    if end_file not in filenames:
        raise FileNotFoundError(f"End file not found in {image_dir}: {end_file}")

    start_index = filenames.index(start_file)
    end_index = filenames.index(end_file)
    if end_index < start_index:
        raise ValueError(f"end-file must not come before start-file. Got {start_file=} and {end_file=}.")

    return frame_paths[start_index : end_index + 1]


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image from disk and return it in RGB format.

    Args:
        image_path: Path to the image file.

    Returns:
        RGB image with shape ``(H, W, 3)``.

    Raises:
        FileNotFoundError: If the file cannot be decoded.

    Examples:
        >>> load_rgb_image(Path("missing.jpg"))
        Traceback (most recent call last):
        ...
        FileNotFoundError: Could not read image: missing.jpg
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb_image(
    image_rgb: np.ndarray,
    output_path: Path,
) -> None:
    """Save an RGB image to disk, creating parent directories as needed.

    Args:
        image_rgb: RGB image with shape ``(H, W, 3)``.
        output_path: Destination path.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     target = Path(directory) / "nested" / "frame.jpg"
        ...     save_rgb_image(np.zeros((4, 4, 3), dtype=np.uint8), target)
        ...     target.exists()
        True
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)


def timestamped_run_dir(output_root: Path) -> Path:
    """Create and return a timestamped run directory under ``output_root``.

    Args:
        output_root: Directory holding one subdirectory per run.

    Returns:
        The created run directory.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     run_dir = timestamped_run_dir(Path(directory))
        ...     run_dir.is_dir()
        True
    """
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def validate_device(device: str, label: str | None = None) -> str:
    """Validate that the requested execution device is available.

    Args:
        device: Requested device string, for example ``cpu`` or ``cuda``.
        label: Optional component name used in the error message.

    Returns:
        The validated device string, unchanged.

    Raises:
        RuntimeError: If a CUDA device is requested but CUDA is unavailable.

    Examples:
        >>> validate_device("cpu")
        'cpu'
    """
    torch = require_torch()
    if device.startswith("cuda") and not torch.cuda.is_available():  # type: ignore[attr-defined]
        subject = f" for {label}" if label else ""
        raise RuntimeError(
            f"CUDA was requested{subject}, but torch.cuda.is_available() is False. "
            "Use a CPU device or install a CUDA-enabled PyTorch build."
        )
    return device


def print_device_info(device: torch.device, label: str = "Device") -> None:
    """Print the execution device and GPU name for one component.

    Args:
        device: Resolved torch device.
        label: Component name used as the line prefix.

    Examples:
        >>> torch = require_torch()
        >>> print_device_info(torch.device("cpu"), label="SAM")
        SAM device: cpu
        SAM GPU: N/A (running on CPU)
    """
    torch_module = require_torch()
    print(f"{label} device: {device}")
    if device.type == "cuda":
        name = torch_module.cuda.get_device_name(device)  # type: ignore[attr-defined]
        print(f"{label} GPU: {name} (CUDA {torch_module.version.cuda})")  # type: ignore[attr-defined]
    else:
        print(f"{label} GPU: N/A (running on CPU)")
