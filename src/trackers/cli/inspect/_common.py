# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Helpers shared by every ``trackers inspect`` component.

What lives where, so that no component keeps a private copy of a shared helper:

- here: input parsing, frame listing, image I/O, run directories, and the
  mask-output accessors that every component reaches for;
- :mod:`trackers.cli._annotate`: the drawing helpers, because the ``benchmark``
  commands render the same overlays and must not drift from ``inspect``;
- :mod:`trackers.utils.device`: device selection and validation, which are not
  CLI-specific. ``validate_device`` is re-exported here because every component
  calls it.

A helper stays local to one component only when its behaviour genuinely differs
there, not merely because the name is repeated.

``torch`` is imported inside the functions that need it rather than at module
scope, matching :mod:`trackers.cli.benchmark.mcbyte`, so that importing the CLI does not
pull in the deep-learning stack.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from trackers.core.masks.base import MaskOutput, TrackletSnapshot

# Device validation is not CLI-specific, so it lives with the other device
# helpers. Re-exported here because every inspect component reaches for it.
from trackers.utils.device import _validate_device as validate_device

if TYPE_CHECKING:
    import torch

__all__ = [
    "IMAGE_EXTENSIONS",
    "INSPECT_OUTPUT_ROOT",
    "get_mask_tracklet_ids_in_order",
    "get_masked_tracklet_ids",
    "list_selected_frame_paths",
    "load_rgb_image",
    "parse_xyxy_box",
    "print_device_info",
    "require_torch",
    "save_rgb_image",
    "timestamped_run_dir",
    "tracklet_boxes",
    "validate_and_clip_xyxy_box",
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


def get_mask_tracklet_ids_in_order(tracklet_mask_dict: dict[int, int]) -> list[int]:
    """Return tracklet IDs in the same order as ``MaskOutput.masks``.

    Args:
        tracklet_mask_dict: Mapping of tracklet ID to its row in ``masks``.

    Returns:
        Tracklet IDs ordered by mask row, so they line up with the mask array.

    Examples:
        >>> get_mask_tracklet_ids_in_order({7: 1, 3: 0})
        [3, 7]
    """
    return [tracklet_id for tracklet_id, _ in sorted(tracklet_mask_dict.items(), key=lambda item: item[1])]


def get_masked_tracklet_ids(mask_output: MaskOutput | None) -> set[int]:
    """Return tracklet IDs currently represented in a mask output.

    Args:
        mask_output: Masks for one frame, or ``None`` when none were produced.

    Returns:
        The tracklet IDs holding a mask, empty when there are none.

    Examples:
        >>> get_masked_tracklet_ids(None)
        set()
    """
    if mask_output is None or mask_output.masks is None:
        return set()
    return set(mask_output.tracklet_mask_dict)


def validate_and_clip_xyxy_box(
    box: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Validate and clip an ``xyxy`` box to image boundaries.

    Args:
        box: Bounding box in ``(x1, y1, x2, y2)`` format.
        image_shape: Image shape as ``(height, width)``.

    Returns:
        Clipped box as a float32 NumPy array.

    Raises:
        ValueError: If the box has non-positive area before or after clipping.

    Examples:
        >>> validate_and_clip_xyxy_box((1.0, 2.0, 5.0, 6.0), (10, 10))
        array([1., 2., 5., 6.], dtype=float32)
    """
    height, width = image_shape
    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid xyxy box with non-positive size: {box}")

    x1 = np.clip(x1, 0, width)
    x2 = np.clip(x2, 0, width)
    y1 = np.clip(y1, 0, height)
    y2 = np.clip(y2, 0, height)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Box is outside image after clipping: {box}")

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def tracklet_boxes(tracklets: list[TrackletSnapshot]) -> np.ndarray:
    """Stack tracklet boxes into an ``(N, 4)`` array for the annotators.

    Args:
        tracklets: Tracklets to stack.

    Returns:
        Boxes with shape ``(N, 4)``, or ``(0, 4)`` when ``tracklets`` is empty.

    Examples:
        >>> tracklet_boxes([]).shape
        (0, 4)
    """
    if not tracklets:
        return np.zeros((0, 4), dtype=float)
    return np.stack([tracklet.xyxy for tracklet in tracklets])


def parse_xyxy_box(box: str) -> tuple[float, float, float, float]:
    """Parse one command-line bounding box in ``x1,y1,x2,y2`` format.

    Args:
        box: Comma-separated box string.

    Returns:
        The box as an ``(x1, y1, x2, y2)`` tuple.

    Raises:
        ValueError: If the string does not hold exactly four numbers.

    Examples:
        >>> parse_xyxy_box("10,20,110,220")
        (10.0, 20.0, 110.0, 220.0)
        >>> parse_xyxy_box("10,20,110,left")
        Traceback (most recent call last):
        ...
        ValueError: Each box must contain exactly 4 comma-separated numbers: x1,y1,x2,y2. Got '10,20,110,left'.
    """
    message = f"Each box must contain exactly 4 comma-separated numbers: x1,y1,x2,y2. Got {box!r}."

    # A non-numeric token is the same user mistake as the wrong count, so it is
    # reported the same way. Letting float() surface its own "could not convert
    # string to float" would name neither the option nor the expected format.
    try:
        values = [float(value) for value in box.split(",")]
    except ValueError as error:
        raise ValueError(message) from error

    if len(values) != 4:
        raise ValueError(message)
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

    Raises:
        RuntimeError: If OpenCV declines to encode or write the file, which it
            reports by returning ``False`` rather than raising.

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
    if not cv2.imwrite(str(output_path), image_bgr):
        raise RuntimeError(f"Could not save visualization: {output_path}")


def timestamped_run_dir(output_root: Path) -> Path:
    """Create and return a timestamped run directory under ``output_root``.

    The timestamp carries microseconds and the directory is created without
    ``exist_ok``, so two runs started within the same second cannot land in one
    directory and merge their outputs. The name stays sortable by start time.

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
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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
