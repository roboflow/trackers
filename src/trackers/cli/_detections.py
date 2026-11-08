# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Detection-file reading shared by ``benchmark`` and ``inspect``.

Both command groups replay pre-computed detections through a tracker, so both need the same parser, the same frame
lookup, and the same frame loader. They had their own copies, which had already drifted apart in error wording and in
the set of image extensions they accepted.

Not merged into :mod:`trackers.io.mot`: ``load_mot_file`` there returns evaluation-shaped ``_MOTFrameData`` — boxes as
``xywh``, plus the class and visibility columns that ground-truth filtering needs — and handles one layout. What these
commands need is a detection record across two layouts. Same file extension, different contract.

Infrastructure module, not a command — see :mod:`trackers.cli` for the layout rule.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import supervision as sv

__all__ = [
    "IMAGE_EXTENSIONS",
    "DetectionFileFormat",
    "DetectionRecord",
    "build_detections",
    "find_frame_path",
    "load_rgb_frame",
    "read_detection_file",
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
"""Frame file suffixes tried when locating a frame by number."""

DetectionFileFormat = Literal["mot_tlwh", "mot", "xyxy"]
"""Detection-file column layout.

- ``mot_tlwh`` (spelled ``mot`` by the benchmark datasets):
  ``frame,id,left,top,width,height,confidence,...`` — the identity column is
  ignored, since tracker identities are what these commands produce.
- ``xyxy``:
  ``frame,x1,y1,x2,y2,confidence``
"""

_FRAME_NUMBER_WIDTHS = (6, 8)

# Per-directory memo of the (width, extension) pair that last matched.
#
# A sequence directory names every frame the same way, so the pair that matched
# the previous frame matches the next one. Trying it first costs one is_file
# call in the steady state instead of the ten the full search can reach; a miss
# just falls through to that search, so the memo can never change which path is
# returned. Bounded by the number of sequence directories one command touches.
_FRAME_LAYOUT_HINTS: dict[Path, tuple[int, str]] = {}


@dataclass(frozen=True)
class DetectionRecord:
    """One detection parsed from a detection file."""

    xyxy: np.ndarray
    confidence: float


def read_detection_file(
    det_file: Path,
    detection_format: DetectionFileFormat,
    confidence_override: float | None = None,
) -> dict[int, list[DetectionRecord]]:
    """Read detections from ``det_file`` and group them by frame number.

    Blank lines are skipped. Boxes with non-positive width or height are dropped
    rather than raising: a malformed box is common enough in these files that
    failing a whole sequence over one of them would be worse than continuing.

    Args:
        det_file: Detection ``.txt`` file to parse.
        detection_format: Column layout, see :data:`DetectionFileFormat`.
        confidence_override: Replaces the parsed confidence column when set.
            Used for SoccerNet, whose confidence column is not meaningful.

    Returns:
        Detections grouped by 1-based frame number. Frames left with no valid
        detections are absent rather than present and empty.

    Raises:
        ValueError: If a line has too few columns for the format, a numeric
            column fails to parse, or a frame number is non-positive.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     path = Path(directory) / "det.txt"
        ...     _ = path.write_text("1,10,20,110,220,0.9\\n")
        ...     frames = read_detection_file(path, "xyxy")
        >>> sorted(frames), frames[1][0].confidence
        ([1], 0.9)
    """
    is_tlwh = detection_format in ("mot_tlwh", "mot")
    minimum_columns = 7 if is_tlwh else 6
    grouped: dict[int, list[DetectionRecord]] = defaultdict(list)

    with det_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            values = [value.strip() for value in line.split(",")]
            if len(values) < minimum_columns:
                raise ValueError(
                    f"Detection format {detection_format!r} requires at least {minimum_columns} "
                    f"columns. Invalid line {line_number} in {det_file}: {line}"
                )

            try:
                frame_number = int(float(values[0]))
                if is_tlwh:
                    left, top, width, height = (float(value) for value in values[2:6])
                    x1, y1, x2, y2 = left, top, left + width, top + height
                    confidence = float(values[6])
                else:
                    x1, y1, x2, y2 = (float(value) for value in values[1:5])
                    confidence = float(values[5])
            except ValueError as error:
                raise ValueError(
                    f"Could not parse detection line {line_number} in {det_file} "
                    f"using format {detection_format!r}: {line}"
                ) from error

            if frame_number <= 0:
                raise ValueError(f"Non-positive frame number on line {line_number} in {det_file}.")
            if x2 <= x1 or y2 <= y1:
                continue

            grouped[frame_number].append(
                DetectionRecord(
                    xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                    confidence=float(confidence if confidence_override is None else confidence_override),
                )
            )

    return dict(grouped)


def build_detections(records: list[DetectionRecord]) -> sv.Detections:
    """Convert parsed records into ``sv.Detections``.

    Args:
        records: Detections for one frame, as produced by
            :func:`read_detection_file`.

    Returns:
        Detections with ``xyxy`` and ``confidence`` populated, or
        ``sv.Detections.empty()`` when ``records`` is empty.

    Examples:
        >>> len(build_detections([]))
        0
    """
    if not records:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.stack([record.xyxy for record in records]),
        confidence=np.asarray([record.confidence for record in records], dtype=np.float32),
    )


def find_frame_path(image_dir: Path, frame_number: int) -> Path:
    """Locate a frame by number, trying the common MOT filename widths.

    Tries 6- and 8-digit zero-padded stems against every suffix in
    :data:`IMAGE_EXTENSIONS`, after first retrying whichever combination last
    matched in ``image_dir`` (see :data:`_FRAME_LAYOUT_HINTS`).

    Args:
        image_dir: Sequence frame directory to search.
        frame_number: 1-based frame number to locate.

    Returns:
        Path to the matching frame file.

    Raises:
        FileNotFoundError: If no combination matches a file in ``image_dir``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     root = Path(directory)
        ...     _ = (root / "000007.jpg").write_bytes(b"")
        ...     find_frame_path(root, 7).name
        '000007.jpg'
    """
    hint = _FRAME_LAYOUT_HINTS.get(image_dir)
    if hint is not None:
        hinted_width, hinted_extension = hint
        frame_path = image_dir / f"{frame_number:0{hinted_width}d}{hinted_extension}"
        if frame_path.is_file():
            return frame_path

    for width in _FRAME_NUMBER_WIDTHS:
        stem = f"{frame_number:0{width}d}"
        for extension in IMAGE_EXTENSIONS:
            frame_path = image_dir / f"{stem}{extension}"
            if frame_path.is_file():
                _FRAME_LAYOUT_HINTS[image_dir] = (width, extension)
                return frame_path

    stems = [f"{frame_number:0{width}d}" for width in _FRAME_NUMBER_WIDTHS]
    attempted = [f"{stem}{extension}" for stem in stems for extension in IMAGE_EXTENSIONS]
    raise FileNotFoundError(f"Could not find frame {frame_number} in {image_dir}. Tried: {attempted}")


def load_rgb_frame(frame_path: Path) -> np.ndarray:
    """Load one frame and convert OpenCV BGR channels to RGB.

    SAM and Cutie expect RGB input while OpenCV decodes BGR, so the conversion
    happens once at load time.

    Args:
        frame_path: Path to the frame image file.

    Returns:
        The frame as an RGB array.

    Raises:
        RuntimeError: If the image cannot be decoded.

    Examples:
        >>> load_rgb_frame(Path("missing.jpg"))
        Traceback (most recent call last):
        ...
        RuntimeError: cv2.imread failed for frame: missing.jpg
    """
    import cv2

    frame_bgr = cv2.imread(str(frame_path))
    if frame_bgr is None:
        raise RuntimeError(f"cv2.imread failed for frame: {frame_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
