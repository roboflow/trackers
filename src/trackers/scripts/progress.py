#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
from rich.console import Console
from rich.live import Live
from rich.text import Text

from trackers.io.video import IMAGE_EXTENSIONS

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_STREAM_PREFIXES = ("rtsp://", "http://", "https://")
_ICON_OK = "✓"
_ICON_FAIL = "✗"


@dataclass
class _SourceInfo:
    """Metadata about a frame source used to drive progress display.

    Attributes:
        source_type: Kind of source (`video`, `image_dir`, `webcam`,
            `stream`).
        total_frames: Total frame count when known, `None` for unbounded
            sources such as webcams and network streams.
        fps: Source frame-rate when known, `None` otherwise.
    """

    source_type: Literal["video", "image_dir", "webcam", "stream"]
    total_frames: int | None = None
    fps: float | None = None


def _classify_source(source: str | Path | int) -> _SourceInfo:
    """Classify a frame source and extract metadata.

    The function inspects *source* without consuming any frames so it can be
    called before the main processing loop.

    Args:
        source: The same value accepted by `frames_from_source`.

    Returns:
        A `_SourceInfo` describing the source.
    """
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return _SourceInfo(source_type="webcam")

    source_str = str(source)

    if any(source_str.lower().startswith(p) for p in _STREAM_PREFIXES):
        return _SourceInfo(source_type="stream")

    path = Path(source_str)
    if path.is_dir():
        count = sum(
            1
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        return _SourceInfo(
            source_type="image_dir",
            total_frames=count if count > 0 else None,
        )

    cap = cv2.VideoCapture(source_str)
    if not cap.isOpened():
        # Cannot open; still classify as video - the real error will come
        # from frames_from_source later.
        return _SourceInfo(source_type="video")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return _SourceInfo(
            source_type="video",
            total_frames=total if total > 0 else None,
            fps=fps if fps and fps > 0 else None,
        )
    finally:
        cap.release()


def _format_time(seconds: float) -> str:
    """Format `seconds` as `H:MM:SS` or `M:SS`."""
    if seconds < 0:
        return "--"
    minutes, seconds_remainder = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds_remainder:02d}"
    return f"{minutes}:{seconds_remainder:02d}"


class _TrackingProgress:
    """Context-manager that renders a single live progress line.

    Args:
        source_info: Source metadata returned by `_classify_source`.
        console: Optional `Console` instance (useful for testing with a
            `StringIO` file).
    """

    def __init__(
        self,
        source_info: _SourceInfo,
        console: Console | None = None,
    ) -> None:
        self._source_info = source_info
        self._console = console or Console()
        self._frames_processed = 0
        self._start_time: float = 0.0
        self._spinner = itertools.cycle(_SPINNER_FRAMES)
        self._live: Live | None = None
        self._interrupted = False

    def update(self) -> None:
        """Record one processed frame and refresh the display."""
        self._frames_processed += 1
        icon = next(self._spinner)
        if self._live is not None:
            self._live.update(self._build_line(icon))

    def complete(self, *, interrupted: bool = False) -> None:
        """Signal that the processing loop has ended.

        Must be called before leaving the `with` block so that `__exit__`
        can render the correct final state.

        Args:
            interrupted: `True` when the loop was terminated early (e.g.
                display-quit).
        """
        self._interrupted = interrupted

    def __enter__(self) -> _TrackingProgress:
        self._start_time = time.monotonic()
        self._live = Live(
            self._build_line("⠋"),
            console=self._console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._live is not None:
            self._live.__exit__(None, None, None)

        icon, suffix = self._resolve_final_state(exc_type)
        final = self._build_line(icon, show_eta=False, suffix=suffix)
        self._console.print(final)

    @property
    def _is_bounded(self) -> bool:
        """Whether the source has a known total frame count."""
        return self._source_info.total_frames is not None

    def _resolve_final_state(
        self, exc_type: type[BaseException] | None
    ) -> tuple[str, str]:
        """Return `(icon, suffix)` for the final printed line."""
        is_real_error = exc_type is not None and not issubclass(
            exc_type, KeyboardInterrupt
        )

        if is_real_error:
            return (_ICON_FAIL, "(source lost)")

        was_stopped_early = exc_type is not None or self._interrupted

        if was_stopped_early and self._is_bounded:
            return (_ICON_FAIL, "(interrupted)")

        return (_ICON_OK, "")

    def _build_line(
        self,
        icon: str,
        *,
        show_eta: bool = True,
        suffix: str = "",
    ) -> Text:
        """Compose the single-line progress string."""
        elapsed = time.monotonic() - self._start_time
        fps = self._frames_processed / elapsed if elapsed > 0 else 0.0
        total = self._source_info.total_frames

        if total is not None:
            total_str = str(total)
            frames_part = f"{self._frames_processed:>{len(total_str)}} / {total_str}"
        else:
            frames_part = f"{self._frames_processed} / --"

        if total is not None and total > 0:
            percentage = self._frames_processed / total * 100
            percentage_part = f"{percentage:>3.0f}%"
        else:
            percentage_part = "  --"

        fps_part = f"{fps:>.1f} fps"
        elapsed_part = f"{_format_time(elapsed)} elapsed"

        parts = [
            f"{icon} Tracking",
            f"{frames_part} frames",
            percentage_part,
            fps_part,
            elapsed_part,
        ]

        if show_eta:
            if total is not None and fps > 0:
                remaining = (total - self._frames_processed) / fps
                parts.append(f"eta {_format_time(remaining)}")
            else:
                parts.append("eta --")

        if suffix:
            parts.append(suffix)

        return Text("   ".join(parts))
