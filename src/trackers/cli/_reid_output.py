# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Stream current and resolved McByte++ observations to their source frames."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import supervision as sv

from trackers.core.mcbyteplusplus.tracker import McBytePlusPlusTracker


class _ReIDResultWriter:
    """Append resolved rows once, retaining only pending source-frame references.

    The tracker stores pending observations. This adapter maps their processing references to original source frame
    numbers and counts unresolved rows. Different identities from the same frame can resolve on different updates, so
    the mapping remains until all those rows have been delivered. Output filters belong in ``write`` and never affect
    that accounting.
    """

    def __init__(self, tracker: McBytePlusPlusTracker, write: Callable[[int, sv.Detections], None]) -> None:
        self.tracker = tracker
        self._write = write
        self._pending: dict[int, tuple[int, int]] = {}
        self.historical_rows = 0

    def write_current(self, source_frame: int, detections: sv.Detections) -> None:
        """Drain resolved rows; without re-ID, preserve the caller's ordinary output policy."""
        joiner = self.tracker.reid_joiner
        if joiner is None:
            self._write(source_frame, detections)
            return
        if joiner.config.buffer_results:
            pending = int(np.count_nonzero(detections.data.get("reid_pending", [])))
            if pending:
                self._pending[self.tracker.frame_id] = (source_frame, pending)
            self._write_finalized(self.tracker.pop_finalized_results())
        self._write_confirmed(source_frame, detections)

    def _write_confirmed(self, source_frame: int, detections: sv.Detections) -> None:
        if detections.tracker_id is not None:
            self._write(source_frame, detections[detections.tracker_id >= 0])

    def _write_finalized(self, results: dict[int, sv.Detections]) -> None:
        for frame, detections in results.items():
            source_frame, pending = self._pending[frame]
            remaining = pending - len(detections)
            if remaining < 0:
                raise RuntimeError(f"Too many finalized observations for processing frame {frame}.")
            if remaining:
                self._pending[frame] = (source_frame, remaining)
            else:
                del self._pending[frame]
            self._write_confirmed(source_frame, detections)
            self.historical_rows += len(detections)

    def finish(self, *, completed_only: bool = False, processing_error: Exception | None = None) -> None:
        """Close the tracker and write its final pending tail before file close.

        With ``completed_only=True``, exclude observations from an update that never returned to the caller and
        therefore has no registered source frame. Callers use this option after interruption or processing failure. When
        ``processing_error`` is supplied, report cleanup failures without replacing that original error; the caller
        remains responsible for re-raising it and retaining failed/partial run status.
        """
        try:
            results = self.tracker.finish()
            if completed_only:
                results = {frame: rows for frame, rows in results.items() if frame in self._pending}
            self._write_finalized(results)
        except Exception as error:
            if processing_error is None:
                raise
            print(
                f"Warning: Could not finalize buffered ReID results after processing failed: {error}", file=sys.stderr
            )


def _sort_mot_results(path: Path | None, *, processing_error: Exception | None = None) -> None:
    """Sort MOT rows by source frame and public ID after output collection ends.

    All serialized text rows are loaded into memory for sorting; no images or Detections are retained. Callers can sort
    complete results at EOF or partial results after interruption or failure. When an original processing error is
    supplied, report sorting failures without replacing it; the caller must still re-raise the original error.
    """
    if path is None:
        return
    try:
        rows = path.read_text(encoding="utf-8").splitlines(keepends=True)
        rows.sort(key=lambda row: tuple(int(value) for value in row.split(",", 2)[:2]))
        path.write_text("".join(rows), encoding="utf-8")
    except (OSError, ValueError) as error:
        if processing_error is None:
            raise
        print(f"Warning: Could not sort partial MOT results after processing failed: {error}", file=sys.stderr)
