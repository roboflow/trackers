#!/usr/bin/env python
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

INPUT_DIR = Path("outputs/rfdetr-all-trackers")
ORIGINAL_VIDEO = Path("people-walking.mp4")
OUTPUT_VIDEO = INPUT_DIR / "people-walking-grid.mp4"
CELL_SIZE = (640, 360)
SKIP_FRAMES = 5
VIDEOS = [
    ("Original", ORIGINAL_VIDEO),
    ("SORT", INPUT_DIR / "people-walking-sort.mp4"),
    ("ByteTrack", INPUT_DIR / "people-walking-bytetrack.mp4"),
    ("OC-SORT", INPUT_DIR / "people-walking-ocsort.mp4"),
    ("BoT-SORT", INPUT_DIR / "people-walking-botsort.mp4"),
    ("C-BIoU", INPUT_DIR / "people-walking-cbiou.mp4"),
]


def _read(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = cap.read()
    return frame if ok else None


def _tile(frame: np.ndarray, label: str) -> np.ndarray:
    frame = cv2.resize(frame, CELL_SIZE, interpolation=cv2.INTER_AREA)
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (194, 0, 87), 2, cv2.LINE_AA)
    return frame


def main() -> int:
    caps = [(label, cv2.VideoCapture(str(path))) for label, path in VIDEOS]
    fps = caps[0][1].get(cv2.CAP_PROP_FPS) or 25.0
    width, height = CELL_SIZE
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height * 3),
    )
    for _, cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, SKIP_FRAMES)

    while True:
        frames = [(label, _read(cap)) for label, cap in caps]
        if any(frame is None for _, frame in frames):
            break

        tiles = [_tile(frame, label) for label, frame in frames if frame is not None]
        writer.write(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4]), np.hstack(tiles[4:])]))
        print(f"\rWrote frame {int(caps[0][1].get(cv2.CAP_PROP_POS_FRAMES))}", end="", flush=True)

    writer.release()
    for _, cap in caps:
        cap.release()
    print(f"\nWrote {OUTPUT_VIDEO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
