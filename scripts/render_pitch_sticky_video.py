#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trackers.calibration.export import load_calibration_jsonl
from trackers.calibration.projection import project_image_points_to_pitch


@dataclass(frozen=True)
class TrackRow:
    frame_idx: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    confidence: float

    @property
    def bottom_center(self) -> np.ndarray:
        return np.array(
            [self.left + (self.width / 2.0), self.top + self.height],
            dtype=np.float64,
        )

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.left)),
            int(round(self.top)),
            int(round(self.left + self.width)),
            int(round(self.top + self.height)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render long pitch-locked streamer trails from tracks and homographies."
    )
    parser.add_argument("source", help="Source video clip")
    parser.add_argument("tracks", help="MOT-format tracks.txt")
    parser.add_argument("homography", help="Calibration homography JSONL")
    parser.add_argument("output", help="Output rendered video")
    parser.add_argument(
        "--trace-seconds",
        type=float,
        default=15.0,
        help="How many seconds of trail history to keep on screen",
    )
    parser.add_argument(
        "--box-thickness",
        type=int,
        default=2,
        help="Bounding-box thickness",
    )
    parser.add_argument(
        "--trail-thickness",
        type=int,
        default=2,
        help="Trail thickness",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.6,
        help="Label font scale",
    )
    return parser.parse_args()


def _load_tracks(path: Path) -> dict[int, list[TrackRow]]:
    tracks_by_frame: dict[int, list[TrackRow]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            track_id = int(float(row[1]))
            if track_id < 0:
                continue
            track = TrackRow(
                frame_idx=int(float(row[0])),
                track_id=track_id,
                left=float(row[2]),
                top=float(row[3]),
                width=float(row[4]),
                height=float(row[5]),
                confidence=float(row[6]) if len(row) > 6 else 1.0,
            )
            tracks_by_frame[track.frame_idx].append(track)
    return tracks_by_frame


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    color = rng.integers(64, 256, size=3, dtype=np.int32)
    return int(color[0]), int(color[1]), int(color[2])


def _draw_label(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    *,
    font_scale: float,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    x = max(0, x)
    y = max(text_size[1] + baseline, y)
    box_tl = (x, y - text_size[1] - baseline - 4)
    box_br = (x + text_size[0] + 6, y + 2)
    cv2.rectangle(frame, box_tl, box_br, color, thickness=-1)
    cv2.putText(
        frame,
        text,
        (x + 3, y - baseline - 1),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )


def _project_pitch_history(
    pitch_points: list[np.ndarray],
    pitch_to_image: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    if len(pitch_points) < 2:
        return np.empty((0, 2), dtype=np.float64)
    points = np.asarray(pitch_points, dtype=np.float64)
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    projected = homogeneous @ pitch_to_image.T
    scale = projected[:, 2:3]
    valid = np.abs(scale[:, 0]) > 1e-9
    projected = projected[valid]
    if projected.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    projected = projected[:, :2] / projected[:, 2:3]
    finite = np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1])
    projected = projected[finite]
    if projected.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    inside = (
        (projected[:, 0] >= -50)
        & (projected[:, 0] <= width + 50)
        & (projected[:, 1] >= -50)
        & (projected[:, 1] <= height + 50)
    )
    return projected[inside]


def main() -> int:
    args = parse_args()
    source_path = Path(args.source).expanduser().resolve()
    tracks_path = Path(args.tracks).expanduser().resolve()
    homography_path = Path(args.homography).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    tracks_by_frame = _load_tracks(tracks_path)
    calibration_frames = {
        frame.frame_idx: frame for frame in load_calibration_jsonl(homography_path)
    }

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open source video: {source_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_history = max(2, int(round(args.trace_seconds * fps)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    trail_history: dict[int, deque[tuple[int, np.ndarray]]] = defaultdict(
        lambda: deque(maxlen=max_history)
    )

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        calibration = calibration_frames.get(frame_idx)
        current_tracks = tracks_by_frame.get(frame_idx, [])

        if calibration is not None and calibration.image_to_pitch is not None:
            for track in current_tracks:
                pitch_point = project_image_points_to_pitch(
                    [track.bottom_center],
                    calibration,
                )[0]
                history = trail_history[track.track_id]
                history.append((frame_idx, pitch_point))

        # Trim any stale history if we skipped frames.
        oldest_allowed = frame_idx - max_history + 1
        for track_id, history in trail_history.items():
            while history and history[0][0] < oldest_allowed:
                history.popleft()

        rendered = frame.copy()
        for track in current_tracks:
            color = _color_for_track(track.track_id)
            x1, y1, x2, y2 = track.xyxy
            cv2.rectangle(
                rendered,
                (x1, y1),
                (x2, y2),
                color,
                thickness=args.box_thickness,
            )
            _draw_label(
                rendered,
                f"ID {track.track_id}",
                (x1, max(18, y1 - 6)),
                color,
                font_scale=args.font_scale,
            )

            if calibration is None or calibration.pitch_to_image is None:
                continue

            history = trail_history.get(track.track_id)
            if not history:
                continue
            pitch_points = [point for _, point in history]
            projected = _project_pitch_history(
                pitch_points,
                calibration.pitch_to_image,
                width,
                height,
            )
            if projected.shape[0] >= 2:
                cv2.polylines(
                    rendered,
                    [projected.astype(np.int32)],
                    isClosed=False,
                    color=color,
                    thickness=args.trail_thickness,
                    lineType=cv2.LINE_AA,
                )

        writer.write(rendered)

    cap.release()
    writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
