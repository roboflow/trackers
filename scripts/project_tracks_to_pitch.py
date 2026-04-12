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
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trackers.calibration.export import (
    load_calibration_jsonl,
    write_track_projections_csv,
)
from trackers.calibration.pitch import PitchModel
from trackers.calibration.projection import project_image_points_to_pitch
from trackers.calibration.types import PitchDimensions, TrackProjection


def _resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _load_track_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            rows.append(
                {
                    "frame_idx": float(raw_row[0]),
                    "track_id": float(raw_row[1]),
                    "bb_left": float(raw_row[2]),
                    "bb_top": float(raw_row[3]),
                    "bb_width": float(raw_row[4]),
                    "bb_height": float(raw_row[5]),
                    "confidence": float(raw_row[6]) if len(raw_row) > 6 else 1.0,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project MOT tracks into pitch coordinates using calibration JSONL."
    )
    parser.add_argument("tracks", help="MOT-format tracks.txt file")
    parser.add_argument("calibration", help="Calibration JSONL file")
    parser.add_argument(
        "--output",
        required=True,
        help="CSV file for projected pitch coordinates",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS for deriving timestamps when calibration timestamps are absent",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tracks_path = _resolve_path(args.tracks)
    calibration_path = _resolve_path(args.calibration)
    output_path = _resolve_path(args.output)

    calibration_frames = load_calibration_jsonl(calibration_path)
    calibration_by_frame = {frame.frame_idx: frame for frame in calibration_frames}
    if not calibration_frames:
        raise ValueError(f"No calibration frames found in {calibration_path}")

    pitch_dimensions = calibration_frames[0].pitch_dimensions
    pitch_model = PitchModel(
        dimensions=PitchDimensions(
            length_m=pitch_dimensions.length_m,
            width_m=pitch_dimensions.width_m,
        )
    )

    projections: list[TrackProjection] = []
    for row in _load_track_rows(tracks_path):
        frame_idx = int(row["frame_idx"])
        track_id = int(row["track_id"])
        if track_id < 0:
            continue

        calibration = calibration_by_frame.get(frame_idx)
        if calibration is None or not calibration.has_homography:
            continue

        image_x = row["bb_left"] + (row["bb_width"] / 2.0)
        image_y = row["bb_top"] + row["bb_height"]
        pitch_point = project_image_points_to_pitch([[image_x, image_y]], calibration)[
            0
        ]
        normalized_point = pitch_model.metric_to_normalized([pitch_point])[0]
        in_pitch_bounds = bool(
            pitch_model.contains_metric_points([pitch_point], tolerance_m=0.0)[0]
        )

        timestamp_s = calibration.timestamp_s
        if timestamp_s is None and args.fps:
            timestamp_s = (frame_idx - 1) / args.fps

        projections.append(
            TrackProjection(
                frame_idx=frame_idx,
                track_id=track_id,
                image_x=float(image_x),
                image_y=float(image_y),
                pitch_x_m=float(pitch_point[0]),
                pitch_y_m=float(pitch_point[1]),
                pitch_x_norm=float(normalized_point[0]),
                pitch_y_norm=float(normalized_point[1]),
                in_pitch_bounds=in_pitch_bounds,
                timestamp_s=timestamp_s,
                calibration_confidence=calibration.confidence,
                source_confidence=row["confidence"],
                provider=calibration.provider,
            )
        )

    write_track_projections_csv(output_path, projections)
    print(f"Wrote {len(projections)} projected rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
