#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrackStats:
    track_id: int
    detections: int
    start_frame: int
    end_frame: int
    span_frames: int
    missing_frames: int
    density: float
    segments: int
    longest_run: int


def _safe_mean(values: list[float | int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _safe_median(values: list[float | int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _segment_stats(frames: list[int]) -> tuple[int, int]:
    if not frames:
        return 0, 0

    segments = 1
    longest_run = 1
    current_run = 1

    for prev_frame, frame in zip(frames, frames[1:]):
        if frame == prev_frame + 1:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            segments += 1
            current_run = 1

    return segments, longest_run


def load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            rows.append(
                {
                    "frame": int(float(row[0])),
                    "track_id": int(float(row[1])),
                    "x": float(row[2]),
                    "y": float(row[3]),
                    "w": float(row[4]),
                    "h": float(row[5]),
                    "confidence": float(row[6]),
                }
            )
    return rows


def analyze(path: Path) -> dict:
    rows = load_rows(path)
    if not rows:
        return {
            "path": str(path),
            "total_rows": 0,
        }

    frame_to_ids: dict[int, set[int]] = defaultdict(set)
    frame_to_all_count: dict[int, int] = defaultdict(int)
    provisional_rows = 0
    track_frames: dict[int, list[int]] = defaultdict(list)

    min_frame = min(int(row["frame"]) for row in rows)
    max_frame = max(int(row["frame"]) for row in rows)

    for row in rows:
        frame = int(row["frame"])
        track_id = int(row["track_id"])
        frame_to_all_count[frame] += 1
        if track_id == -1:
            provisional_rows += 1
            continue
        frame_to_ids[frame].add(track_id)
        track_frames[track_id].append(frame)

    total_frames = max_frame - min_frame + 1
    frame_counts = [frame_to_all_count.get(frame, 0) for frame in range(min_frame, max_frame + 1)]
    confirmed_frame_counts = [len(frame_to_ids.get(frame, set())) for frame in range(min_frame, max_frame + 1)]

    adjacent_retention: list[float] = []
    adjacent_jaccard: list[float] = []
    for frame in range(min_frame, max_frame):
        ids_now = frame_to_ids.get(frame, set())
        ids_next = frame_to_ids.get(frame + 1, set())
        if ids_now:
            adjacent_retention.append(len(ids_now & ids_next) / len(ids_now))
        union = ids_now | ids_next
        if union:
            adjacent_jaccard.append(len(ids_now & ids_next) / len(union))

    per_track: list[TrackStats] = []
    for track_id, frames in sorted(track_frames.items()):
        frames = sorted(set(frames))
        start_frame = frames[0]
        end_frame = frames[-1]
        span_frames = end_frame - start_frame + 1
        detections = len(frames)
        missing_frames = span_frames - detections
        density = detections / span_frames if span_frames else 0.0
        segments, longest_run = _segment_stats(frames)
        per_track.append(
            TrackStats(
                track_id=track_id,
                detections=detections,
                start_frame=start_frame,
                end_frame=end_frame,
                span_frames=span_frames,
                missing_frames=missing_frames,
                density=density,
                segments=segments,
                longest_run=longest_run,
            )
        )

    detections_per_track = [track.detections for track in per_track]
    densities = [track.density for track in per_track]
    segments = [track.segments for track in per_track]
    longest_runs = [track.longest_run for track in per_track]
    single_segment_tracks = [track for track in per_track if track.segments == 1]
    fragmented_tracks = sorted(
        [track for track in per_track if track.segments > 1],
        key=lambda track: (track.segments, track.missing_frames, track.detections),
        reverse=True,
    )

    return {
        "path": str(path),
        "total_rows": len(rows),
        "provisional_rows": provisional_rows,
        "confirmed_rows": len(rows) - provisional_rows,
        "min_frame": min_frame,
        "max_frame": max_frame,
        "total_frames": total_frames,
        "confirmed_track_count": len(per_track),
        "detections_per_frame": {
            "mean": round(_safe_mean(frame_counts), 3),
            "median": round(_safe_median(frame_counts), 3),
            "p95": round(_percentile(frame_counts, 0.95), 3),
        },
        "confirmed_ids_per_frame": {
            "mean": round(_safe_mean(confirmed_frame_counts), 3),
            "median": round(_safe_median(confirmed_frame_counts), 3),
            "p95": round(_percentile(confirmed_frame_counts, 0.95), 3),
        },
        "adjacent_frame_continuity": {
            "mean_retention": round(_safe_mean(adjacent_retention), 4),
            "mean_jaccard": round(_safe_mean(adjacent_jaccard), 4),
        },
        "per_track": {
            "mean_detections": round(_safe_mean(detections_per_track), 3),
            "median_detections": round(_safe_median(detections_per_track), 3),
            "mean_density": round(_safe_mean(densities), 4),
            "median_density": round(_safe_median(densities), 4),
            "mean_segments": round(_safe_mean(segments), 3),
            "median_segments": round(_safe_median(segments), 3),
            "mean_longest_run": round(_safe_mean(longest_runs), 3),
            "median_longest_run": round(_safe_median(longest_runs), 3),
            "single_segment_fraction": round(
                len(single_segment_tracks) / len(per_track), 4
            )
            if per_track
            else 0.0,
        },
        "most_fragmented_tracks": [asdict(track) for track in fragmented_tracks[:10]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize continuity-style metrics from a MOT-format tracks file."
    )
    parser.add_argument("path", type=Path, help="Path to a MOT-format tracks.txt file.")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit JSON instead of a human-readable summary.",
    )
    args = parser.parse_args()

    result = analyze(args.path)
    if args.as_json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"path: {result['path']}")
    print(
        f"rows: {result['total_rows']} total, "
        f"{result['confirmed_rows']} confirmed, {result['provisional_rows']} provisional"
    )
    print(
        f"frames: {result['min_frame']}..{result['max_frame']} "
        f"({result['total_frames']} total)"
    )
    print(f"confirmed track ids: {result['confirmed_track_count']}")
    print(
        "detections/frame: "
        f"mean={result['detections_per_frame']['mean']}, "
        f"median={result['detections_per_frame']['median']}, "
        f"p95={result['detections_per_frame']['p95']}"
    )
    print(
        "confirmed ids/frame: "
        f"mean={result['confirmed_ids_per_frame']['mean']}, "
        f"median={result['confirmed_ids_per_frame']['median']}, "
        f"p95={result['confirmed_ids_per_frame']['p95']}"
    )
    print(
        "adjacent continuity: "
        f"retention={result['adjacent_frame_continuity']['mean_retention']}, "
        f"jaccard={result['adjacent_frame_continuity']['mean_jaccard']}"
    )
    print(
        "per-track continuity: "
        f"mean_density={result['per_track']['mean_density']}, "
        f"median_density={result['per_track']['median_density']}, "
        f"mean_segments={result['per_track']['mean_segments']}, "
        f"single_segment_fraction={result['per_track']['single_segment_fraction']}"
    )
    print("top fragmented tracks:")
    for track in result["most_fragmented_tracks"][:5]:
        print(
            f"  id={track['track_id']} dets={track['detections']} "
            f"segments={track['segments']} density={track['density']:.4f} "
            f"longest_run={track['longest_run']} missing={track['missing_frames']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
