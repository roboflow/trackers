#!/usr/bin/env python3
"""Build a Codabench submission from raw YOLOX detections (notebook-style loop).

Uses each tracker's library defaults (or an optional params JSON) directly,
avoiding the ``trackers track`` CLI shared-parameter default bug.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import supervision as sv

_BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def det_root(data_root: Path, dataset: str, split: str) -> Path:
    rel = {
        "mot17": data_root / "mot17" / "MOT17_yolox_dets" / split,
        "sportsmot": data_root / "sportsmot" / "sportsmot_yolox_dets" / split,
        "dancetrack": data_root / "dancetrack" / "dancetrack_yolox_dets" / split,
    }
    if dataset not in rel:
        raise ValueError(f"unsupported dataset: {dataset}")
    return rel[dataset]


def _build_index(det_list: list[str]) -> dict[int, list[str]]:
    dets_by_frame: dict[int, list[str]] = defaultdict(list)
    for line in det_list:
        dets_by_frame[int(line.split(",")[0])].append(line)
    return dets_by_frame


def _yolox_rows(frame_id: int, dets_by_frame: dict[int, list[str]]) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in dets_by_frame.get(frame_id, []):
        parts = line.split(",")
        rows.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
    return rows


def _write_mot_line(frame_id: int, track_id: int, left: float, top: float, right: float, bottom: float) -> str:
    width = right - left
    height = bottom - top
    return f"{frame_id},{int(track_id)},{left:.1f},{top:.1f},{width:.1f},{height:.1f},-1,-1,-1,-1\n"


def _init_tracker(tracker_id: str, params: dict):
    import trackers as _trackers  # noqa: F401
    from trackers.core.base import BaseTracker

    info = BaseTracker._lookup_tracker(tracker_id)
    if info is None:
        raise ValueError(f"unknown tracker: {tracker_id}")
    return info.tracker_class(**params)


def _frame_path(images_root: Path | None, seq_name: str, frame_id: int, *, dataset: str) -> Path | None:
    if images_root is None:
        return None
    frame_seq = seq_name
    if dataset == "mot17" and not seq_name.endswith("-FRCNN"):
        frame_seq = f"{seq_name}-FRCNN"
    return images_root / frame_seq / "img1" / f"{frame_id:06d}.jpg"


def _read_frame(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Missing frame for CMC: {path}")
    import cv2

    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Failed to read frame: {path}")
    return frame


def run_yolox_submit(
    tracker_id: str,
    params: dict,
    *,
    dataset: str,
    split: str,
    detections_dir: Path,
    out_dir: Path,
    images_root: Path | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = _init_tracker(tracker_id, params)

    for det_file in sorted(detections_dir.glob("*.txt")):
        tracker.reset()
        seq_name = det_file.stem
        det_list = det_file.read_text().splitlines()
        if not det_list:
            print(f"  skip empty {seq_name}")
            continue
        dets_by_frame = _build_index(det_list)
        last_frame = int(det_list[-1].split(",")[0])
        lines: list[str] = []

        for frame_id in range(1, last_frame + 1):
            raw = _yolox_rows(frame_id, dets_by_frame)
            if raw:
                arr = np.array(raw)
                dets = sv.Detections(xyxy=arr[:, :4], confidence=arr[:, 4])
            else:
                dets = sv.Detections.empty()

            frame = _read_frame(_frame_path(images_root, seq_name, frame_id, dataset=dataset))
            tracked = tracker.update(detections=dets, frame=frame)
            if tracked.tracker_id is None:
                continue
            for tid, (left, top, right, bottom) in zip(tracked.tracker_id, tracked.xyxy):
                if tid == -1:
                    continue
                left_f, top_f, right_f, bottom_f = map(float, (left, top, right, bottom))
                if not np.isfinite((left_f, top_f, right_f, bottom_f)).all():
                    continue
                if right_f <= left_f or bottom_f <= top_f:
                    continue
                lines.append(_write_mot_line(frame_id, int(tid), left_f, top_f, right_f, bottom_f))

        (out_dir / f"{seq_name}.txt").write_text("".join(lines))
        print(f"  tracked {seq_name} ({last_frame} frames)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracker", required=True)
    p.add_argument("--dataset", choices=("mot17", "sportsmot", "dancetrack"), required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--data-root", type=Path, default=_BENCHMARK_ROOT / "data")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--params", type=Path, default=None, help="JSON tracker params (default: library defaults)")
    p.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Sequence root with <seq>/img1/ frames (required for BoT-SORT CMC on submit)",
    )
    args = p.parse_args(argv)

    params: dict = {}
    if args.params is not None:
        params = json.loads(args.params.read_text())

    dets = det_root(args.data_root, args.dataset, args.split)
    if not dets.is_dir():
        print(f"Missing detections: {dets}", file=sys.stderr)
        return 1

    import importlib.metadata as md

    print(f"trackers {md.version('trackers')} | {args.tracker} | {args.dataset}/{args.split}")
    run_yolox_submit(
        args.tracker,
        params,
        dataset=args.dataset,
        split=args.split,
        detections_dir=dets,
        out_dir=args.output_dir,
        images_root=args.images_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
