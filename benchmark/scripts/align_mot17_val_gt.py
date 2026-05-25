#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Filter MOT17 validation GT to the frame range covered by YOLOX val detections.

Public YOLOX val detections for MOT17 cover only part of each sequence. The full
MOT17 GT under ``TrackEval/data/gt/MOT17/train_val`` includes extra frames, which
misaligns tuning if used as-is.

For each sequence, this script reads ``MOT17_yolox_dets/val/MOT17-XX_val.txt`` to
find the detection frame range, filters ``gt/gt.txt`` to those frames, and writes
the result under ``TrackEval/data/gt/MOT17_yolox_val/train_val/``.

Usage (from ``benchmark/``):

    python scripts/align_mot17_val_gt.py --data-root /path/to/datasets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def align_mot17_val_gt(data_root: Path) -> int:
    mot17 = data_root / "mot17"
    val_det_root = mot17 / "MOT17_yolox_dets" / "val"
    src_gt_root = mot17 / "TrackEval" / "data" / "gt" / "MOT17" / "train_val"
    dst_gt_root = mot17 / "TrackEval" / "data" / "gt" / "MOT17_yolox_val" / "train_val"

    if not src_gt_root.is_dir():
        print(f"missing source GT: {src_gt_root}", file=sys.stderr)
        return 1
    if not val_det_root.is_dir():
        print(f"missing YOLOX val detections: {val_det_root}", file=sys.stderr)
        return 1

    dst_gt_root.mkdir(parents=True, exist_ok=True)
    seq_dirs = sorted(p for p in src_gt_root.iterdir() if p.is_dir())
    if not seq_dirs:
        print(f"no sequences under {src_gt_root}", file=sys.stderr)
        return 1

    wrote = 0
    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        prefix = seq_name.split("-FRCNN")[0]
        det_file = val_det_root / f"{prefix}_val.txt"
        if not det_file.is_file():
            print(f"skip {seq_name}: missing {det_file.name}")
            continue

        frames: list[int] = []
        for line in det_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(int(line.split(",")[0]))
            except ValueError:
                continue
        if not frames:
            print(f"skip {seq_name}: no frames in {det_file.name}")
            continue

        f_min, f_max = min(frames), max(frames)
        src_gt = seq_dir / "gt" / "gt.txt"
        if not src_gt.is_file():
            print(f"skip {seq_name}: missing {src_gt}")
            continue

        kept = [
            ln.strip()
            for ln in src_gt.read_text().splitlines()
            if ln.strip() and f_min <= int(ln.split(",")[0]) <= f_max
        ]

        dst_seq_dir = dst_gt_root / seq_name
        dst_gt_dir = dst_seq_dir / "gt"
        dst_seq_dir.mkdir(parents=True, exist_ok=True)
        dst_gt_dir.mkdir(parents=True, exist_ok=True)
        for item in seq_dir.iterdir():
            if item.name != "gt" and item.is_file():
                (dst_seq_dir / item.name).write_bytes(item.read_bytes())
        (dst_gt_dir / "gt.txt").write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"{seq_name}: frames [{f_min}, {f_max}] → {len(kept)} GT lines")
        wrote += 1

    if wrote == 0:
        print("no sequences aligned", file=sys.stderr)
        return 1
    print(f"wrote aligned GT under {dst_gt_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True, help="Root containing mot17/, sportsmot/, …")
    return align_mot17_val_gt(p.parse_args(argv).data_root)


if __name__ == "__main__":
    raise SystemExit(main())
