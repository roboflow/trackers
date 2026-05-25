#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Convert vendor MOT detections + GT into flat per-sequence MOT files.

Output layout (under ``--prep-dir``):

    <prep>/<dataset>/<split>/
      dets/<seq>.txt   # MOT lines: frame,-1,left,top,width,height,conf
      gt/<seq>.txt     # vanilla MOT gt.txt (copied)

Run via the Makefile (``make prep``) or directly:

    python prep_data.py --dataset mot17 --split val --data-root ./data --prep-dir ./benchmark_prep
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from datasets import DATASETS, EVAL_SPLIT, SUBMIT_SPLIT, TUNE_SPLIT, prep_split_dir, split_paths


def _convert_dets(src: Path, dst: Path, fmt: str) -> None:
    """Write one MOT-format detection file from a vendor file."""
    if fmt == "mot":
        shutil.copy(src, dst)
        return
    with src.open() as fin, dst.open("w") as fout:
        for raw in fin:
            parts = raw.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            if fmt == "ltwh_mot":
                left, top, w, h = (float(parts[i]) for i in range(2, 6))
                conf = float(parts[6]) if len(parts) > 6 else 1.0
            elif fmt == "xyxy":
                x1, y1, x2, y2, conf = (float(p) for p in parts[1:6])
                left, top, w, h = x1, y1, x2 - x1, y2 - y1
            else:
                raise ValueError(f"unknown det format: {fmt}")
            fout.write(f"{frame},-1,{left:.4f},{top:.4f},{w:.4f},{h:.4f},{conf:.4f}\n")


def prep_split(data_root: Path, prep_root: Path, dataset: str, split: str) -> Path:
    """Prepare flat MOT dets (+ optional GT) for one (dataset, split). Returns the output dir."""
    paths = split_paths(data_root, dataset, split)
    if not paths.det_dir.is_dir():
        raise FileNotFoundError(f"missing detections: {paths.det_dir}")
    out = prep_split_dir(prep_root, dataset, split)
    dets_out = out / "dets"
    dets_out.mkdir(parents=True, exist_ok=True)
    for det_file in sorted(paths.det_dir.glob("*.txt")):
        seq = paths.seq_name_fn(det_file.stem) if paths.seq_name_fn else det_file.stem
        _convert_dets(det_file, dets_out / f"{seq}.txt", paths.det_format)
    if paths.gt_dir is not None:
        if not paths.gt_dir.is_dir():
            raise FileNotFoundError(f"missing GT: {paths.gt_dir}")
        gt_out = out / "gt"
        gt_out.mkdir(parents=True, exist_ok=True)
        for seq_dir in sorted(paths.gt_dir.iterdir()):
            gt = seq_dir / "gt" / "gt.txt"
            if seq_dir.is_dir() and gt.is_file():
                shutil.copy(gt, gt_out / f"{seq_dir.name}.txt")
    print(f"prepped {dataset}/{split} → {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=[*DATASETS, "all"], required=True)
    p.add_argument("--split", choices=["all", "tune", "eval", "submit"], default="all")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--prep-dir", type=Path, required=True)
    args = p.parse_args(argv)

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    failed = False
    for dataset in datasets:
        splits: set[str] = set()
        if args.split in {"all", "tune"}:
            splits.add(TUNE_SPLIT[dataset])
        if args.split in {"all", "eval"}:
            splits.add(EVAL_SPLIT[dataset])
        if args.split in {"all", "submit"} and dataset in SUBMIT_SPLIT:
            splits.add(SUBMIT_SPLIT[dataset])
        for split in sorted(splits):
            try:
                prep_split(args.data_root, args.prep_dir, dataset, split)
            except FileNotFoundError as exc:
                print(f"SKIP {dataset}/{split}: {exc}", file=sys.stderr)
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
