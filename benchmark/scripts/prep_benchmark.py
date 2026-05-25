#!/usr/bin/env python3
"""Convert local benchmark detections/GT into flat MOT dirs for ``trackers tune``."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def _soccer_seq_name(stem: str) -> str:
    return stem.replace("__det", "")


def _mot17_val_seq_name(stem: str) -> str:
    return stem.split("_")[0] + "-FRCNN"


@dataclass(frozen=True)
class SplitSpec:
    det_dir: Path
    gt_dir: Path | None
    det_format: str  # xyxy | mot | ltwh_mot
    seq_name_fn: Callable[[str], str] | None = None


def split_spec(data_root: Path, dataset: str, split: str) -> SplitSpec | None:
    root = data_root
    if dataset == "soccernet":
        if split == "train":
            return SplitSpec(
                det_dir=root / "soccernet/SoccerNet_dets/SoccerNet_tracking/train",
                gt_dir=root / "soccernet/TrackEval/data/gt/SoccerNet_tracking/train",
                det_format="mot",
                seq_name_fn=_soccer_seq_name,
            )
        if split == "test":
            return SplitSpec(
                det_dir=root / "soccernet/SoccerNet_dets/SoccerNet_tracking_2022_all_dets",
                gt_dir=root / "soccernet/TrackEval/data/gt/SoccerNet_tracking/SoccerNet_tracking_2022_all_gts",
                det_format="ltwh_mot",
                seq_name_fn=_soccer_seq_name,
            )
    if dataset == "dancetrack" and split in {"train", "val", "test"}:
        gt_dir = None if split == "test" else root / f"dancetrack/TrackEval/data/gt/dancetrack/{split}"
        return SplitSpec(
            det_dir=root / f"dancetrack/dancetrack_yolox_dets/{split}",
            gt_dir=gt_dir,
            det_format="xyxy",
        )
    if dataset == "sportsmot" and split in {"val", "test"}:
        gt_dir = None if split == "test" else root / f"sportsmot/TrackEval/data/gt/sportsmot/{split}"
        return SplitSpec(
            det_dir=root / f"sportsmot/sportsmot_yolox_dets/{split}",
            gt_dir=gt_dir,
            det_format="xyxy",
        )
    if dataset == "mot17":
        if split == "val":
            return SplitSpec(
                det_dir=root / "mot17/MOT17_yolox_dets/val",
                gt_dir=root / "mot17/TrackEval/data/gt/MOT17_yolox_val/train_val",
                det_format="xyxy",
                seq_name_fn=_mot17_val_seq_name,
            )
        if split == "test":
            return SplitSpec(
                det_dir=root / "mot17/MOT17_yolox_dets/test",
                gt_dir=None,
                det_format="xyxy",
            )
    return None


def prepare_mot_dets(
    src_dir: Path,
    dst_dir: Path,
    *,
    src_format: str,
    seq_name_fn: Callable[[str], str] | None = None,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for det_file in sorted(src_dir.glob("*.txt")):
        seq_name = seq_name_fn(det_file.stem) if seq_name_fn else det_file.stem
        dst_path = dst_dir / f"{seq_name}.txt"
        if src_format == "mot":
            shutil.copy(det_file, dst_path)
            continue
        with det_file.open() as fin, dst_path.open("w") as fout:
            for line in fin:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                frame = int(parts[0])
                if src_format == "ltwh_mot":
                    left, top, w, h = (float(parts[i]) for i in range(2, 6))
                    conf = float(parts[6]) if len(parts) > 6 else 1.0
                else:
                    x1, y1, x2, y2, conf = (float(p) for p in parts[1:6])
                    left, top, w, h = x1, y1, x2 - x1, y2 - y1
                fout.write(f"{frame},-1,{left:.4f},{top:.4f},{w:.4f},{h:.4f},{conf:.4f}\n")


def prepare_flat_gt(gt_root: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for seq_dir in sorted(gt_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        gt_path = seq_dir / "gt" / "gt.txt"
        if gt_path.is_file():
            shutil.copy(gt_path, dst_dir / f"{seq_dir.name}.txt")


def prep_split(data_root: Path, prep_root: Path, dataset: str, split: str) -> Path:
    spec = split_spec(data_root, dataset, split)
    if spec is None:
        raise ValueError(f"Unknown dataset/split: {dataset}/{split}")
    if not spec.det_dir.is_dir():
        raise FileNotFoundError(f"Missing detections: {spec.det_dir}")

    out = prep_root / dataset / split
    dets_out = out / "dets"
    gt_out = out / "gt"
    prepare_mot_dets(
        spec.det_dir,
        dets_out,
        src_format=spec.det_format,
        seq_name_fn=spec.seq_name_fn,
    )
    if spec.gt_dir is not None:
        if not spec.gt_dir.is_dir():
            raise FileNotFoundError(f"Missing GT: {spec.gt_dir}")
        prepare_flat_gt(spec.gt_dir, gt_out)
    print(f"Prepared {dataset}/{split} → {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    datasets = ("soccernet", "dancetrack", "sportsmot", "mot17")
    tune_splits = {
        "soccernet": "train",
        "dancetrack": "train",
        "sportsmot": "val",
        "mot17": "val",
    }
    eval_splits = {
        "soccernet": "test",
        "dancetrack": "val",
        "sportsmot": "val",
        "mot17": "val",
    }
    submit_splits = {
        "dancetrack": "test",
        "sportsmot": "test",
        "mot17": "test",
    }

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=_BENCHMARK_ROOT / "data")
    p.add_argument("--prep-dir", type=Path, default=_BENCHMARK_ROOT / "benchmark_prep")
    p.add_argument("--dataset", choices=[*datasets, "all"], default="all")
    p.add_argument(
        "--split",
        choices=["all", "tune", "eval", "submit", "train", "val", "test"],
        default="tune",
        help="Which split to prep (all=tune+eval+submit for dataset, or explicit split name).",
    )
    args = p.parse_args(argv)

    picked = list(datasets) if args.dataset == "all" else [args.dataset]
    split_aliases = ("tune", "eval", "submit")
    for dataset in picked:
        splits_to_run: list[str]
        if args.split == "all":
            splits_to_run = list(split_aliases)
        else:
            splits_to_run = [args.split]

        for split_key in splits_to_run:
            if split_key in {"tune", "eval", "submit"}:
                split_map = {"tune": tune_splits, "eval": eval_splits, "submit": submit_splits}
                if split_key == "submit" and dataset not in submit_splits:
                    continue
                split = split_map[split_key][dataset]
            else:
                split = split_key
            try:
                prep_split(args.data_root, args.prep_dir, dataset, split)
            except (FileNotFoundError, ValueError) as exc:
                print(f"SKIP {dataset}/{split}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
