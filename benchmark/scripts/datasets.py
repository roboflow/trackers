# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Benchmark dataset layout: paths, splits, Codabench targets.

Single source of truth shared by all benchmark scripts. Not a CLI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]

DATASETS = ("mot17", "sportsmot", "soccernet", "dancetrack")
LABELS = {"mot17": "MOT17", "sportsmot": "SportsMOT", "soccernet": "SoccerNet", "dancetrack": "DanceTrack"}

# Trackers shown side-by-side in docs/trackers/comparison.md — single source of truth for
# Makefile (via `datasets.py --field comparison_trackers`) and collect.py.
COMPARISON_TRACKERS = ("sort", "bytetrack", "ocsort", "botsort")
TRACKER_LABELS = {
    "sort": "SORT",
    "bytetrack": "ByteTrack",
    "ocsort": "OC-SORT",
    "botsort": "BoT-SORT",
}

# Per-dataset splits used by the benchmark workflow.
TUNE_SPLIT = {"soccernet": "train", "dancetrack": "train", "sportsmot": "val", "mot17": "val"}
EVAL_SPLIT = {"soccernet": "test", "dancetrack": "val", "sportsmot": "val", "mot17": "val"}
SUBMIT_SPLIT = {"dancetrack": "test", "sportsmot": "test", "mot17": "test"}  # soccernet has no Codabench

# Codabench (competition_id, phase_id) — submission targets for the published comparison table.
CODABENCH = {
    "mot17": (10049, 16382),
    "sportsmot": (13077, 21402),
    "dancetrack": (14885, 24635),
}

METRICS = ["CLEAR", "HOTA", "Identity"]

_MOT17_EXISTING = ("01", "03", "06", "07", "08", "12", "14")
_MOT17_MISSING = ("02", "04", "05", "09", "10", "11", "13")
_MOT17_SUFFIXES = ("FRCNN", "SDP", "DPM")


def _soccernet_seq(stem: str) -> str:
    return stem.replace("__det", "")


def _mot17_val_seq(stem: str) -> str:
    return stem.split("_")[0] + "-FRCNN"


@dataclass(frozen=True)
class SplitPaths:
    det_dir: Path
    det_format: str  # "mot" | "ltwh_mot" | "xyxy"
    gt_dir: Path | None
    images_dir: Path | None
    seqmap: Path | None
    seq_name_fn: Callable[[str], str] | None = None


def split_paths(data_root: Path, dataset: str, split: str) -> SplitPaths:
    """Resolve where vendor detections, GT, frames, and seqmap live for one (dataset, split)."""
    root = data_root
    if dataset == "soccernet":
        gt_root = root / "soccernet/TrackEval/data/gt/SoccerNet_tracking"
        img_root = root / "soccernet/soccernet_data/tracking"
        if split == "train":
            return SplitPaths(
                root / "soccernet/SoccerNet_dets/SoccerNet_tracking/train",
                "mot",
                gt_root / "train",
                img_root / "train",
                None,
                _soccernet_seq,
            )
        if split == "test":
            return SplitPaths(
                root / "soccernet/SoccerNet_dets/SoccerNet_tracking_2022_all_dets",
                "ltwh_mot",
                gt_root / "SoccerNet_tracking_2022_all_gts",
                img_root / "test",
                None,
                _soccernet_seq,
            )
    if dataset == "dancetrack":
        gt = root / f"dancetrack/TrackEval/data/gt/dancetrack/{split}" if split != "test" else None
        images = root / f"dancetrack/{split}_images"
        seqmap = root / f"dancetrack/TrackEval/data/gt/dancetrack/DanceTrack-{split}.txt"
        seqmap_or_none = seqmap if seqmap.parent.is_dir() else None
        if split in {"train", "val", "test"}:
            return SplitPaths(root / f"dancetrack/dancetrack_yolox_dets/{split}", "xyxy", gt, images, seqmap_or_none)
    if dataset == "sportsmot":
        if split in {"val", "test"}:
            gt = root / "sportsmot/TrackEval/data/gt/sportsmot/val" if split == "val" else None
            images = root / "sportsmot" / split
            return SplitPaths(root / f"sportsmot/sportsmot_yolox_dets/{split}", "xyxy", gt, images, None)
    if dataset == "mot17":
        if split == "val":
            seqmap = root / "mot17/TrackEval/data/gt/MOT17/MOT17-val.txt"
            return SplitPaths(
                root / "mot17/MOT17_yolox_dets/val",
                "xyxy",
                root / "mot17/TrackEval/data/gt/MOT17_yolox_val/train_val",
                root / "mot17/val",
                seqmap if seqmap.is_file() else None,
                _mot17_val_seq,
            )
        if split == "test":
            return SplitPaths(root / "mot17/MOT17_yolox_dets/test", "xyxy", None, root / "mot17/test", None)
    raise ValueError(f"unknown (dataset, split): ({dataset!r}, {split!r})")


def prep_split_dir(prep_root: Path, dataset: str, split: str) -> Path:
    """Output dir for prepared flat MOT detections + GT for one (dataset, split)."""
    return prep_root / dataset / split


def job_dir(output_dir: Path, tracker: str, dataset: str) -> Path:
    """Per-(tracker, dataset) output directory for tuned params, predictions, scores."""
    return output_dir / tracker / dataset


def needs_frames(tracker: str, params: dict) -> bool:
    """Whether tracking requires source frames (e.g. BoT-SORT with CMC enabled)."""
    return tracker == "botsort" and bool(params.get("enable_cmc", False))


def mot17_server_filenames() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Constants for the MOT17 Codabench server: must triplicate (FRCNN/SDP/DPM) and stub missing seqs."""
    return _MOT17_EXISTING, _MOT17_MISSING, _MOT17_SUFFIXES


def _print_field(data_root: Path, dataset: str, split: str, what: str) -> str:
    paths = split_paths(data_root, dataset, split)
    value = getattr(paths, what)
    return "" if value is None else str(value)


_GLOBAL_FIELDS = {
    "comparison_trackers": lambda _root: " ".join(COMPARISON_TRACKERS),
    "comparison_trackers_csv": lambda _root: ",".join(COMPARISON_TRACKERS),
    "datasets": lambda _root: " ".join(DATASETS),
    "datasets_csv": lambda _root: ",".join(DATASETS),
}
_LAYOUT_FIELDS = ("det_dir", "gt_dir", "images_dir", "seqmap")


def _print_global(data_root: Path, field: str) -> str:
    return _GLOBAL_FIELDS[field](data_root)


# Tiny CLI so the Makefile can query layout values without duplicating paths.
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Print one layout field (used by Makefile).")
    p.add_argument("--data-root", type=Path, default=BENCHMARK_ROOT / "data")
    p.add_argument("--dataset")
    p.add_argument("--split")
    p.add_argument(
        "--field",
        choices=[*_LAYOUT_FIELDS, *_GLOBAL_FIELDS],
        required=True,
    )
    args = p.parse_args()
    if args.field in _GLOBAL_FIELDS:
        print(_print_global(args.data_root, args.field))
    else:
        if not args.dataset or not args.split:
            p.error(f"--dataset and --split are required for --field {args.field!r}")
        print(_print_field(args.data_root, args.dataset, args.split, args.field))
