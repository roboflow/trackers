#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Run one tracker over a prepared MOT detection directory and write MOT predictions.

This script intentionally bypasses the `trackers track` CLI for two reasons:

1. Per-sequence subprocess startup dominates wall time for fast trackers (SORT/ByteTrack).
2. The CLI currently has a shared-parameter bug where flag defaults can leak between
   trackers (#TODO: fix and switch this script to a Makefile loop calling `trackers track`).

We sidestep both by importing the registry directly: defaults come from the chosen
tracker's `ParameterInfo`, optionally overridden by a tuned `best_params.json`, and
the merged dict is filtered to the tracker's `__init__` signature.

Usage (see Makefile for the wiring):

    python track_split.py --tracker sort --dataset mot17 --split val \
        --data-root ./data --prep-dir ./benchmark_prep \
        --output-dir ./benchmark_outputs/sort/mot17/default \
        [--params best_params.json]

    # BoT-SORT + official FastReID SBS (requires `pip install 'trackers[reid]'`):
    python track_split.py --tracker botsort --dataset mot17 --split test \
        --data-root ./data --prep-dir ./benchmark_prep \
        --output-dir ./benchmark_outputs/botsort/mot17/default \
        --reid fastreid_mot17_sbs50 --appearance-threshold 0.2
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from datasets import DATASETS, needs_frames, split_paths

# trackers is installed as a regular package via `pip install -e ../`.
from trackers.core.base import BaseTracker
from trackers.tune.tuner import _run_tracker_on_detections


def _registry_defaults(tracker_id: str) -> dict[str, Any]:
    info = BaseTracker._lookup_tracker(tracker_id)
    if info is None:
        raise ValueError(f"unknown tracker: {tracker_id!r}")
    out: dict[str, Any] = {}
    for name, param in info.parameters.items():
        if name == "state_estimator_class":
            out[name] = param.default_value
        elif not isinstance(param.default_value, type):
            out[name] = param.default_value
    return out


def _init_kwargs(tracker_id: str, params: dict[str, Any]) -> dict[str, Any]:
    info = BaseTracker._lookup_tracker(tracker_id)
    if info is None:
        raise ValueError(f"unknown tracker: {tracker_id!r}")
    names = {n for n in inspect.signature(info.tracker_class.__init__).parameters if n != "self"}
    return {k: v for k, v in params.items() if k in names}


def _resolve_params(tracker_id: str, *, params_file: Path | None) -> dict[str, Any]:
    """Merge registry defaults with optional tuned overrides; force CMC on for BoT-SORT."""
    merged = _registry_defaults(tracker_id)
    if params_file is not None and params_file.is_file():
        merged.update(json.loads(params_file.read_text()))
    if tracker_id == "botsort" and "enable_cmc" not in merged:
        merged["enable_cmc"] = True
    return _init_kwargs(tracker_id, merged)


def _load_reid_model(source: str, appearance_threshold: float | None) -> dict[str, Any]:
    """Load a ``reid.ReIDModel`` and optional appearance threshold overrides."""
    try:
        from reid import ReIDModel
    except ImportError as exc:
        raise ImportError(
            "BoT-SORT ReID requires the standalone reid package. "
            "Install with: pip install 'trackers[reid]'"
        ) from exc

    overrides: dict[str, Any] = {"reid_model": ReIDModel.from_pretrained(source)}
    if appearance_threshold is not None:
        overrides["appearance_threshold"] = appearance_threshold
    return overrides


def _build(tracker_id: str, params: dict[str, Any]) -> BaseTracker:
    info = BaseTracker._lookup_tracker(tracker_id)
    if info is None:
        raise ValueError(f"unknown tracker: {tracker_id!r}")
    return info.tracker_class(**params)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracker", required=True)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--split", required=True, help="Prepared split (tune/eval/submit name, e.g. val, test).")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--prep-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="Predictions root: writes pred/<seq>.txt under here.")
    p.add_argument("--params", type=Path, default=None, help="Optional tuned best_params.json")
    p.add_argument(
        "--reid",
        default=None,
        help="Optional reid alias/path for BoT-SORT (e.g. fastreid_mot17_sbs50).",
    )
    p.add_argument(
        "--appearance-threshold",
        type=float,
        default=None,
        help="BoT-SORT appearance_threshold when --reid is set (e.g. 0.2).",
    )
    args = p.parse_args(argv)

    if args.reid is not None and args.tracker != "botsort":
        print(f"--reid is only supported for botsort (got {args.tracker!r})", file=sys.stderr)
        return 1
    if args.appearance_threshold is not None and args.reid is None:
        print("--appearance-threshold requires --reid", file=sys.stderr)
        return 1

    dets_dir = args.prep_dir / args.dataset / args.split / "dets"
    if not dets_dir.is_dir():
        print(f"missing prepared dets: {dets_dir} (run `make prep DATASET={args.dataset}`)", file=sys.stderr)
        return 1

    params = _resolve_params(args.tracker, params_file=args.params)
    if args.reid is not None:
        params.update(_load_reid_model(args.reid, args.appearance_threshold))

    paths = split_paths(args.data_root, args.dataset, args.split)
    images_dir = paths.images_dir if needs_frames(args.tracker, params) else None
    if images_dir is not None and not images_dir.is_dir():
        print(f"missing frames for CMC/ReID: {images_dir}", file=sys.stderr)
        return 1

    pred_dir = args.output_dir / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    tracker = _build(args.tracker, params)
    for det_path in sorted(dets_dir.glob("*.txt")):
        seq = det_path.stem
        # Pred names stay as det stems (MOT17-01); frames may live under MOT17-01-FRCNN.
        image_seq = paths.image_seq_name_fn(seq) if paths.image_seq_name_fn is not None else seq
        tracker.reset()
        _run_tracker_on_detections(
            tracker,
            det_path,
            pred_dir / f"{seq}.txt",
            images_dir=images_dir,
            seq_name=image_seq,
        )
        print(f"  tracked {seq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
