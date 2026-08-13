#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Regenerate the docs appearance-distance figures from MOT17 and SoccerNet GT.

Sampling, separability and plotting all come from
``trackers.core.reid.thresholds``; what lives here is the dataset-specific part,
namely reading MOT-format ground truth, encoding the crops, and caching the
result so the figures can be re-drawn without re-encoding.

Writes:
  docs/assets/reid/mot17-fastreid-appearance-distances.png
  docs/assets/reid/soccernet-osnet-appearance-distances.png

Examples:
  python notebooks/plot_reid_appearance_distances.py --dataset mot17
  python notebooks/plot_reid_appearance_distances.py --dataset soccernet \\
      --soccernet-root "/path/to/soccernet_data/tracking/test"
  python notebooks/plot_reid_appearance_distances.py --dataset both
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from matplotlib.figure import Figure
from reid import FASTREID_MOT17_SBS50, ReIDModel

from trackers.core.reid import (
    DEFAULT_FRAME_GAP_BANDS,
    AppearanceDistances,
    plot_appearance_distances,
    plot_frame_gap_sweep,
    sample_appearance_distances,
    sweep_frame_gap,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "docs" / "assets" / "reid"

N_INTRA = 5000
N_INTER = 10000
MIN_FRAME_GAP = 1
MAX_FRAME_GAP = 30
N_SWEEP_PER_CLASS = 8000

# 0.25 is the default, both ours and BoT-SORT's ``appearance_thresh``; 0.20 is what
# these figures argue for on MOT17.
REFERENCE_THRESHOLDS = {0.20: "selected", 0.25: "default"}

MOT17_VAL_SEQS = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
]

# Same subset used for the original docs SoccerNet figure (manageable CPU/MPS run).
SOCCERNET_DOC_SEQS = [
    "SNMOT-116",
    "SNMOT-118",
    "SNMOT-120",
    "SNMOT-122",
    "SNMOT-124",
    "SNMOT-126",
    "SNMOT-130",
    "SNMOT-134",
]


def load_gt_boxes(
    gt_path: Path,
    *,
    pedestrian_only: bool,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    by_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    with gt_path.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            if pedestrian_only:
                if len(parts) < 8:
                    continue
                conf, cls = float(parts[6]), int(float(parts[7]))
                if conf <= 0 or cls != 1:
                    continue
            elif len(parts) >= 7 and float(parts[6]) <= 0:
                continue
            by_frame[frame].append((tid, np.array([x, y, x + w, y + h], dtype=np.float32)))
    return by_frame


def collect_embeddings(
    model: ReIDModel,
    sequences: list[tuple[str, Path, Path]],
    *,
    pedestrian_only: bool,
    frame_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    frame_ids: list[int] = []
    seq_ids: list[int] = []
    label_by_key: dict[str, int] = {}
    for sid, (seq, img_dir, gt_path) in enumerate(sequences):
        gt_by_frame = load_gt_boxes(gt_path, pedestrian_only=pedestrian_only)
        images = sorted(img_dir.glob("*.jpg"))
        if not images:
            images = sorted(img_dir.glob("*.png"))
        n_frames = len(images)
        n_used = 0
        for frame_idx in range(1, n_frames + 1, frame_stride):
            rows = gt_by_frame.get(frame_idx)
            if not rows:
                continue
            bgr = cv2.imread(str(images[frame_idx - 1]))
            if bgr is None:
                continue
            xyxy = np.stack([r[1] for r in rows], axis=0)
            track_ids = [r[0] for r in rows]
            feats = model.extract_features(sv.Detections(xyxy=xyxy), bgr)
            for i, tid in enumerate(track_ids):
                key = f"{seq}_{int(tid)}"
                if key not in label_by_key:
                    label_by_key[key] = len(label_by_key)
                embeddings.append(feats[i])
                labels.append(label_by_key[key])
                frame_ids.append(frame_idx)
                seq_ids.append(sid)
            n_used += 1
        print(
            f"  {seq}: {n_used} frames -> {len(embeddings)} crops total",
            flush=True,
        )
    if not embeddings:
        raise RuntimeError("No GT embeddings collected.")
    return (
        np.stack(embeddings),
        np.asarray(labels, dtype=np.int64),
        np.asarray(frame_ids, dtype=np.int64),
        np.asarray(seq_ids, dtype=np.int64),
    )


def load_or_collect_embeddings(
    cache_path: Path | None,
    build: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cache_path is not None and cache_path.is_file():
        print(f"  loading cached embeddings from {cache_path}", flush=True)
        blob = np.load(cache_path)
        return blob["emb"], blob["ids"], blob["frames"], blob["seqs"]
    emb, ids, frames, seqs = build()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, emb=emb, ids=ids, frames=frames, seqs=seqs)
        print(f"  cached embeddings to {cache_path}", flush=True)
    return emb, ids, frames, seqs


def mot17_sequences(mot17_val: Path) -> list[tuple[str, Path, Path]]:
    out: list[tuple[str, Path, Path]] = []
    for seq in MOT17_VAL_SEQS:
        img = mot17_val / seq / "img1"
        gt = mot17_val / seq / "gt" / "gt.txt"
        if not gt.is_file() or not img.is_dir():
            raise FileNotFoundError(f"Missing MOT17 val sequence: {seq} under {mot17_val}")
        out.append((seq, img, gt))
    return out


def soccernet_sequences(
    test_root: Path,
    max_seqs: int | None,
    *,
    seq_names: list[str] | None = None,
) -> list[tuple[str, Path, Path]]:
    if seq_names is None:
        seqs = sorted(p.name for p in test_root.iterdir() if p.is_dir())
    else:
        seqs = list(seq_names)
    if max_seqs is not None:
        seqs = seqs[:max_seqs]
    out: list[tuple[str, Path, Path]] = []
    for seq in seqs:
        img = test_root / seq / "img1"
        gt = test_root / seq / "gt" / "gt.txt"
        if gt.is_file() and img.is_dir():
            out.append((seq, img, gt))
    if not out:
        raise FileNotFoundError(f"No SoccerNet sequences under {test_root}")
    return out


def save(figure: Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {out_path}", flush=True)


def print_sweep_table(sweep: list[AppearanceDistances]) -> None:
    """Print the numbers the docs tables quote, so figure and prose cannot drift."""
    rate_columns = "".join(f"{f'same<{t:.2f}':>11}{f'diff<{t:.2f}':>11}" for t in REFERENCE_THRESHOLDS)
    print(f"  {'gap':>8} {'AUC':>6} {'same p50':>9} {'same p90':>9} {'diff p10':>9}{rate_columns}")
    for band in sweep:
        rates = ""
        for threshold in REFERENCE_THRESHOLDS:
            same_rate, different_rate = band.rates_at(threshold)
            rates += f"{100 * same_rate:10.1f}%{100 * different_rate:10.1f}%"
        print(
            f"  {band.label:>8} {band.roc_auc:6.3f} "
            f"{np.median(band.same_id):9.3f} {np.percentile(band.same_id, 90):9.3f} "
            f"{np.percentile(band.different_id, 10):9.3f}{rates}"
        )


def report(
    emb: np.ndarray,
    ids: np.ndarray,
    frames: np.ndarray,
    seqs: np.ndarray,
    args: argparse.Namespace,
    *,
    title: str,
    slug: str,
) -> None:
    """Write the association-local histogram and, optionally, the frame-gap sweep."""
    print(f"pool={len(emb)} crops, {len(np.unique(ids))} ids", flush=True)
    distances = sample_appearance_distances(
        emb,
        ids,
        frames,
        seqs,
        same_id_pairs=args.n_intra,
        different_id_pairs=args.n_inter,
        minimum_frame_gap=MIN_FRAME_GAP,
        maximum_frame_gap=args.max_frame_gap,
        seed=args.seed,
    )
    save(
        plot_appearance_distances(
            distances,
            thresholds=REFERENCE_THRESHOLDS,
            title=f"{title} (gap {MIN_FRAME_GAP}-{args.max_frame_gap})",
        ),
        ASSETS / f"{slug}-appearance-distances.png",
    )
    print_sweep_table([distances])
    if not args.gap_sweep:
        return
    sweep = sweep_frame_gap(
        emb,
        ids,
        frames,
        seqs,
        gap_bands=DEFAULT_FRAME_GAP_BANDS,
        pairs_per_class=args.n_sweep,
        seed=args.seed,
    )
    save(
        plot_frame_gap_sweep(
            sweep,
            thresholds=REFERENCE_THRESHOLDS,
            title=f"{title}: separability vs frame gap",
        ),
        ASSETS / f"{slug}-appearance-distances-vs-gap.png",
    )
    print_sweep_table(sweep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mot17", "soccernet", "both"), default="both")
    parser.add_argument("--mot17-val", type=Path, default=REPO_ROOT / "mot17" / "val")
    parser.add_argument(
        "--soccernet-root",
        type=Path,
        default=None,
        help="SoccerNet-tracking test root (folders SNMOT-*/img1 + gt/gt.txt)",
    )
    parser.add_argument("--soccernet-max-seqs", type=int, default=None)
    parser.add_argument(
        "--soccernet-all-seqs",
        action="store_true",
        help="Use every sequence under --soccernet-root (default: docs 8-seq subset).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache GT embeddings here so sampling can be re-run without re-encoding.",
    )
    parser.add_argument("--max-frame-gap", type=int, default=MAX_FRAME_GAP)
    parser.add_argument("--n-intra", type=int, default=N_INTRA)
    parser.add_argument("--n-inter", type=int, default=N_INTER)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gap-sweep",
        action="store_true",
        help="Also write the separability-vs-frame-gap figure.",
    )
    parser.add_argument("--n-sweep", type=int, default=N_SWEEP_PER_CLASS, help="Pairs per class per gap band.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Embed every Nth frame (1 = dense; 2-5 speeds up CPU/MPS runs).",
    )
    args = parser.parse_args()

    if args.dataset in ("mot17", "both"):
        print("=== MOT17 val / fastreid_mot17_sbs50 ===", flush=True)
        emb, ids, frames, seqs = load_or_collect_embeddings(
            None if args.cache_dir is None else args.cache_dir / f"mot17-fastreid-stride{args.frame_stride}.npz",
            lambda: collect_embeddings(
                ReIDModel.from_pretrained(FASTREID_MOT17_SBS50),
                mot17_sequences(args.mot17_val),
                pedestrian_only=True,
                frame_stride=args.frame_stride,
            ),
        )
        report(
            emb,
            ids,
            frames,
            seqs,
            args,
            title="fastreid_mot17_sbs50 on MOT17 val GT",
            slug="mot17-fastreid",
        )

    if args.dataset in ("soccernet", "both"):
        root = args.soccernet_root
        if root is None:
            raise SystemExit(
                "Pass --soccernet-root to the SoccerNet-tracking test directory (contains SNMOT-*/img1 and gt/gt.txt)."
            )
        # Same association-local protocol as the MOT17 branch above.
        print("=== SoccerNet test / osnet_x1_0_msmt17_combineall ===", flush=True)
        emb, ids, frames, seqs = load_or_collect_embeddings(
            None if args.cache_dir is None else args.cache_dir / f"soccernet-osnet-stride{args.frame_stride}.npz",
            lambda: collect_embeddings(
                ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall"),
                soccernet_sequences(
                    root,
                    args.soccernet_max_seqs,
                    seq_names=None if args.soccernet_all_seqs else SOCCERNET_DOC_SEQS,
                ),
                pedestrian_only=False,
                frame_stride=args.frame_stride,
            ),
        )
        report(
            emb,
            ids,
            frames,
            seqs,
            args,
            title="osnet_x1_0_msmt17_combineall on SoccerNet test GT",
            slug="soccernet-osnet",
        )


if __name__ == "__main__":
    main()
