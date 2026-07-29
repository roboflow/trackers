#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Regenerate docs appearance-distance histograms (association-local sampling).

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
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from reid import FASTREID_MOT17_SBS50, ReIDModel

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "docs" / "assets" / "reid"

N_INTRA = 5000
N_INTER = 10000
MIN_FRAME_GAP = 1
MAX_FRAME_GAP = 30
BINS = np.linspace(0.0, 1.0, 51)

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


def _d_app(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * (1.0 - float(a @ b))


def sample_association_local(
    embeddings: np.ndarray,
    gt_ids: np.ndarray,
    frame_ids: np.ndarray,
    seq_ids: np.ndarray,
    *,
    n_intra: int,
    n_inter: int,
    min_frame_gap: int,
    max_frame_gap: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(seed)
    by_seq_id: dict[tuple[int, int], list[int]] = defaultdict(list)
    by_seq: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(gt_ids)):
        by_seq_id[(int(seq_ids[idx]), int(gt_ids[idx]))].append(idx)
        by_seq[int(seq_ids[idx])].append(idx)

    pos_pairs: list[tuple[int, int]] = []
    for idxs in by_seq_id.values():
        for a in range(len(idxs)):
            ia, fa = idxs[a], int(frame_ids[idxs[a]])
            for b in range(a + 1, len(idxs)):
                ib = idxs[b]
                gap = abs(fa - int(frame_ids[ib]))
                if min_frame_gap <= gap <= max_frame_gap:
                    pos_pairs.append((ia, ib))
    if not pos_pairs:
        raise ValueError("No same-ID association-local pairs.")

    neg_pairs: list[tuple[int, int]] = []
    max_neg_pool = max(n_inter * 40, 50_000)
    for idxs in by_seq.values():
        idxs_sorted = sorted(idxs, key=lambda i: int(frame_ids[i]))
        frames = [int(frame_ids[i]) for i in idxs_sorted]
        for a, ia in enumerate(idxs_sorted):
            fa, pid_a = frames[a], int(gt_ids[ia])
            b = a + 1
            while b < len(idxs_sorted) and frames[b] - fa <= max_frame_gap:
                ib = idxs_sorted[b]
                if int(gt_ids[ib]) != pid_a:
                    neg_pairs.append((ia, ib))
                    if len(neg_pairs) >= max_neg_pool:
                        break
                b += 1
            if len(neg_pairs) >= max_neg_pool:
                break
        if len(neg_pairs) >= max_neg_pool:
            break
    if not neg_pairs:
        raise ValueError("No different-ID association-local pairs.")

    intra = np.empty(n_intra)
    inter = np.empty(n_inter)
    for k in range(n_intra):
        i, j = pos_pairs[int(rng.integers(len(pos_pairs)))]
        intra[k] = _d_app(normed[i], normed[j])
    for k in range(n_inter):
        i, j = neg_pairs[int(rng.integers(len(neg_pairs)))]
        inter[k] = _d_app(normed[i], normed[j])
    return intra, inter


def plot_and_save(
    intra: np.ndarray,
    inter: np.ndarray,
    *,
    title: str,
    theta: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        intra,
        bins=BINS,
        weights=np.full(len(intra), 1.0 / len(intra)),
        alpha=0.65,
        label=f"same-ID (n={len(intra)})",
        color="#3366CC",
    )
    ax.hist(
        inter,
        bins=BINS,
        weights=np.full(len(inter), 1.0 / len(inter)),
        alpha=0.65,
        label=f"diff-ID (n={len(inter)})",
        color="#DC3912",
    )
    ax.axvline(0.25, color="#666666", ls=":", lw=1.5, label="θ=0.25 (default)")
    ax.axvline(theta, color="#111111", ls="--", lw=1.8, label=f"θ={theta:.2f} (selected)")
    ax.set(
        xlabel=r"$0.5\cdot$ cosine distance  ($0.5\cdot(1-\cos)$)",
        ylabel="probability",
        title=title,
        xlim=(0.0, 0.6),
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    for t in (0.10, 0.20, 0.25):
        print(f"  θ={t:.2f}: same-ID<{t}={100 * np.mean(intra < t):.1f}%  diff-ID<{t}={100 * np.mean(inter < t):.1f}%")


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
    parser.add_argument("--max-frame-gap", type=int, default=MAX_FRAME_GAP)
    parser.add_argument("--n-intra", type=int, default=N_INTRA)
    parser.add_argument("--n-inter", type=int, default=N_INTER)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Embed every Nth frame (1 = dense; 2-5 speeds up CPU/MPS runs).",
    )
    args = parser.parse_args()

    if args.dataset in ("mot17", "both"):
        print("=== MOT17 val / fastreid_mot17_sbs50 ===", flush=True)
        model = ReIDModel.from_pretrained(FASTREID_MOT17_SBS50)
        emb, ids, frames, seqs = collect_embeddings(
            model,
            mot17_sequences(args.mot17_val),
            pedestrian_only=True,
            frame_stride=args.frame_stride,
        )
        print(f"pool={len(emb)} crops, {len(np.unique(ids))} ids", flush=True)
        intra, inter = sample_association_local(
            emb,
            ids,
            frames,
            seqs,
            n_intra=args.n_intra,
            n_inter=args.n_inter,
            min_frame_gap=MIN_FRAME_GAP,
            max_frame_gap=args.max_frame_gap,
        )
        plot_and_save(
            intra,
            inter,
            title="fastreid_mot17_sbs50 on MOT17 val GT",
            theta=0.2,
            out_path=ASSETS / "mot17-fastreid-appearance-distances.png",
        )

    if args.dataset in ("soccernet", "both"):
        root = args.soccernet_root
        if root is None:
            raise SystemExit(
                "Pass --soccernet-root to the SoccerNet-tracking test directory (contains SNMOT-*/img1 and gt/gt.txt)."
            )
        # Same association-local protocol as notebooks/eval_trackers_reid.ipynb §6b.
        print("=== SoccerNet test / osnet_x1_0_msmt17_combineall ===", flush=True)
        model = ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")
        emb, ids, frames, seqs = collect_embeddings(
            model,
            soccernet_sequences(
                root,
                args.soccernet_max_seqs,
                seq_names=None if args.soccernet_all_seqs else SOCCERNET_DOC_SEQS,
            ),
            pedestrian_only=False,
            frame_stride=args.frame_stride,
        )
        print(f"pool={len(emb)} crops, {len(np.unique(ids))} ids", flush=True)
        intra, inter = sample_association_local(
            emb,
            ids,
            frames,
            seqs,
            n_intra=args.n_intra,
            n_inter=args.n_inter,
            min_frame_gap=MIN_FRAME_GAP,
            max_frame_gap=args.max_frame_gap,
        )
        plot_and_save(
            intra,
            inter,
            title="osnet_x1_0_msmt17_combineall on SoccerNet test GT",
            theta=0.2,
            out_path=ASSETS / "soccernet-osnet-appearance-distances.png",
        )


if __name__ == "__main__":
    main()
