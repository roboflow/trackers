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
from collections.abc import Callable
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

# Gap bands for the sweep. The first four sit inside the default lost-track buffer
# (30 frames); the rest probe re-association after longer occlusions.
GAP_BUCKETS = [(1, 1), (2, 5), (6, 15), (16, 30), (31, 60), (61, 120), (121, 240)]
N_SWEEP_PER_CLASS = 8000
# Symmetric band so both classes are read the same way, wide enough to show the
# tails that decide where the two distributions start to overlap.
BAND_PERCENTILES = (10, 90)

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


def _d_app(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * (1.0 - float(a @ b))


class SequenceIndex:
    """Frame-sorted crop index for one sequence, with O(log n) frame-window lookup.

    ``order`` holds global crop indexes sorted by frame; ``frames`` is the matching frame array. ``positions_by_id``
    maps a GT id to its slots inside ``order``, which are themselves frame-sorted because the sort is stable.
    """

    def __init__(self, global_indexes: np.ndarray, gt_ids: np.ndarray, frame_ids: np.ndarray) -> None:
        order = np.argsort(frame_ids[global_indexes], kind="stable")
        self.order = global_indexes[order]
        self.frames = frame_ids[self.order]
        self.ids = gt_ids[self.order]
        positions_by_id: dict[int, list[int]] = defaultdict(list)
        for slot, pid in enumerate(self.ids):
            positions_by_id[int(pid)].append(slot)
        self.positions_by_id = {pid: np.asarray(slots) for pid, slots in positions_by_id.items()}
        self.pairable_ids = [pid for pid, slots in self.positions_by_id.items() if len(slots) > 1]

    def window(self, frames: np.ndarray, anchor_frame: int, lo: int, hi: int) -> tuple[int, int, int, int]:
        """Half-open slot ranges of ``frames`` whose gap to ``anchor_frame`` is in ``[lo, hi]``."""
        before = (
            int(np.searchsorted(frames, anchor_frame - hi, side="left")),
            int(np.searchsorted(frames, anchor_frame - lo, side="right")),
        )
        after = (
            int(np.searchsorted(frames, anchor_frame + lo, side="left")),
            int(np.searchsorted(frames, anchor_frame + hi, side="right")),
        )
        return before[0], before[1], after[0], after[1]


def _pick_in_window(rng: np.random.Generator, window: tuple[int, int, int, int]) -> int | None:
    """Uniformly pick one slot across the two half-open ranges, or ``None`` if empty."""
    b_lo, b_hi, a_lo, a_hi = window
    n_before = max(0, b_hi - b_lo)
    n_after = max(0, a_hi - a_lo)
    total = n_before + n_after
    if total == 0:
        return None
    draw = int(rng.integers(total))
    return b_lo + draw if draw < n_before else a_lo + (draw - n_before)


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
    max_tries_per_sample: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample same-ID and different-ID crop pairs inside the association horizon.

    Pairs are drawn directly rather than enumerated into a pool, so no cap can bias the result toward whichever sequence
    happens to be visited first. Each sequence gets an equal quota. Within a sequence, same-ID pairs pick an identity
    uniformly (so long tracks do not dominate) and different-ID pairs pick an anchor crop uniformly, then a partner
    uniformly among the crops inside the gap band.
    """
    # A zero gap would let a crop pair with itself, which is the artefact that puts a
    # spike at distance 0 in the original BoT-SORT figure.
    if min_frame_gap < 1 or max_frame_gap < min_frame_gap:
        raise ValueError(f"invalid gap band [{min_frame_gap}, {max_frame_gap}], expected 1 <= min <= max")
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(seed)

    unique_seqs = np.unique(seq_ids)
    indexes = {int(sid): SequenceIndex(np.flatnonzero(seq_ids == sid), gt_ids, frame_ids) for sid in unique_seqs}

    intra: list[float] = []
    inter: list[float] = []
    for quota, out, same_id in ((n_intra, intra, True), (n_inter, inter, False)):
        per_seq = _split_quota(quota, len(unique_seqs))
        for sid, n_wanted in zip(sorted(indexes), per_seq, strict=True):
            index = indexes[sid]
            drawn = 0
            attempts = 0
            budget = n_wanted * max_tries_per_sample
            while drawn < n_wanted and attempts < budget:
                attempts += 1
                pair = (
                    _draw_same_id_pair(rng, index, min_frame_gap, max_frame_gap)
                    if same_id
                    else _draw_diff_id_pair(rng, index, min_frame_gap, max_frame_gap)
                )
                if pair is None:
                    continue
                i, j = pair
                out.append(_d_app(normed[i], normed[j]))
                drawn += 1
            if drawn < n_wanted:
                label = "same-ID" if same_id else "different-ID"
                print(
                    f"  warning: sequence {sid} yielded {drawn}/{n_wanted} {label} pairs "
                    f"in gap band [{min_frame_gap}, {max_frame_gap}]",
                    flush=True,
                )
    if not intra or not inter:
        raise ValueError(f"No association-local pairs in gap band [{min_frame_gap}, {max_frame_gap}].")
    return np.asarray(intra), np.asarray(inter)


def _split_quota(total: int, n_buckets: int) -> list[int]:
    base, remainder = divmod(total, n_buckets)
    return [base + (1 if k < remainder else 0) for k in range(n_buckets)]


def _draw_same_id_pair(
    rng: np.random.Generator,
    index: SequenceIndex,
    lo: int,
    hi: int,
) -> tuple[int, int] | None:
    if not index.pairable_ids:
        return None
    slots = index.positions_by_id[index.pairable_ids[int(rng.integers(len(index.pairable_ids)))]]
    anchor = int(slots[int(rng.integers(len(slots)))])
    partner_slot = _pick_in_window(rng, index.window(index.frames[slots], int(index.frames[anchor]), lo, hi))
    if partner_slot is None:
        return None
    partner = int(slots[partner_slot])
    if partner == anchor:
        return None
    return int(index.order[anchor]), int(index.order[partner])


def _draw_diff_id_pair(
    rng: np.random.Generator,
    index: SequenceIndex,
    lo: int,
    hi: int,
) -> tuple[int, int] | None:
    anchor = int(rng.integers(len(index.order)))
    partner = _pick_in_window(rng, index.window(index.frames, int(index.frames[anchor]), lo, hi))
    if partner is None or index.ids[partner] == index.ids[anchor]:
        return None
    return int(index.order[anchor]), int(index.order[partner])


def roc_auc(intra: np.ndarray, inter: np.ndarray) -> float:
    """P(same-ID distance < different-ID distance), ties counted as half.

    Threshold-free separability, so it needs no true-positive or false-positive target. 1.0 means the two distributions
    are disjoint, 0.5 means appearance carries no information.
    """
    inter_sorted = np.sort(inter)
    n_greater = len(inter) - np.searchsorted(inter_sorted, intra, side="right")
    n_equal = np.searchsorted(inter_sorted, intra, side="right") - np.searchsorted(inter_sorted, intra, side="left")
    return float(np.mean((n_greater + 0.5 * n_equal) / len(inter)))


def sweep_frame_gap(
    embeddings: np.ndarray,
    gt_ids: np.ndarray,
    frame_ids: np.ndarray,
    seq_ids: np.ndarray,
    *,
    buckets: list[tuple[int, int]],
    n_per_class: int,
    seed: int,
) -> list[dict[str, object]]:
    """Measure separability inside each frame-gap band."""
    rows: list[dict[str, object]] = []
    for lo, hi in buckets:
        try:
            intra, inter = sample_association_local(
                embeddings,
                gt_ids,
                frame_ids,
                seq_ids,
                n_intra=n_per_class,
                n_inter=n_per_class,
                min_frame_gap=lo,
                max_frame_gap=hi,
                seed=seed,
            )
        except ValueError as exc:
            print(f"  gap [{lo},{hi}]: skipped ({exc})", flush=True)
            continue
        rows.append(
            {
                "lo": lo,
                "hi": hi,
                "label": f"{lo}" if lo == hi else f"{lo}-{hi}",
                "intra": intra,
                "inter": inter,
                "auc": roc_auc(intra, inter),
            }
        )
    return rows


def plot_gap_sweep(rows: list[dict[str, object]], *, title: str, out_path: Path) -> None:
    """Plot how the same-ID and different-ID distance ranges move with the frame gap.

    Deliberately threshold-free: the bands are distribution quantiles, and the only
    thresholds drawn are the ones Trackers actually ships, as reference lines.
    """
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    lo_pct, hi_pct = BAND_PERCENTILES
    intra_q = np.array([np.percentile(r["intra"], [lo_pct, 50, hi_pct]) for r in rows])
    inter_q = np.array([np.percentile(r["inter"], [lo_pct, 50, hi_pct]) for r in rows])

    fig, (ax_dist, ax_auc) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    ax_dist.fill_between(x, intra_q[:, 0], intra_q[:, 2], color="#3366CC", alpha=0.22)
    ax_dist.plot(x, intra_q[:, 1], color="#3366CC", marker="o", lw=2, label="same ID")
    ax_dist.fill_between(x, inter_q[:, 0], inter_q[:, 2], color="#DC3912", alpha=0.22)
    ax_dist.plot(x, inter_q[:, 1], color="#DC3912", marker="o", lw=2, label="different ID")
    ax_dist.axhline(0.20, color="#111111", ls="--", lw=1.5, label="θ = 0.20 (Trackers)")
    ax_dist.axhline(0.25, color="#666666", ls=":", lw=1.5, label="θ = 0.25 (BoT-SORT)")
    ax_dist.set(ylabel="appearance distance")
    ax_dist.set_title(
        f"line = median, shaded = {lo_pct}th to {hi_pct}th percentile",
        fontsize=8.5,
        color="#333333",
        pad=4,
    )
    fig.suptitle(title, y=0.995)
    ax_dist.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.92, edgecolor="none")
    ax_dist.grid(True, alpha=0.25)

    aucs = [r["auc"] for r in rows]
    ax_auc.plot(x, aucs, color="#111111", marker="s", lw=2)
    for xi, auc in zip(x, aucs):
        ax_auc.annotate(
            f"{auc:.3f}",
            (xi, auc),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7.5,
            color="#111111",
        )
    ax_auc.axhline(0.5, color="#999999", ls=":", lw=1.2)
    ax_auc.set(
        xlabel="frames between the two crops",
        ylabel="separability",
        ylim=(0.42, 1.12),
        xticks=x,
        xticklabels=labels,
    )
    ax_auc.set_title(
        "take one same-ID and one different-ID pair at random: how often is the same-ID one closer?"
        "\n1.0 = always, 0.5 = coin flip. Counts every sampled pair, not the shaded overlap above.",
        fontsize=8,
        color="#333333",
        pad=4,
    )
    ax_auc.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    header = f"  {'gap':>8} {'AUC':>6} {'same p50':>9} {f'same p{hi_pct}':>9} {f'diff p{lo_pct}':>9}"
    print(f"{header} {'T<0.20':>7} {'F<0.20':>7} {'T<0.25':>7} {'F<0.25':>7}")
    for r in rows:
        intra, inter = r["intra"], r["inter"]
        print(
            f"  {r['label']:>8} {r['auc']:6.3f} {np.median(intra):9.3f} {np.percentile(intra, hi_pct):9.3f} "
            f"{np.percentile(inter, lo_pct):9.3f} {100 * np.mean(intra < 0.20):6.1f}% "
            f"{100 * np.mean(inter < 0.20):6.1f}% "
            f"{100 * np.mean(intra < 0.25):6.1f}% {100 * np.mean(inter < 0.25):6.1f}%"
        )


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
    intra, inter = sample_association_local(
        emb,
        ids,
        frames,
        seqs,
        n_intra=args.n_intra,
        n_inter=args.n_inter,
        min_frame_gap=MIN_FRAME_GAP,
        max_frame_gap=args.max_frame_gap,
        seed=args.seed,
    )
    plot_and_save(
        intra,
        inter,
        title=f"{title} (gap {MIN_FRAME_GAP}-{args.max_frame_gap})",
        theta=0.2,
        out_path=ASSETS / f"{slug}-appearance-distances.png",
    )
    if not args.gap_sweep:
        return
    rows = sweep_frame_gap(
        emb,
        ids,
        frames,
        seqs,
        buckets=GAP_BUCKETS,
        n_per_class=args.n_sweep,
        seed=args.seed,
    )
    plot_gap_sweep(
        rows,
        title=f"{title}: separability vs frame gap",
        out_path=ASSETS / f"{slug}-appearance-distances-vs-gap.png",
    )


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
