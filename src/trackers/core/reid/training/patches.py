# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Generate per-identity ReID crops from MOT-format sequences.

Use ``split="train_half"`` to stay frame-disjoint from the standard MOT17
val-half tracking evaluation.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# MOT17 sequences are released three times, once per public detector. The
# frames and ground truth are identical across variants, so we keep only one
# variant per base sequence to avoid emitting triplicate crops.
_DETECTOR_SUFFIXES = ("-FRCNN", "-DPM", "-SDP")

# MOT Challenge class id for the "pedestrian" category we train re-ID on.
_PEDESTRIAN_CLASS = 1

# Column layout of a MOT ground-truth row:
# frame, id, bb_left, bb_top, bb_width, bb_height, conf/flag, class, visibility
_COL_FRAME = 0
_COL_ID = 1
_COL_BBOX = slice(2, 6)
_COL_CONF = 6
_COL_CLASS = 7
_COL_VISIBILITY = 8

_VALID_SPLITS = ("train_half", "val_half", "full")


@dataclass
class PatchGenerationStats:
    """Summary counts from :func:`generate_mot_patches`."""

    sequences: list[str] = field(default_factory=list)
    num_identities: int = 0
    num_crops: int = 0
    crops_per_sequence: dict[str, int] = field(default_factory=dict)
    identities_per_sequence: dict[str, int] = field(default_factory=dict)
    skipped_low_visibility: int = 0
    skipped_small: int = 0
    skipped_degenerate: int = 0


def _sequence_base_name(name: str) -> str:
    """Strip the public-detector suffix from a MOT17 sequence name."""
    for suffix in _DETECTOR_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _read_seq_length(sequence_dir: Path) -> int | None:
    """Read ``seqLength`` from a sequence's ``seqinfo.ini`` if present."""
    seqinfo = sequence_dir / "seqinfo.ini"
    if not seqinfo.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(seqinfo)
        return int(parser["Sequence"]["seqLength"])
    except (configparser.Error, KeyError, ValueError):
        return None


def _load_gt_rows(gt_path: Path) -> np.ndarray:
    """Load a MOT ground-truth file as a ``(N, >=9)`` float array.

    Parsed directly (rather than via :func:`trackers.io.mot.load_mot_file`)
    because crop generation needs the visibility column, which that loader
    does not retain.
    """
    rows = np.loadtxt(gt_path, delimiter=",", ndmin=2)
    if rows.size == 0:
        return np.empty((0, 9), dtype=np.float64)
    if rows.shape[1] < 8:
        raise ValueError(f"Malformed MOT ground truth {gt_path}: expected >= 8 columns, got {rows.shape[1]}.")
    return rows


def _train_half_cutoff(sequence_dir: Path, gt_rows: np.ndarray) -> int:
    """Return ``L // 2``, the last frame index belonging to the train-half."""
    length = _read_seq_length(sequence_dir)
    if length is None:
        length = int(gt_rows[:, _COL_FRAME].max()) if len(gt_rows) else 0
    return length // 2


def _select_split_mask(
    gt_rows: np.ndarray,
    split: str,
    cutoff: int,
) -> np.ndarray:
    """Boolean mask selecting rows whose frame belongs to ``split``."""
    frames = gt_rows[:, _COL_FRAME]
    if split == "train_half":
        return frames <= cutoff
    if split == "val_half":
        return frames > cutoff
    return np.ones(len(gt_rows), dtype=bool)


def _discover_sequences(mot_root: Path) -> list[Path]:
    """Find sequence directories under ``mot_root`` that have GT and frames."""
    sequences = []
    for child in sorted(mot_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "gt" / "gt.txt").exists() and (child / "img1").is_dir():
            sequences.append(child)
    return sequences


def _dedupe_detector_variants(
    sequence_dirs: list[Path],
    prefer_detector: str,
) -> list[Path]:
    """Keep one directory per base sequence, preferring ``prefer_detector``."""
    preferred_suffix = f"-{prefer_detector}"
    by_base: dict[str, Path] = {}
    for sequence_dir in sequence_dirs:
        base = _sequence_base_name(sequence_dir.name)
        current = by_base.get(base)
        if current is None or sequence_dir.name.endswith(preferred_suffix):
            by_base[base] = sequence_dir
    return [by_base[base] for base in sorted(by_base)]


def _crop_rows_from_sequence(
    seq_name: str,
    gt_path: Path,
    image_dir: Path,
    output_dir: Path,
    *,
    split: str,
    min_visibility: float,
    min_side: int,
    image_ext: str,
    stats: PatchGenerationStats,
    identities: set[str],
) -> None:
    """Crop one sequence and update ``stats`` / ``identities`` in place."""
    base = _sequence_base_name(seq_name)
    gt_rows = _load_gt_rows(gt_path)
    if len(gt_rows) == 0:
        return

    if split == "full":
        cutoff = int(gt_rows[:, _COL_FRAME].max()) if len(gt_rows) else 0
    else:
        cutoff = _train_half_cutoff(image_dir.parent, gt_rows)

    keep = _select_split_mask(gt_rows, split, cutoff)
    keep &= gt_rows[:, _COL_CLASS].astype(int) == _PEDESTRIAN_CLASS
    keep &= gt_rows[:, _COL_CONF] > 0
    rows = gt_rows[keep]

    seq_crops = 0
    seq_identities: set[str] = set()
    frame_cache: dict[int, np.ndarray | None] = {}

    for row in rows:
        visibility = row[_COL_VISIBILITY] if len(row) > _COL_VISIBILITY else 1.0
        if visibility < min_visibility:
            stats.skipped_low_visibility += 1
            continue

        frame = int(row[_COL_FRAME])
        if frame not in frame_cache:
            image_path = image_dir / f"{frame:06d}.{image_ext}"
            frame_cache[frame] = cv2.imread(str(image_path)) if image_path.exists() else None
        image = frame_cache[frame]
        if image is None:
            stats.skipped_degenerate += 1
            continue

        height, width = image.shape[:2]
        x, y, w, h = (float(v) for v in row[_COL_BBOX])
        x1 = max(0, round(x))
        y1 = max(0, round(y))
        x2 = min(width, round(x + w))
        y2 = min(height, round(y + h))
        if x2 <= x1 or y2 <= y1:
            stats.skipped_degenerate += 1
            continue
        if min_side > 0 and min(x2 - x1, y2 - y1) < min_side:
            stats.skipped_small += 1
            continue

        identity = f"{base}_{int(row[_COL_ID])}"
        identity_dir = output_dir / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        crop_path = identity_dir / f"{base}_{frame:06d}.{image_ext}"
        cv2.imwrite(str(crop_path), image[y1:y2, x1:x2])

        seq_crops += 1
        seq_identities.add(identity)
        identities.add(identity)

    if seq_crops:
        stats.sequences.append(base)
        stats.crops_per_sequence[base] = seq_crops
        stats.identities_per_sequence[base] = len(seq_identities)
        stats.num_crops += seq_crops


def generate_mot_patches(
    mot_root: str | Path,
    output_dir: str | Path,
    *,
    split: str = "train_half",
    sequences: list[str] | None = None,
    min_visibility: float = 0.0,
    min_side: int = 0,
    prefer_detector: str = "FRCNN",
    dedupe_detectors: bool = True,
    image_ext: str = "jpg",
) -> PatchGenerationStats:
    """Crop MOT ground-truth boxes to ``output_dir/<seq>_<id>/<seq>_<frame>.jpg``.

    Args:
        mot_root: MOT root with sequence folders (``gt/gt.txt``, ``img1/``).
        output_dir: Destination crop root.
        split: ``"train_half"``, ``"val_half"``, or ``"full"``.
        sequences: Optional sequence names to process.
        min_visibility: Minimum GT visibility to keep a box.
        min_side: Minimum crop short side in pixels.
        prefer_detector: Preferred MOT17 detector variant when de-duplicating.
        dedupe_detectors: Keep one detector variant per base sequence.
        image_ext: Frame file extension.

    Returns:
        :class:`PatchGenerationStats`.
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}.")

    mot_root = Path(mot_root)
    if not mot_root.is_dir():
        raise FileNotFoundError(f"MOT root not found: {mot_root}")

    output_dir = Path(output_dir)

    if sequences is not None:
        sequence_dirs = [mot_root / name for name in sequences]
        missing = [d.name for d in sequence_dirs if not d.is_dir()]
        if missing:
            raise FileNotFoundError(f"Requested sequences not found under {mot_root}: {missing}")
    else:
        sequence_dirs = _discover_sequences(mot_root)

    if not sequence_dirs:
        raise FileNotFoundError(f"No valid MOT sequences found under {mot_root}")

    if dedupe_detectors:
        sequence_dirs = _dedupe_detector_variants(sequence_dirs, prefer_detector)

    stats = PatchGenerationStats()
    identities: set[str] = set()

    for sequence_dir in sequence_dirs:
        _crop_rows_from_sequence(
            sequence_dir.name,
            sequence_dir / "gt" / "gt.txt",
            sequence_dir / "img1",
            output_dir,
            split=split,
            min_visibility=min_visibility,
            min_side=min_side,
            image_ext=image_ext,
            stats=stats,
            identities=identities,
        )

    stats.num_identities = len(identities)
    return stats
