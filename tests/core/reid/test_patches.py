# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from trackers.core.reid.training.patches import generate_mot_patches

# A ground-truth row: frame, id, x, y, w, h, conf/flag, class, visibility
_FRAME_H, _FRAME_W = 128, 64


def _write_sequence(
    root: Path,
    name: str,
    seq_length: int,
    rows: list[list[float]],
) -> Path:
    """Create a minimal MOT sequence (seqinfo.ini, img1/ frames, gt/gt.txt)."""
    sequence_dir = root / name
    (sequence_dir / "img1").mkdir(parents=True)
    (sequence_dir / "gt").mkdir(parents=True)

    (sequence_dir / "seqinfo.ini").write_text(f"[Sequence]\nname={name}\nseqLength={seq_length}\n")

    for frame in range(1, seq_length + 1):
        image = np.full((_FRAME_H, _FRAME_W, 3), frame % 256, dtype=np.uint8)
        cv2.imwrite(str(sequence_dir / "img1" / f"{frame:06d}.jpg"), image)

    lines = [",".join(str(value) for value in row) for row in rows]
    (sequence_dir / "gt" / "gt.txt").write_text("\n".join(lines) + "\n")
    return sequence_dir


def _box(frame: int, identity: int, *, conf: int = 1, cls: int = 1, vis: float = 1.0):
    return [frame, identity, 5, 5, 20, 40, conf, cls, vis]


def _pedestrian_rows(seq_length: int, identities: list[int]) -> list[list[float]]:
    return [_box(frame, identity) for identity in identities for frame in range(1, seq_length + 1)]


def test_train_half_split_is_frame_disjoint_from_val_half(tmp_path: Path) -> None:
    mot_root = tmp_path / "MOT"
    # seqLength=4 -> cutoff L//2 = 2: train-half frames {1,2}, val-half {3,4}.
    _write_sequence(mot_root, "SEQ-01", seq_length=4, rows=_pedestrian_rows(4, [1, 2]))

    train_out = tmp_path / "train"
    val_out = tmp_path / "val"
    train_stats = generate_mot_patches(mot_root, train_out, split="train_half")
    val_stats = generate_mot_patches(mot_root, val_out, split="val_half")

    def crop_frames(out: Path) -> set[int]:
        return {int(p.stem.split("_")[-1]) for p in out.rglob("*.jpg")}

    assert crop_frames(train_out) == {1, 2}
    assert crop_frames(val_out) == {3, 4}
    # 2 identities x 2 frames each.
    assert train_stats.num_crops == 4
    assert val_stats.num_crops == 4
    assert train_stats.num_identities == 2


def test_identity_folders_are_globally_unique_across_sequences(tmp_path: Path) -> None:
    mot_root = tmp_path / "MOT"
    _write_sequence(mot_root, "SEQ-01", seq_length=2, rows=_pedestrian_rows(2, [1]))
    _write_sequence(mot_root, "SEQ-02", seq_length=2, rows=_pedestrian_rows(2, [1]))

    out = tmp_path / "out"
    stats = generate_mot_patches(mot_root, out, split="full")

    identity_dirs = sorted(p.name for p in out.iterdir() if p.is_dir())
    # Same numeric id 1 in two sequences -> two distinct identities.
    assert identity_dirs == ["SEQ-01_1", "SEQ-02_1"]
    assert stats.num_identities == 2


def test_filters_distractors_low_conf_and_low_visibility(tmp_path: Path) -> None:
    mot_root = tmp_path / "MOT"
    rows = [
        _box(1, 1),  # kept
        _box(2, 1),  # kept
        _box(1, 2, cls=2),  # dropped: not pedestrian
        _box(2, 3, conf=0),  # dropped: ignore flag
        _box(1, 4, vis=0.1),  # dropped by min_visibility=0.5
    ]
    _write_sequence(mot_root, "SEQ-01", seq_length=2, rows=rows)

    out = tmp_path / "out"
    stats = generate_mot_patches(mot_root, out, split="full", min_visibility=0.5)

    assert stats.num_identities == 1
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["SEQ-01_1"]
    assert stats.num_crops == 2
    assert stats.skipped_low_visibility == 1


def test_detector_variants_are_deduplicated(tmp_path: Path) -> None:
    mot_root = tmp_path / "MOT"
    _write_sequence(mot_root, "MOT17-02-FRCNN", seq_length=2, rows=_pedestrian_rows(2, [1]))
    _write_sequence(mot_root, "MOT17-02-SDP", seq_length=2, rows=_pedestrian_rows(2, [1]))

    out = tmp_path / "out"
    stats = generate_mot_patches(mot_root, out, split="full")

    # Identical frames/GT across detectors collapse to one base sequence.
    assert stats.sequences == ["MOT17-02"]
    assert stats.num_identities == 1
    assert stats.num_crops == 2


def test_min_side_drops_small_crops(tmp_path: Path) -> None:
    mot_root = tmp_path / "MOT"
    rows = [
        [1, 1, 5, 5, 20, 40, 1, 1, 1.0],  # 20x40 kept
        [1, 2, 5, 5, 4, 40, 1, 1, 1.0],  # width 4 -> dropped by min_side=10
    ]
    _write_sequence(mot_root, "SEQ-01", seq_length=2, rows=rows)

    out = tmp_path / "out"
    stats = generate_mot_patches(mot_root, out, split="full", min_side=10)

    assert stats.num_identities == 1
    assert stats.skipped_small == 1


def test_invalid_split_raises(tmp_path: Path) -> None:
    mot_root = tmp_path / "MOT"
    _write_sequence(mot_root, "SEQ-01", seq_length=2, rows=_pedestrian_rows(2, [1]))
    with pytest.raises(ValueError, match="split must be one of"):
        generate_mot_patches(mot_root, tmp_path / "out", split="nope")
