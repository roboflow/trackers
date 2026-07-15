# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Re-ID benchmark dataset loaders (Market-1501, MSMT17)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Market-1501 gallery ``0000_*`` images are distractors (pid 0). Query identities
# remain valid when pid=0 on other datasets (e.g. MSMT17).
MARKET1501_GALLERY_JUNK_PIDS = frozenset({-1, 0})


@dataclass
class ReIDSplit:
    """Query or gallery split: image paths, person IDs, and camera IDs.

    ``gallery_junk_pids`` controls which gallery person IDs are excluded from
    ranking during retrieval evaluation (see :func:`compute_reid_metrics`).
    """

    image_paths: list[str]
    pids: np.ndarray
    camids: np.ndarray
    gallery_junk_pids: frozenset[int] = frozenset({-1})

    def __len__(self) -> int:
        return len(self.image_paths)


# --------------------------------------------------------------------------- #
# MSMT17
# --------------------------------------------------------------------------- #


def _parse_msmt17_camid(filename: str) -> int:
    """Extract 0-indexed camid from an MSMT17 image filename.

    Filename format: ``<pid>_<idx>_<camid>_<scene>_<frame>_<track>.jpg``
    Example: ``0001_019_07_0303morning_0020_1.jpg`` → camid = 6 (0-indexed).
    """
    stem = Path(filename).stem
    return int(stem.split("_")[2]) - 1


def load_msmt17(root: str | Path) -> tuple[ReIDSplit, ReIDSplit]:
    """Load MSMT17 query and gallery splits from *root*.

    Expects ``test/``, ``list_query.txt``, and ``list_gallery.txt``. List files
    may be 2-column (``path pid``; camid from filename) or 3-column.

    Args:
        root: Path to the MSMT17 directory.

    Returns:
        ``(query, gallery)`` :class:`ReIDSplit` pair.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"MSMT17 root not found: {root}")

    def _parse_list(list_file: Path, image_root: Path) -> ReIDSplit:
        if not list_file.exists():
            raise FileNotFoundError(f"MSMT17 list file not found: {list_file}")
        paths, pids, camids = [], [], []
        for line in list_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rel_path = parts[0]
            pid = int(parts[1])
            camid = int(parts[2]) if len(parts) >= 3 else _parse_msmt17_camid(Path(rel_path).name)
            paths.append(str(image_root / rel_path))
            pids.append(pid)
            camids.append(camid)
        return ReIDSplit(
            image_paths=paths,
            pids=np.array(pids, dtype=np.int32),
            camids=np.array(camids, dtype=np.int32),
            gallery_junk_pids=frozenset({-1}),
        )

    test_root = root / "test"
    query = _parse_list(root / "list_query.txt", test_root)
    gallery = _parse_list(root / "list_gallery.txt", test_root)
    return query, gallery


# --------------------------------------------------------------------------- #
# Market-1501
# --------------------------------------------------------------------------- #


def _parse_market_filename(filename: str) -> tuple[int, int]:
    """Parse ``(pid, camid)`` from a Market-1501 filename.

    ``-1_*`` → junk (``pid=-1``); ``0000_*`` → distractor (``pid=0``).
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    pid = int(parts[0])
    camid = int(parts[1][1]) - 1  # "c1" → 0
    return pid, camid


def load_market1501(root: str | Path) -> tuple[ReIDSplit, ReIDSplit]:
    """Load Market-1501 query and gallery splits from *root*.

    Expects ``query/`` and ``bounding_box_test/``. Person and camera IDs are
    parsed from filenames (see :func:`_parse_market_filename`).

    Args:
        root: Path to the Market-1501 directory.

    Returns:
        ``(query, gallery)`` :class:`ReIDSplit` pair.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Market-1501 root not found: {root}")

    def _load_dir(subdir: Path, *, gallery_junk_pids: frozenset[int]) -> ReIDSplit:
        if not subdir.exists():
            raise FileNotFoundError(f"Market-1501 sub-directory not found: {subdir}")
        paths, pids, camids = [], [], []
        for img_path in sorted(subdir.glob("*.jpg")):
            pid, camid = _parse_market_filename(img_path.name)
            paths.append(str(img_path))
            pids.append(pid)
            camids.append(camid)
        return ReIDSplit(
            image_paths=paths,
            pids=np.array(pids, dtype=np.int32),
            camids=np.array(camids, dtype=np.int32),
            gallery_junk_pids=gallery_junk_pids,
        )

    query = _load_dir(root / "query", gallery_junk_pids=frozenset({-1}))
    gallery = _load_dir(root / "bounding_box_test", gallery_junk_pids=MARKET1501_GALLERY_JUNK_PIDS)
    return query, gallery
