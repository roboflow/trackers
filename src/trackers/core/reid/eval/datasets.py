# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Dataset loaders for re-ID evaluation.

Supports the two standard benchmarks used in this RFC:

- **MSMT17** — large-scale pedestrian re-ID; 15 cameras, 4,101 identities.
  Requires accepting the original license:
  http://www.pkuvmc.com/publications/msmt17.html
- **Market-1501** — smaller pedestrian benchmark; 6 cameras, 1,501 identities.
  Freely available; useful as a fast sanity-check.

Both loaders return a ``(query, gallery)`` pair of :class:`ReidSplit` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ReidSplit:
    """A single query or gallery split of a re-ID dataset.

    Attributes:
        image_paths: Absolute paths to each image, one per sample.
        pids: Integer person (identity) IDs, shape ``(N,)``.
        camids: Integer camera IDs, shape ``(N,)``.
    """

    image_paths: list[str]
    pids: np.ndarray
    camids: np.ndarray

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


def load_msmt17(root: str | Path) -> tuple[ReidSplit, ReidSplit]:
    """Load the MSMT17 query and gallery splits from a local directory.

    MSMT17 must be downloaded separately by accepting the dataset license at
    http://www.pkuvmc.com/publications/msmt17.html

    Expected directory layout::

        <root>/
        ├── test/
        │   └── <pid>/      (one folder per identity)
        │       └── *.jpg
        ├── list_query.txt
        └── list_gallery.txt

    Each list file contains one sample per line.  Two formats are supported:

    * **2-column** (community mirrors): ``<relative_path>  <pid>``
      Camera ID is extracted from the filename (3rd ``_``-separated field).
    * **3-column** (original release): ``<relative_path>  <pid>  <camid>``

    Args:
        root: Path to the ``MSMT17_V1`` (or ``MSMT17``) directory.

    Returns:
        ``(query, gallery)`` tuple of :class:`ReidSplit` objects.

    Raises:
        FileNotFoundError: If *root* does not exist or list files are missing.

    Examples:
        >>> import os
        >>> load_msmt17("/nonexistent")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        FileNotFoundError: MSMT17 root not found: /nonexistent
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"MSMT17 root not found: {root}")

    def _parse_list(list_file: Path, image_root: Path) -> ReidSplit:
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
            camid = int(parts[2]) if len(parts) >= 3 else _parse_msmt17_camid(
                Path(rel_path).name
            )
            paths.append(str(image_root / rel_path))
            pids.append(pid)
            camids.append(camid)
        return ReidSplit(
            image_paths=paths,
            pids=np.array(pids, dtype=np.int32),
            camids=np.array(camids, dtype=np.int32),
        )

    test_root = root / "test"
    query = _parse_list(root / "list_query.txt", test_root)
    gallery = _parse_list(root / "list_gallery.txt", test_root)
    return query, gallery


# --------------------------------------------------------------------------- #
# Market-1501
# --------------------------------------------------------------------------- #

def _parse_market_filename(filename: str) -> tuple[int, int]:
    """Extract (pid, camid) from a Market-1501 image filename.

    Filename format: ``<pid>_c<camid>s<seq>_<frame>_<det>.jpg``
    Example: ``0001_c1s1_000001_00.jpg`` → pid=1, camid=0

    Person-ID conventions (standard Market-1501 / torchreid protocol):

    - ``-1`` → **junk** images (false detections). Kept as ``pid = -1`` so the
      metric's junk rule (:func:`~trackers.core.reid.eval.metrics.compute_reid_metrics`)
      discards them entirely — they must not appear in the ranked gallery.
    - ``0000`` → **distractor** images. Kept as ``pid = 0``; real query IDs start
      at 1, so distractors never count as a correct match but remain in the
      gallery as noise that the model has to rank below true matches.

    Args:
        filename: Basename of the image file (with or without extension).

    Returns:
        ``(pid, camid)`` tuple where camid is **0-indexed**.

    Examples:
        >>> _parse_market_filename("0001_c1s1_000001_00.jpg")
        (1, 0)
        >>> _parse_market_filename("0000_c2s1_000001_00.jpg")
        (0, 1)
        >>> _parse_market_filename("-1_c3s1_000001_00.jpg")
        (-1, 2)
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    pid = int(parts[0])
    camid = int(parts[1][1]) - 1  # "c1" → 0

    return pid, camid


def load_market1501(root: str | Path) -> tuple[ReidSplit, ReidSplit]:
    """Load the Market-1501 query and gallery splits from a local directory.

    Market-1501 can be downloaded from the project page:
    http://www.liangzheng.org/Project/project_reid.html

    Expected directory layout::

        <root>/
        ├── query/
        │   └── *.jpg
        └── bounding_box_test/   (gallery)
            └── *.jpg

    Person IDs and camera IDs are parsed from filenames following the
    ``<pid>_c<camid>s<seq>_<frame>_<det>.jpg`` convention. Junk images
    (labelled ``-1_…``) keep ``pid = -1`` so the junk rule in
    :func:`~trackers.core.reid.eval.metrics.compute_reid_metrics` discards
    them, while distractor images (labelled ``0000_…``) keep ``pid = 0`` and
    remain in the gallery as noise. This reproduces the standard Market-1501
    protocol (effective gallery of 15 913 images out of 19 732 files).

    Args:
        root: Path to the ``Market-1501-v15.09.15`` directory.

    Returns:
        ``(query, gallery)`` tuple of :class:`ReidSplit` objects.

    Raises:
        FileNotFoundError: If *root* or expected sub-directories are missing.

    Examples:
        >>> import os
        >>> load_market1501("/nonexistent")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        FileNotFoundError: Market-1501 root not found: /nonexistent
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Market-1501 root not found: {root}")

    def _load_dir(subdir: Path) -> ReidSplit:
        if not subdir.exists():
            raise FileNotFoundError(f"Market-1501 sub-directory not found: {subdir}")
        paths, pids, camids = [], [], []
        for img_path in sorted(subdir.glob("*.jpg")):
            pid, camid = _parse_market_filename(img_path.name)
            paths.append(str(img_path))
            pids.append(pid)
            camids.append(camid)
        return ReidSplit(
            image_paths=paths,
            pids=np.array(pids, dtype=np.int32),
            camids=np.array(camids, dtype=np.int32),
        )

    query = _load_dir(root / "query")
    gallery = _load_dir(root / "bounding_box_test")
    return query, gallery
