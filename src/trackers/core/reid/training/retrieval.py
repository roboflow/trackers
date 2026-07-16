# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Build query/gallery splits from per-identity crop datasets.

Query and gallery crops use synthetic ``camid`` 0/1 so single-camera MOT crops
still produce valid retrieval pairs.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from trackers.core.reid.eval.datasets import ReIDSplit

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

_QUERY_CAMID = 0
_GALLERY_CAMID = 1


def _identity_images(identity_dir: Path) -> list[Path]:
    return sorted(p for p in identity_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)


def _eligible_identities(
    crop_root: Path,
    *,
    min_crops: int,
    identity_filter: set[str] | None = None,
) -> list[str]:
    all_identities = sorted(p.name for p in crop_root.iterdir() if p.is_dir())
    if identity_filter is not None:
        all_identities = [name for name in all_identities if name in identity_filter]
    return [name for name in all_identities if len(_identity_images(crop_root / name)) >= min_crops]


def _build_query_gallery_splits(
    crop_root: Path,
    identities: list[str],
    *,
    queries_per_id: int,
) -> tuple[ReIDSplit, ReIDSplit]:
    query_paths: list[str] = []
    query_pids: list[int] = []
    query_camids: list[int] = []
    gallery_paths: list[str] = []
    gallery_pids: list[int] = []
    gallery_camids: list[int] = []

    for pid, identity in enumerate(identities):
        images = _identity_images(crop_root / identity)
        for image_path in images[:queries_per_id]:
            query_paths.append(str(image_path))
            query_pids.append(pid)
            query_camids.append(_QUERY_CAMID)
        for image_path in images[queries_per_id:]:
            gallery_paths.append(str(image_path))
            gallery_pids.append(pid)
            gallery_camids.append(_GALLERY_CAMID)

    query = ReIDSplit(
        image_paths=query_paths,
        pids=np.array(query_pids, dtype=np.int32),
        camids=np.array(query_camids, dtype=np.int32),
    )
    gallery = ReIDSplit(
        image_paths=gallery_paths,
        pids=np.array(gallery_pids, dtype=np.int32),
        camids=np.array(gallery_camids, dtype=np.int32),
    )
    return query, gallery


def build_retrieval_split(
    crop_root: str | Path,
    *,
    queries_per_id: int = 1,
    min_crops: int = 2,
    identities: list[str] | None = None,
) -> tuple[list[str], ReIDSplit, ReIDSplit]:
    """Build query/gallery splits from eligible identities under ``crop_root``."""
    crop_root = Path(crop_root)
    if not crop_root.is_dir():
        raise FileNotFoundError(f"Crop dataset root not found: {crop_root}")
    if queries_per_id < 1:
        raise ValueError(f"queries_per_id must be >= 1, got {queries_per_id}.")
    min_crops = max(min_crops, queries_per_id + 1)

    identity_filter = set(identities) if identities is not None else None
    eligible = _eligible_identities(crop_root, min_crops=min_crops, identity_filter=identity_filter)
    if not eligible:
        raise ValueError(f"No identity under {crop_root} has >= {min_crops} crops; cannot build a retrieval split.")

    query, gallery = _build_query_gallery_splits(crop_root, eligible, queries_per_id=queries_per_id)
    return eligible, query, gallery


def build_identity_holdout_split(
    crop_root: str | Path,
    *,
    holdout_fraction: float = 0.2,
    queries_per_id: int = 1,
    min_crops: int = 2,
    seed: int = 0,
) -> tuple[list[str], list[str], ReIDSplit, ReIDSplit]:
    """Hold out a fraction of identities for retrieval; return train + eval splits."""
    crop_root = Path(crop_root)
    if not crop_root.is_dir():
        raise FileNotFoundError(f"Crop dataset root not found: {crop_root}")
    if queries_per_id < 1:
        raise ValueError(f"queries_per_id must be >= 1, got {queries_per_id}.")
    min_crops = max(min_crops, queries_per_id + 1)

    all_identities = sorted(p.name for p in crop_root.iterdir() if p.is_dir())
    eligible = _eligible_identities(crop_root, min_crops=min_crops)
    if not eligible:
        raise ValueError(f"No identity under {crop_root} has >= {min_crops} crops; cannot build a retrieval holdout.")

    rng = random.Random(seed)  # noqa: S311
    shuffled = eligible.copy()
    rng.shuffle(shuffled)
    num_holdout = max(1, round(len(shuffled) * holdout_fraction))
    holdout_identities = sorted(shuffled[:num_holdout])
    holdout_set = set(holdout_identities)
    train_identities = sorted(name for name in all_identities if name not in holdout_set)

    query, gallery = _build_query_gallery_splits(crop_root, holdout_identities, queries_per_id=queries_per_id)
    return train_identities, holdout_identities, query, gallery
