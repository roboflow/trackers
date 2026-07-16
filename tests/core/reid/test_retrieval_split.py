# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from trackers.core.reid.training import build_identity_holdout_split, build_retrieval_split


def _make_crops(root: Path, num_ids: int, num_per: int) -> None:
    rng = np.random.default_rng(0)
    for i in range(num_ids):
        identity_dir = root / f"id_{i:02d}"
        identity_dir.mkdir(parents=True)
        for j in range(num_per):
            array = (rng.random((48, 24, 3)) * 255).astype("uint8")
            Image.fromarray(array).save(identity_dir / f"{j:02d}.jpg")


def test_holdout_is_identity_disjoint_from_train(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=10, num_per=4)
    train_ids, holdout_ids, _query, _gallery = build_identity_holdout_split(tmp_path, holdout_fraction=0.2, seed=0)

    assert set(train_ids).isdisjoint(set(holdout_ids))
    assert set(train_ids) | set(holdout_ids) == {f"id_{i:02d}" for i in range(10)}
    # 20% of 10 identities held out.
    assert len(holdout_ids) == 2


def test_query_and_gallery_use_distinct_camids(tmp_path: Path) -> None:
    # Single-camera MOT crops: query camid must differ from gallery camid so the
    # cross-camera junk rule keeps same-identity matches.
    _make_crops(tmp_path, num_ids=6, num_per=5)
    _, holdout_ids, query, gallery = build_identity_holdout_split(
        tmp_path, holdout_fraction=0.5, queries_per_id=1, seed=1
    )

    assert set(query.camids.tolist()) == {0}
    assert set(gallery.camids.tolist()) == {1}
    # Every held-out identity contributes exactly one query.
    assert len(query) == len(holdout_ids)
    # Query pids are all present in the gallery (matches exist).
    assert set(query.pids.tolist()).issubset(set(gallery.pids.tolist()))


def test_min_crops_filters_ineligible_identities(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=4, num_per=4)
    # An identity with a single crop cannot be a holdout (needs query+gallery).
    (tmp_path / "id_single").mkdir()
    Image.fromarray((np.random.default_rng(2).random((48, 24, 3)) * 255).astype("uint8")).save(
        tmp_path / "id_single" / "00.jpg"
    )

    train_ids, holdout_ids, _, _ = build_identity_holdout_split(tmp_path, holdout_fraction=1.0, min_crops=2, seed=0)
    # The single-crop identity is never held out (ineligible) -> stays trainable.
    assert "id_single" not in holdout_ids
    assert "id_single" in train_ids


def test_retrieval_split_uses_all_eligible_identities(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=8, num_per=4)
    identities, query, gallery = build_retrieval_split(tmp_path, queries_per_id=1)

    assert identities == [f"id_{i:02d}" for i in range(8)]
    assert len(query) == 8
    assert len(gallery) == 8 * 3
    assert set(query.camids.tolist()) == {0}
    assert set(gallery.camids.tolist()) == {1}
