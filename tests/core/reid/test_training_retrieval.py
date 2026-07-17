# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for MOT crop retrieval split builders."""

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
    train_ids, holdout_ids, query, gallery = build_identity_holdout_split(tmp_path, holdout_fraction=0.2, seed=0)

    assert set(train_ids).isdisjoint(set(holdout_ids))
    assert set(train_ids) | set(holdout_ids) == {f"id_{i:02d}" for i in range(10)}
    assert len(holdout_ids) == 2
    # Synthetic camids keep same-identity query/gallery pairs valid under the
    # cross-camera junk rule used by gallery eval.
    assert set(query.camids.tolist()) == {0}
    assert set(gallery.camids.tolist()) == {1}
    assert len(query) == len(holdout_ids)
    assert set(query.pids.tolist()).issubset(set(gallery.pids.tolist()))


def test_min_crops_filters_ineligible_identities(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=4, num_per=4)
    (tmp_path / "id_single").mkdir()
    Image.fromarray((np.random.default_rng(2).random((48, 24, 3)) * 255).astype("uint8")).save(
        tmp_path / "id_single" / "00.jpg"
    )

    train_ids, holdout_ids, _, _ = build_identity_holdout_split(tmp_path, holdout_fraction=1.0, min_crops=2, seed=0)
    assert "id_single" not in holdout_ids
    assert "id_single" in train_ids


def test_retrieval_split_uses_all_eligible_identities(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=8, num_per=4)
    identities, query, gallery = build_retrieval_split(tmp_path, queries_per_id=1)

    assert identities == [f"id_{i:02d}" for i in range(8)]
    assert len(query) == 8
    assert len(gallery) == 8 * 3
