# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from trackers.core.reid.training.datasets import (
    PKSampler,
    ReIDCropDataset,
    build_train_transform,
)


def _make_crops(root: Path, num_ids: int = 4, num_per: int = 6) -> None:
    rng = np.random.default_rng(0)
    for i in range(num_ids):
        identity_dir = root / f"id_{i:02d}"
        identity_dir.mkdir(parents=True)
        for j in range(num_per):
            array = (rng.random((48, 24, 3)) * 255).astype("uint8")
            Image.fromarray(array).save(identity_dir / f"{j}.jpg")


def test_dataset_scans_identities_and_labels(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=4, num_per=5)
    dataset = ReIDCropDataset(tmp_path, transform=build_train_transform())

    assert dataset.num_classes == 4
    assert len(dataset) == 20
    image, label = dataset[0]
    # (3, H, W) with the pinned re-ID input geometry.
    assert tuple(image.shape) == (3, 256, 128)
    assert 0 <= label < 4


def test_pk_sampler_batch_structure(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=5, num_per=4)
    dataset = ReIDCropDataset(tmp_path)
    p, k = 2, 3
    sampler = PKSampler(dataset.labels, p=p, k=k, seed=123)

    # num_identities // p batches.
    batches = list(sampler)
    assert len(sampler) == 5 // p
    assert len(batches) == len(sampler)

    for batch in batches:
        assert len(batch) == p * k
        label_counts = Counter(dataset.labels[i] for i in batch)
        # Exactly p identities, each appearing k times.
        assert len(label_counts) == p
        assert all(count == k for count in label_counts.values())


def test_pk_sampler_handles_identity_with_few_instances(tmp_path: Path) -> None:
    # One identity has fewer than k instances -> sampled with replacement.
    (tmp_path / "id_a").mkdir(parents=True)
    (tmp_path / "id_b").mkdir(parents=True)
    rng = np.random.default_rng(1)
    for j in range(5):
        Image.fromarray((rng.random((48, 24, 3)) * 255).astype("uint8")).save(tmp_path / "id_a" / f"{j}.jpg")
    Image.fromarray((rng.random((48, 24, 3)) * 255).astype("uint8")).save(tmp_path / "id_b" / "0.jpg")
    dataset = ReIDCropDataset(tmp_path)
    sampler = PKSampler(dataset.labels, p=2, k=4, seed=0)
    batch = next(iter(sampler))
    assert len(batch) == 8


def test_pk_sampler_rejects_too_few_identities(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=2, num_per=4)
    dataset = ReIDCropDataset(tmp_path)
    with pytest.raises(ValueError, match="at least p="):
        PKSampler(dataset.labels, p=4, k=2)
