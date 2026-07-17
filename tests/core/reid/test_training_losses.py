# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for ReID training losses."""

from __future__ import annotations

import torch

from trackers.core.reid.training.losses import (
    BatchHardTripletLoss,
    CrossEntropyLabelSmooth,
    ReIDLoss,
)


def test_label_smooth_ce_is_lower_when_logits_match_labels() -> None:
    loss_fn = CrossEntropyLabelSmooth(num_classes=4, epsilon=0.1)
    labels = torch.tensor([0, 1, 2, 3])
    confident = torch.full((4, 4), -5.0)
    confident[torch.arange(4), labels] = 5.0

    good = loss_fn(confident, labels)
    bad = loss_fn(-confident, labels)
    assert good.ndim == 0
    assert good < bad


def test_batch_hard_triplet_zero_for_well_separated_clusters() -> None:
    loss_fn = BatchHardTripletLoss(margin=0.3, normalize=False)
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0], [51.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    assert loss_fn(embeddings, labels).item() == 0.0


def test_batch_hard_triplet_positive_when_negatives_are_closer() -> None:
    loss_fn = BatchHardTripletLoss(margin=0.3, normalize=False)
    # id 0 spread far apart; id 1 sits between them -> violates margin.
    embeddings = torch.tensor([[0.0, 0.0], [10.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    assert loss_fn(embeddings, labels).item() > 0.0


def test_reid_loss_default_excludes_center() -> None:
    loss_fn = ReIDLoss(num_classes=4, feat_dim=8)
    embeddings = torch.randn(8, 8, requires_grad=True)
    logits = torch.randn(8, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    total, components = loss_fn(embeddings, logits, labels)

    assert set(components) == {"ce", "triplet", "total"}
    assert loss_fn.center is None
    total.backward()
    assert embeddings.grad is not None


def test_reid_loss_optional_center_term() -> None:
    loss_fn = ReIDLoss(num_classes=4, feat_dim=8, use_center=True, center_weight=0.001)
    embeddings = torch.randn(8, 8)
    logits = torch.randn(8, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    total, components = loss_fn(embeddings, logits, labels)

    assert "center" in components
    assert loss_fn.center is not None
    assert total.ndim == 0
