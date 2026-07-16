# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Fine-tune a re-ID encoder on a per-identity crop dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from trackers.core.reid.models.preprocessing import ReIDPreprocessing

if TYPE_CHECKING:
    import torch
    from torch import nn

    from trackers.core.reid.model import ReIDModel

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Hyper-parameters for :func:`train_reid`."""

    epochs: int = 30
    p: int = 8
    k: int = 4
    lr: float = 3e-4
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    lr_milestones: tuple[int, ...] = (20, 27)
    lr_gamma: float = 0.1
    margin: float = 0.3
    label_smoothing: float = 0.1
    triplet_weight: float = 1.0
    use_center: bool = False
    center_weight: float = 0.0005
    center_lr: float = 0.5
    freeze_backbone_epochs: int = 0
    num_workers: int = 4
    amp: bool = True
    device: str = "auto"
    seed: int = 42
    log_interval: int = 20


@dataclass
class TrainResult:
    """Output of :func:`train_reid` (model, class count, history, save path)."""

    model: ReIDModel
    num_classes: int
    history: list[dict[str, float]] = field(default_factory=list)
    output_dir: str | None = None


def _forward_embeddings_logits(
    backbone: nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(embeddings, logits)`` from an OSNet-style backbone."""
    required = ("_featuremaps", "global_avgpool", "fc", "classifier")
    if not all(hasattr(backbone, name) for name in required):
        raise TypeError(
            f"train_reid expects an OSNet-style backbone exposing {required}. Got {type(backbone).__name__}."
        )
    features = backbone._featuremaps(images)
    pooled = backbone.global_avgpool(features).flatten(1)
    embeddings = backbone.fc(pooled)
    logits = backbone.classifier(embeddings)
    return embeddings, logits


def _set_backbone_frozen(backbone: nn.Module, frozen: bool) -> None:
    """Freeze/unfreeze all backbone params except the ``fc``/``classifier`` head."""
    head = {"fc", "classifier"}
    for name, param in backbone.named_parameters():
        top = name.split(".")[0]
        param.requires_grad = (top in head) or (not frozen)


def _build_lr_lambda(config: TrainConfig):
    def lr_lambda(epoch: int) -> float:
        if config.warmup_epochs > 0 and epoch < config.warmup_epochs:
            return float(epoch + 1) / float(config.warmup_epochs)
        factor = 1.0
        for milestone in config.lr_milestones:
            if epoch >= milestone:
                factor *= config.lr_gamma
        return factor

    return lr_lambda


def _resolve_training_recipe(
    architecture: str | None,
    pretrained: str | None,
    preprocessing: ReIDPreprocessing | None,
) -> tuple[str, str | None, ReIDPreprocessing]:
    """Resolve architecture name, warm-start weights, and preprocessing."""
    from trackers.core.reid.models.registry import (
        DEFAULT_MODEL,
        default_preprocessing_for_architecture,
        resolve_model_card,
    )

    if architecture is not None:
        arch_name = architecture
        weights_source = pretrained
        resolved_preprocessing = (
            preprocessing if preprocessing is not None else default_preprocessing_for_architecture(architecture)
        )
        return arch_name, weights_source, resolved_preprocessing

    card = resolve_model_card(pretrained or DEFAULT_MODEL)
    if card is None:
        raise ValueError(
            f"Could not resolve a ModelCard for {pretrained or DEFAULT_MODEL!r}. "
            "Pass architecture= for bare weight files, or use a curated alias."
        )
    return (
        card.architecture,
        card.weights,
        preprocessing or card.preprocessing,
    )


def _build_trainable_backbone(
    arch_name: str,
    weights_source: str | None,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """Build a backbone with a classifier head and optional warm-start weights."""
    from trackers.core.reid.architectures import build_architecture
    from trackers.core.reid.models.loaders import load_state_dict_into, resolve_weights

    backbone = build_architecture(arch_name, num_classes=num_classes)
    if weights_source is not None:
        local_path = resolve_weights(weights_source)
        report = load_state_dict_into(backbone, local_path, device)
        logger.info("Warm-start from %s: %s", weights_source, report.summary())
    return backbone


def train_reid(
    data_root: str | Path,
    config: TrainConfig | None = None,
    *,
    architecture: str | None = None,
    pretrained: str | None = None,
    preprocessing: ReIDPreprocessing | None = None,
    output_dir: str | Path | None = None,
    include_identities: list[str] | None = None,
) -> TrainResult:
    """Fine-tune a re-ID encoder on crops under ``data_root/<identity>/``.

    Args:
        data_root: Crop dataset root (e.g. from :func:`generate_mot_patches`).
        config: Training hyper-parameters.
        architecture: Backbone name override.
        pretrained: Weights source / alias to warm-start from.
        preprocessing: Preprocessing override.
        output_dir: If set, save via :meth:`ReIDModel.save_pretrained`.
        include_identities: Optional identity folder whitelist.

    Returns:
        :class:`TrainResult`.
    """
    import torch
    from torch.utils.data import DataLoader

    from trackers.core.reid.model import ReIDModel, _select_device
    from trackers.core.reid.training.datasets import (
        PKSampler,
        ReIDCropDataset,
        build_train_transform,
    )
    from trackers.core.reid.training.losses import ReIDLoss

    config = config or TrainConfig()
    device = _select_device(config.device)
    torch.manual_seed(config.seed)

    arch_name, weights_source, resolved_preprocessing = _resolve_training_recipe(
        architecture, pretrained, preprocessing
    )
    transform = build_train_transform(
        input_size=resolved_preprocessing.input_size,
        mean=resolved_preprocessing.mean,
        std=resolved_preprocessing.std,
    )
    dataset = ReIDCropDataset(data_root, transform=transform, include_identities=include_identities)
    num_classes = dataset.num_classes
    logger.info("Crop dataset: %d identities, %d crops", num_classes, len(dataset))

    sampler = PKSampler(dataset.labels, p=config.p, k=config.k, seed=config.seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    backbone = _build_trainable_backbone(arch_name, weights_source, num_classes, device)
    backbone.to(device).train()
    feat_dim = int(getattr(backbone, "feature_dim", 512))
    loss_fn = ReIDLoss(
        num_classes,
        feat_dim,
        margin=config.margin,
        label_smoothing=config.label_smoothing,
        triplet_weight=config.triplet_weight,
        use_center=config.use_center,
        center_weight=config.center_weight,
    ).to(device)

    param_groups: list[dict] = [{"params": backbone.parameters()}]
    if config.use_center and loss_fn.center is not None:
        param_groups.append(
            {
                "params": loss_fn.center.parameters(),
                "lr": config.center_lr,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.Adam(param_groups, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _build_lr_lambda(config))

    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        _set_backbone_frozen(backbone, frozen=epoch < config.freeze_backbone_epochs)

        epoch_totals: dict[str, float] = {}
        num_batches = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                embeddings, logits = _forward_embeddings_logits(backbone, images)
                loss, components = loss_fn(embeddings, logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for key, value in components.items():
                epoch_totals[key] = epoch_totals.get(key, 0.0) + value
            num_batches += 1
            step += 1
            if step % config.log_interval == 0:
                logger.info(
                    "epoch %d step %d: %s",
                    epoch,
                    step,
                    {k: round(v, 4) for k, v in components.items()},
                )

        scheduler.step()
        epoch_mean = {k: v / max(1, num_batches) for k, v in epoch_totals.items()}
        epoch_mean["epoch"] = epoch
        epoch_mean["lr"] = optimizer.param_groups[0]["lr"]
        history.append(epoch_mean)
        logger.info("epoch %d mean: %s", epoch, {k: round(v, 4) for k, v in epoch_mean.items()})

    backbone.eval()
    model = ReIDModel(backbone, device, resolved_preprocessing)
    model._architecture = arch_name if isinstance(arch_name, str) else None

    saved_dir = None
    if output_dir is not None:
        saved_dir = str(output_dir)
        model.save_pretrained(saved_dir)

    return TrainResult(
        model=model,
        num_classes=num_classes,
        history=history,
        output_dir=saved_dir,
    )
