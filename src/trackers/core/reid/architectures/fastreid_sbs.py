# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
#
# Adapted from JDAI-CV/fast-reid (Apache-2.0)
# Copyright 2019 JD.com Inc. JD AI
# Source: https://github.com/JDAI-CV/fast-reid
#   - GeM: fastreid/layers/pooling.py (GeneralizedMeanPooling / GeneralizedMeanPoolingP)
#   - SBS head layout: fastreid/modeling/heads/embedding_head.py
#   - last_stride=1 ResNeSt semantics: fastreid/modeling/backbones/resnest.py
# Also used via BoT-SORT's FastReIDInterface (NirAharon/BoT-SORT, MIT).
# ------------------------------------------------------------------------

"""FastReID Strong Baseline (SBS) inference stack for BoT-SORT checkpoints.

Uses ``timm`` ``resnest50d`` with a small FastReID ``last_stride=1`` patch on
``layer4[0]``, then Generalized Mean Pooling and a BatchNorm neck — the same
inference path as FastReID ``EmbeddingHead`` in eval.
"""

from __future__ import annotations

import torch
import timm
from torch import nn
from torch.nn import functional as F

__all__ = [
    "FASTREID_SBS_ARCHITECTURE",
    "FastReIDSBSResNeSt50",
    "build_fastreid_sbs_resnest50",
    "remap_fastreid_sbs_state_dict",
]

FASTREID_SBS_ARCHITECTURE = "fastreid_sbs_resnest50"
FASTREID_SBS_EMBED_DIM = 2048


def _patch_resnest50d_for_fastreid_last_stride(backbone: nn.Module) -> nn.Module:
    """Align ``timm`` ``resnest50d`` (``output_stride=16``) with FastReID ResNeSt.

    FastReID ``LAST_STRIDE=1`` keeps layer4 spatial stride at 1 via:
    - downsample avg-pool kernel 1 (identity spatial size), and
    - average-downsampling (AVD) after the Split-Attention conv with stride 1.

    ``timm``'s dilated / stride-16 path instead uses ``AvgPool2dSame(2, stride=1)``
    and leaves ``avd_last`` unset on ``layer4[0]``, which shifts embeddings even
    when checkpoint keys load 100%.
    """
    block = backbone.layer4[0]
    block.downsample[0] = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
    block.avd_last = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    return backbone


class GeneralizedMeanPooling(nn.Module):
    """GeM pooling used by FastReID SBS (learnable exponent *p*).

    ``p`` defaults to 3.0 only until checkpoint load; MOT17 SBS-S50 stores the
    trained value under ``heads.pool_layer.p`` (≈1.72). See
    :func:`remap_fastreid_sbs_state_dict`.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p)


class FastReIDSBSResNeSt50(nn.Module):
    """ResNeSt50 SBS re-ID encoder matching BoT-SORT / FastReID MOT17 checkpoints.

    Inference path: patched ``timm`` ResNeSt-50 → GeM (:attr:`pool`) → BNNeck
    (:attr:`bottleneck`). Weights for all three stages load from a BoT-SORT
    ``.pth`` via :func:`remap_fastreid_sbs_state_dict`.
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = timm.create_model(
            "resnest50d",
            pretrained=False,
            num_classes=0,
            global_pool="",
            output_stride=16,
        )
        self.backbone = _patch_resnest50d_for_fastreid_last_stride(backbone)
        self.pool = GeneralizedMeanPooling()
        self.bottleneck = nn.BatchNorm1d(FASTREID_SBS_EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.bottleneck(x)


def build_fastreid_sbs_resnest50(*, num_classes: int = 0, pretrained: bool = False) -> FastReIDSBSResNeSt50:
    """Build the BoT-SORT FastReID SBS ResNeSt50 encoder (weights loaded separately)."""
    del num_classes, pretrained
    return FastReIDSBSResNeSt50()


def remap_fastreid_sbs_state_dict(state_dict: dict) -> dict:
    """Map BoT-SORT / FastReID SBS checkpoint keys onto :class:`FastReIDSBSResNeSt50`.

    Renames FastReID ``heads.*`` keys to ``pool.*`` / ``bottleneck.*`` and passes
    ``backbone.*`` through unchanged. Skips ``heads.weight`` (classifier). GeM
    ``heads.pool_layer.p`` becomes ``pool.p`` and replaces the 3.0 init default.
    """
    mapped: dict = {}
    for key, value in state_dict.items():
        key = key[7:] if key.startswith("module.") else key
        if key.startswith("backbone."):
            mapped[key] = value
        elif key == "heads.pool_layer.p":
            mapped["pool.p"] = value
        elif key.startswith("heads.bottleneck.0."):
            mapped["bottleneck." + key[len("heads.bottleneck.0.") :]] = value
        # Skip heads.weight (identity classifier; unused at inference).
    return mapped
