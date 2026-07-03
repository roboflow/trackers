# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""FastReID Strong Baseline (SBS) inference stack for BoT-SORT checkpoints.

The BoT-SORT MOT17/MOT20 SBS-S50 weights use a ResNeSt50 backbone (timm-compatible
at ``output_stride=16``), Generalized Mean Pooling, and a BatchNorm neck — the
same inference path as FastReID ``EmbeddingHead`` in eval mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FastReIDSBSResNeSt50", "GeneralizedMeanPooling", "build_fastreid_sbs_resnest50"]

FASTREID_SBS_EMBED_DIM = 2048


class GeneralizedMeanPooling(nn.Module):
    """GeM pooling used by FastReID SBS (learnable exponent *p*)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p)


class FastReIDSBSResNeSt50(nn.Module):
    """ResNeSt50 SBS re-ID encoder matching BoT-SORT / FastReID MOT17 checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "resnest50d",
            num_classes=0,
            global_pool="",
            output_stride=16,
        )
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
