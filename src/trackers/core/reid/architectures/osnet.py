# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
#
# Adapted from KaiyangZhou/deep-person-reid
# Copyright (c) 2018-2021 Kaiyang Zhou
# Licensed under the MIT License
# Source: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py
# Paper: Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
# ------------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["OSNet", "build_osnet"]


# --------------------------------------------------------------------------- #
# Basic layers
# --------------------------------------------------------------------------- #


class _ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1(_ConvBnRelu):
    """1x1 conv + BN + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__(in_channels, out_channels, 1, stride=stride, padding=0, groups=groups)


class _Conv1x1Linear(nn.Module):
    """1x1 conv + BN (no non-linearity)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class _LightConv3x3(nn.Module):
    """Lightweight 3x3 conv: 1x1 linear + depthwise 3x3 + BN + ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


# --------------------------------------------------------------------------- #
# Omni-scale building blocks
# --------------------------------------------------------------------------- #


class _ChannelGate(nn.Module):
    """Channel-wise gating via squeeze-and-excitation."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.sigmoid(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * g


class OSBlock(nn.Module):
    """Omni-scale feature learning residual block.

    Aggregates features from four parallel streams with different effective
    receptive-field depths (1, 2, 3, 4 stacked lightweight 3x3 convs) gated
    by a shared channel-wise gate before the residual addition.
    """

    def __init__(self, in_channels: int, out_channels: int, bottleneck_reduction: int = 4) -> None:
        super().__init__()
        mid = out_channels // bottleneck_reduction
        self.conv1 = _Conv1x1(in_channels, mid)
        self.conv2a = _LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(_LightConv3x3(mid, mid), _LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(_LightConv3x3(mid, mid), _LightConv3x3(mid, mid), _LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(
            _LightConv3x3(mid, mid),
            _LightConv3x3(mid, mid),
            _LightConv3x3(mid, mid),
            _LightConv3x3(mid, mid),
        )
        self.gate = _ChannelGate(mid)
        self.conv3 = _Conv1x1Linear(mid, out_channels)
        self.downsample = _Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)
        x2 = (
            self.gate(self.conv2a(x1))
            + self.gate(self.conv2b(x1))
            + self.gate(self.conv2c(x1))
            + self.gate(self.conv2d(x1))
        )
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


# --------------------------------------------------------------------------- #
# Full OSNet architecture
# --------------------------------------------------------------------------- #


class OSNet(nn.Module):
    """Omni-Scale Network for instance re-identification.

    Architecture from:
        Zhou et al. *Omni-Scale Feature Learning for Person Re-Identification*. ICCV, 2019.
        Zhou et al. *Learning Generalisable Omni-Scale Representations for Person Re-Identification*. TPAMI, 2021.

    At inference (``model.eval()``), and whenever no classifier was built
    (``num_classes <= 0``), ``forward()`` returns the ``feature_dim`` embedding.
    Classifier logits are returned only in ``train()`` mode with a head.

    Args:
        num_classes: Number of identity classes for the classification head.
            Use ``0`` (or less) to omit the head for inference-only models.
        blocks: List of block classes for each stage.
        layers: Number of blocks per stage.
        channels: Channel widths ``[stem, stage1, stage2, stage3]``.
        feature_dim: Dimensionality of the final embedding vector.
    """

    def __init__(
        self,
        num_classes: int,
        blocks: list[type[nn.Module]],
        layers: list[int],
        channels: list[int],
        feature_dim: int = 512,
    ) -> None:
        super().__init__()
        if len(blocks) != len(layers) or len(layers) != len(channels) - 1:
            raise ValueError("blocks, layers, and channels must have consistent lengths")

        self.feature_dim = feature_dim
        self.conv1 = _ConvBnRelu(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1], downsample=True)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2], downsample=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3], downsample=False)
        self.conv5 = _Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        # Inference builds use num_classes=0 and omit the head so checkpoints
        # that drop classifier.* can load with required_match_fraction=1.0.
        if num_classes > 0:
            self.classifier = nn.Linear(feature_dim, num_classes)
        self._init_params()

    @staticmethod
    def _make_layer(
        block: type[nn.Module],
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        downsample: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [block(in_channels, out_channels)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        if downsample:
            layers.append(nn.Sequential(_Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _init_params(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _featuremaps(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Returns the embedding at inference time (``model.eval()``) and when no
        classifier was built. Returns classifier logits only in training mode
        with ``num_classes > 0``.
        """
        x = self._featuremaps(x)
        v = self.global_avgpool(x).view(x.size(0), -1)
        v = self.fc(v)
        if self.training and hasattr(self, "classifier"):
            return self.classifier(v)
        return v


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

_CONFIGS: dict[str, dict] = {
    "x1_0": {"channels": [64, 256, 384, 512]},
    "x0_75": {"channels": [48, 192, 288, 384]},
    "x0_5": {"channels": [32, 128, 192, 256]},
    "x0_25": {"channels": [16, 64, 96, 128]},
}


def build_osnet(variant: str = "x1_0", num_classes: int = 0) -> OSNet:
    """Instantiate an OSNet architecture without loading weights.

    Args:
        variant: Width multiplier variant. One of ``"x1_0"`` (default),
            ``"x0_75"``, ``"x0_5"``, ``"x0_25"``.
        num_classes: Identity classes for the classification head. Use ``0``
            (default) for inference-only models with no classifier.

    Returns:
        An initialised :class:`OSNet` in training mode.

    Raises:
        ValueError: If *variant* is not one of the supported values.

    Examples:
        >>> model = build_osnet("x1_0")
        >>> model.feature_dim
        512
    """
    if variant not in _CONFIGS:
        raise ValueError(f"Unknown OSNet variant '{variant}'. Choose from: {list(_CONFIGS)}")
    cfg = _CONFIGS[variant]
    return OSNet(
        num_classes=num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=cfg["channels"],
    )
