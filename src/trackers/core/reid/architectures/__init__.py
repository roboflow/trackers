# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Re-ID backbone builders (OSNet variants and ``timm:`` models)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

# Width variants of the clean-room OSNet implementation.
_OSNET_VARIANTS: dict[str, str] = {
    "osnet_x0_25": "x0_25",
    "osnet_x0_5": "x0_5",
    "osnet_x0_75": "x0_75",
    "osnet_x1_0": "x1_0",
}

_TIMM_PREFIX = "timm:"


def build_architecture(
    architecture: str | nn.Module,
    *,
    num_classes: int = 0,
    pretrained: bool = False,
) -> nn.Module:
    """Build a backbone from ``osnet_*``, ``timm:<name>``, or a pre-built module."""
    if not isinstance(architecture, str):
        # Pre-built module: use as-is, ignoring num_classes and pretrained.
        return architecture  # type: ignore[return-value]

    if architecture.startswith(_TIMM_PREFIX):
        import timm

        name = architecture[len(_TIMM_PREFIX) :]
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    if architecture in _OSNET_VARIANTS:
        from trackers.core.reid.architectures.osnet import build_osnet

        # OSNet weights are always loaded explicitly via the weights axis, so
        # pretrained is intentionally ignored here.
        return build_osnet(variant=_OSNET_VARIANTS[architecture], num_classes=num_classes)

    if architecture == "fastreid_sbs_resnest50":
        from trackers.core.reid.architectures.fastreid_sbs import build_fastreid_sbs_resnest50

        return build_fastreid_sbs_resnest50(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(
        f"Unknown architecture {architecture!r}. Choose a registered name "
        f"({list_architectures()}), a timm model as 'timm:<name>' (e.g. "
        f"'timm:resnet50'), or pass a torch.nn.Module instance."
    )


def list_architectures() -> list[str]:
    """Return registered architecture names (timm models use ``timm:<name>``)."""
    return sorted([*_OSNET_VARIANTS, "fastreid_sbs_resnest50"])
