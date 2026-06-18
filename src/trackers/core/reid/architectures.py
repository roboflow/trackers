# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Architecture-agnostic backbone builder for :class:`~trackers.core.reid.model.ReIDModel`.

The re-ID feature is **architecture-agnostic**: a model is described by three
independent axes — *architecture* (this module), *weights*
(:mod:`trackers.core.reid.weights`), and *preprocessing*
(:mod:`trackers.core.reid.preprocessing`). Swapping any one of them is a
parameter change, never a new class.

Adding a new architecture (FastReID, SOLIDER, …) is a matter of registering a
builder variant here — no changes to ``ReIDModel`` are required.

All torch / timm imports are performed lazily inside the build callables so
that importing this module (and therefore ``trackers.core.reid``) does not
require the optional ``[reid]`` dependencies.
"""

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
    """Build a backbone neural network from an architecture selector.

    Args:
        architecture: One of:

            - a registered OSNet name, e.g. ``"osnet_x1_0"`` (see
              :func:`list_architectures`);
            - a timm model name prefixed with ``"timm:"``, e.g.
              ``"timm:resnet50"``;
            - a pre-built :class:`torch.nn.Module` instance (used as-is).

        num_classes: Number of output classes. ``0`` returns pooled features
            directly (feature-extractor mode). ``>0`` adds a classification
            head (training mode). Ignored for pre-built modules.
        pretrained: Load the architecture's own pretrained weights (e.g.
            ImageNet for timm). Ignored for OSNet (weights always supplied via
            the weights axis) and for pre-built modules.

    Returns:
        A :class:`torch.nn.Module` that yields a ``(B, D)`` embedding tensor
        when called with a ``(B, 3, H, W)`` float input in eval mode.

    Raises:
        ValueError: If *architecture* is an unknown string selector.

    Examples:
        >>> backbone = build_architecture("osnet_x1_0")  # doctest: +SKIP
        >>> backbone = build_architecture("timm:resnet50", pretrained=True)  # doctest: +SKIP
    """
    if not isinstance(architecture, str):
        # Pre-built module: use as-is, ignoring num_classes and pretrained.
        return architecture  # type: ignore[return-value]

    if architecture.startswith(_TIMM_PREFIX):
        import timm

        name = architecture[len(_TIMM_PREFIX):]
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    if architecture in _OSNET_VARIANTS:
        from trackers.core.reid.osnet import build_osnet

        # OSNet weights are always loaded explicitly via the weights axis, so
        # pretrained is intentionally ignored here.
        return build_osnet(variant=_OSNET_VARIANTS[architecture], num_classes=num_classes)

    raise ValueError(
        f"Unknown architecture {architecture!r}. Choose a registered name "
        f"({list_architectures()}), a timm model as 'timm:<name>' (e.g. "
        f"'timm:resnet50'), or pass a torch.nn.Module instance."
    )


def list_architectures() -> list[str]:
    """Return the registered architecture names.

    Returns:
        Sorted list of registered names. timm models are available via the
        ``"timm:<name>"`` prefix (e.g. ``"timm:resnet50"``).

    Examples:
        >>> list_architectures()
        ['osnet_x0_25', 'osnet_x0_5', 'osnet_x0_75', 'osnet_x1_0']
    """
    return sorted(_OSNET_VARIANTS)
