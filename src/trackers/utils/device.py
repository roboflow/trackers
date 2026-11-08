# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _best_device() -> torch.device:
    """Return the best available PyTorch compute device, preferring acceleration.

    Returns:
        The selected device (``cuda``, ``mps``, or ``cpu``).

    Raises:
        ImportError: If PyTorch is not installed.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _validate_device(device: str, label: str | None = None) -> str:
    """Validate that a user-requested compute device is actually available.

    Distinct from :func:`_best_device`, which *selects* a device. This one takes
    what the caller asked for and fails loudly when it cannot be honoured, so a
    typo or a CPU-only install surfaces immediately rather than after model
    weights have been loaded.

    Args:
        device: Requested device string, for example ``cpu`` or ``cuda``.
        label: Optional component name used in the error message.

    Returns:
        The validated device string, unchanged.

    Raises:
        RuntimeError: If a CUDA device is requested but CUDA is unavailable.

    Examples:
        >>> _validate_device("cpu")
        'cpu'
    """
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        subject = f" for {label}" if label else ""
        raise RuntimeError(
            f"CUDA was requested{subject}, but torch.cuda.is_available() is False. "
            "Use a CPU device or install a CUDA-enabled PyTorch build."
        )
    return device
