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
