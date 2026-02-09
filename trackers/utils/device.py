# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import torch


def best_device() -> torch.device:
    """Auto-detect the best available compute device.

    Returns:
        torch.device: 'cuda' if NVIDIA GPU available, 'mps' if Apple Silicon,
        otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
