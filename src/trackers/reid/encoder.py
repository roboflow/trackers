# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Encoder protocol for ReID association, and loading an encoder by name."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import supervision as sv


class ReIDEncoder(Protocol):
    """Encoder with ``extract_embeddings(detections, frame)``."""

    def extract_embeddings(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        """Return appearance embeddings for each detection box.

        Args:
            detections: Boxes to embed (``xyxy``).
            frame: BGR frame the detections were produced on.

        Returns:
            Float32 array of shape ``(N, D)``, or ``(0, 0)`` when empty.
        """
        ...


def _load_reid_encoder(source: str, *, device: str = "auto", architecture: str | None = None) -> ReIDEncoder:
    """Load a model from the ``reid`` package to use as the Re-ID encoder.

    ``reid`` is imported here instead of at the top of the module, so importing
    trackers does not load it or torch.

    Args:
        source: Model alias, ``hf://`` URL or local path, as accepted by
            ``ReIDModel.from_pretrained``.
        device: Device to run on: ``auto``, ``cpu``, ``cuda`` or ``mps``.
        architecture: Architecture of the weights, when ``source`` is a bare
            weights file.

    Returns:
        The loaded model.

    Raises:
        ImportError: If ``reid`` is installed but fails to import.
        ValueError: If the ``reid`` extra is not installed or the model fails
            to load.
    """
    try:
        from reid import ReIDModel
    except ImportError as exc:
        if exc.name != "reid":
            raise
        raise ValueError(
            "ReID tracking requires the optional `trackers[reid]` extra.\nInstall with: pip install 'trackers[reid]'"
        ) from exc

    try:
        return ReIDModel.from_pretrained(source=source, architecture=architecture, device=device)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Failed to load ReID model: {exc}") from exc
