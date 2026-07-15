# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID appearance encoder: loading, inference, and checkpoint I/O."""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import supervision as sv
import torch
import torch.nn as nn
from PIL import Image as PILImage
from safetensors.torch import save_file

from trackers.core.reid.architectures import build_architecture, checkpoint_remap_for_architecture
from trackers.core.reid.models.loaders import load_state_dict_for_architecture, resolve_weights
from trackers.core.reid.models.preprocessing import ReIDPreprocessing
from trackers.core.reid.models.registry import (
    DEFAULT_MODEL,
    FASTREID_MOT17_SBS50,
    ModelCard,
    default_preprocessing_for_architecture,
    resolve_model_card,
    save_model_config,
)
from trackers.utils.device import _best_device

logger = logging.getLogger(__name__)


def _clamp_xyxy_to_frame(box: np.ndarray, height: int, width: int) -> np.ndarray:
    """Clip a box to frame bounds with at least 1 px width and height."""
    x1, y1, x2, y2 = box.astype(float)
    max_x = max(width, 1)
    max_y = max(height, 1)
    x1 = float(np.clip(x1, 0, max_x - 1))
    y1 = float(np.clip(y1, 0, max_y - 1))
    x2 = float(np.clip(x2, x1 + 1, max_x))
    y2 = float(np.clip(y2, y1 + 1, max_y))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _select_device(device: str) -> torch.device:
    """Resolve ``"auto"`` or a device string to a ``torch.device``."""
    if device == "auto":
        return _best_device()
    return torch.device(device)


class ReIDModel:
    """Appearance encoder with loading, preprocessing, and checkpoint I/O.

    Implements ``ReIDEncoder`` (``extract_features`` and
    ``extract_features_from_paths``), plus ``from_pretrained`` and
    ``save_pretrained``. The default checkpoint is pedestrian-trained; pass
    ``source`` / ``architecture`` for other domains.

    Args:
        backbone: Feature-extractor module (``(B, 3, H, W)`` → ``(B, D)`` in eval).
        device: Device the backbone runs on.
        preprocessing: Crop and embedding preprocessing.
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        preprocessing: ReIDPreprocessing,
    ) -> None:
        self._backbone = backbone
        self._device = device
        self._preprocessing = preprocessing
        self._transforms = preprocessing.build_transform()
        # Set by from_pretrained() to enable save_pretrained(); None if the
        # model was constructed directly with a raw nn.Module architecture.
        self._architecture: str | None = None
        logger.info("ReIDModel preprocessing: %s", preprocessing.describe())

    @property
    def preprocessing(self) -> ReIDPreprocessing:
        """Active preprocessing pipeline."""
        return self._preprocessing

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        source: str | None = None,
        *,
        architecture: str | nn.Module | None = None,
        preprocessing: ReIDPreprocessing | None = None,
        device: str = "auto",
    ) -> ReIDModel:
        """Load a ``ReIDModel`` from an alias, Hub/local checkpoint, or architecture-only init.

        ``source`` may be a curated alias, ``hf://`` repo or file, a local path
        or directory with ``reid_config.json``, or ``None`` for the default
        model. A bare ``.pth`` / ``.safetensors`` file requires ``architecture``.

        Args:
            source: Model source (alias, Hub URL, local path, or ``None``).
            architecture: Backbone override; required for bare weight files.
            preprocessing: Preprocessing override.
            device: Compute device (``"auto"`` picks the best available).

        Returns:
            Loaded model with the backbone on ``device``.

        Raises:
            ValueError: If a bare weights file is given without ``architecture``.
        """
        resolved_device = _select_device(device)

        # No source and no architecture → use the default curated alias.
        if source is None and architecture is None:
            source = DEFAULT_MODEL

        # Resolve a ModelCard from an alias or config-bearing directory/repo.
        card = None
        if source is not None:
            card = resolve_model_card(source)

        if card is not None:
            resolved_arch = architecture if architecture is not None else card.architecture
            resolved_weights = card.weights
            resolved_preprocessing = preprocessing if preprocessing is not None else card.preprocessing
            resolved_warning = card.domain_warning

        elif source is None:
            # Architecture-only build: random init, no network downloads.
            if architecture is None:
                raise ValueError("architecture is required when source is None")
            resolved_arch = architecture
            resolved_weights = None
            resolved_preprocessing = (
                preprocessing
                if preprocessing is not None
                else default_preprocessing_for_architecture(architecture if isinstance(architecture, str) else None)
            )
            resolved_warning = None

        else:
            # Bare weights file — architecture is required.
            if architecture is None:
                raise ValueError(
                    f"Cannot load {source!r}: it appears to be a bare weights "
                    "file but no `architecture` was provided. Pass e.g. "
                    "architecture='osnet_x1_0' to specify the model "
                    "architecture, or point to a directory produced by "
                    "save_pretrained() for a self-describing checkpoint."
                )
            resolved_arch = architecture
            resolved_weights = source
            resolved_preprocessing = (
                preprocessing
                if preprocessing is not None
                else default_preprocessing_for_architecture(architecture if isinstance(architecture, str) else None)
            )
            resolved_warning = None

        if resolved_warning:
            warnings.warn(resolved_warning, UserWarning, stacklevel=2)

        # Always build offline here. Checkpoint weights (when present) are loaded
        # explicitly below; architecture-only builds stay randomly initialised.
        backbone = build_architecture(resolved_arch, pretrained=False)
        backbone.eval()

        if resolved_weights is not None:
            local_path = resolve_weights(resolved_weights)
            arch_name = resolved_arch if isinstance(resolved_arch, str) else ""
            required_fraction = 1.0 if source in (DEFAULT_MODEL, FASTREID_MOT17_SBS50) else None
            report = load_state_dict_for_architecture(
                backbone,
                local_path,
                resolved_device,
                arch_name,
                required_match_fraction=required_fraction,
                remap=checkpoint_remap_for_architecture(arch_name) if arch_name else None,
            )
            logger.info("ReIDModel weights (%s): %s", resolved_weights, report.summary())

        backbone.to(resolved_device)

        instance = cls(backbone, resolved_device, resolved_preprocessing)
        # Store the architecture name (string only) to enable save_pretrained.
        instance._architecture = resolved_arch if isinstance(resolved_arch, str) else None
        return instance

    def save_pretrained(self, directory: str) -> None:
        """Persist ``weights.safetensors`` and ``reid_config.json`` for later loads.

        Args:
            directory: Output directory (created if missing).

        Raises:
            ValueError: If the architecture name is unknown.
        """
        if self._architecture is None:
            raise ValueError(
                "Cannot save a model whose architecture name is unknown. "
                "Build the model via from_pretrained() with a named "
                "architecture (e.g. 'osnet_x1_0', 'timm:resnet50') to "
                "enable save_pretrained()."
            )

        os.makedirs(directory, exist_ok=True)

        weights_path = os.path.join(directory, "weights.safetensors")
        save_file(self._backbone.state_dict(), weights_path)

        card = ModelCard(
            architecture=self._architecture,
            weights=None,
            preprocessing=self._preprocessing,
        )
        save_model_config(card, directory)
        logger.info("ReIDModel saved to %s", directory)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def extract_features_from_paths(
        self,
        image_paths: list[str],
        *,
        batch_size: int = 64,
        normalize: bool = False,
    ) -> np.ndarray:
        """Extract embeddings from pre-cropped image paths.

        For bbox crops from a video frame, use ``extract_features``.

        Args:
            image_paths: Paths to RGB-ready crop images.
            batch_size: Images per forward pass.
            normalize: L2-normalise embeddings when ``True``.

        Returns:
            Float32 array of shape ``(N, D)``.

        Raises:
            ValueError: If ``batch_size`` is less than 1.
        """
        if not image_paths:
            return np.empty((0, 0), dtype=np.float32)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []
            for p in batch_paths:
                img = PILImage.open(p).convert("RGB")
                crop = self._preprocessing.resize_crop(np.asarray(img))
                tensors.append(self._transforms(PILImage.fromarray(crop)))

            batch = torch.stack(tensors).to(self._device)
            with torch.inference_mode():
                embs = self._backbone(batch)
            if normalize:
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            batch_np = embs.cpu().numpy().astype(np.float32)
            if batch_np.ndim != 2:
                raise ValueError(f"embeddings must be 2-D, got shape {batch_np.shape}")
            if batch_np.size > 0 and not np.all(np.isfinite(batch_np)):
                raise ValueError("embeddings must contain only finite values")
            all_embeddings.append(batch_np)

        return np.concatenate(all_embeddings, axis=0)

    def extract_features(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Extract appearance embeddings for each detection.

        Args:
            detections: Detections whose ``xyxy`` boxes define the crops.
            frame: BGR video frame the detections were produced on.

        Returns:
            Float32 array of shape ``(N, D)``, or ``(0, 0)`` when empty.

        Note:
            Embeddings are L2-normalised when
            ``preprocessing.normalize_embeddings`` is ``True``.
        """
        if len(detections) == 0:
            return np.empty((0, 0), dtype=np.float32)

        frame_h, frame_w = frame.shape[:2]
        crops = []
        for box in detections.xyxy:
            safe_box = _clamp_xyxy_to_frame(box, frame_h, frame_w)
            crop = sv.crop_image(image=frame, xyxy=safe_box.astype(int))
            if self._preprocessing.to_rgb:
                crop = crop[:, :, ::-1].copy()
            crop = self._preprocessing.resize_crop(crop)
            pil_crop = PILImage.fromarray(crop)
            crops.append(self._transforms(pil_crop))

        batch = torch.stack(crops).to(self._device)

        with torch.inference_mode():
            embeddings = self._backbone(batch)

        if self._preprocessing.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        out = embeddings.cpu().numpy().astype(np.float32)
        if out.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {out.shape}")
        if out.size > 0 and not np.all(np.isfinite(out)):
            raise ValueError("embeddings must contain only finite values")
        return out
