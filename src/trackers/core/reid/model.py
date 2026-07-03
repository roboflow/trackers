# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Re-ID appearance encoder: loading, inference, and checkpoint I/O."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

from trackers.core.reid.models.loaders import load_state_dict_for_architecture, resolve_weights
from trackers.core.reid.models.preprocessing import ReIDPreprocessing
from trackers.core.reid.models.registry import default_preprocessing_for_architecture

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

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


def _require_reid_deps() -> None:
    """Raise a descriptive ImportError when the reid optional deps are absent."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The reid feature requires optional dependencies. Install them with:  pip install trackers[reid]"
        ) from exc


def _select_device(device: str) -> torch.device:
    """Resolve ``"auto"`` or a device string to a :class:`torch.device`."""
    import torch

    if device == "auto":
        from trackers.utils.device import _best_device

        return _best_device()
    return torch.device(device)


class ReIDModel:
    """Appearance feature extractor for object re-identification.

    Wraps a backbone and preprocessing pipeline. The default checkpoint is
    pedestrian-trained; pass ``source``/``architecture`` for other domains.

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
        """Build a :class:`ReIDModel` from a checkpoint source.

        ``source`` may be a curated alias, ``hf://`` repo or file, a local path
        or directory with ``reid_config.json``, or ``None`` for the default
        model. A bare ``.pth``/``.safetensors`` file requires ``architecture``.

        Args:
            source: Model source (alias, Hub URL, local path, or ``None``).
            architecture: Backbone override; required for bare weight files.
            preprocessing: Preprocessing override.
            device: Compute device (``"auto"`` picks the best available).

        Returns:
            Loaded :class:`ReIDModel`.
        """
        from trackers.core.reid.architectures import build_architecture
        from trackers.core.reid.models.registry import DEFAULT_MODEL, resolve_model_card

        _require_reid_deps()
        resolved_device = _select_device(device)

        # §2.2 step 1: no source and no architecture → use the default alias.
        if source is None and architecture is None:
            source = DEFAULT_MODEL

        # §2.2 steps 2-3: resolve a ModelCard from an alias or config-bearing
        # directory/repo.
        card = None
        if source is not None:
            card = resolve_model_card(source)

        if card is not None:
            resolved_arch = architecture if architecture is not None else card.architecture
            resolved_weights = card.weights
            resolved_preprocessing = preprocessing if preprocessing is not None else card.preprocessing
            resolved_warning = card.domain_warning

        elif source is None:
            # §2.2 step 5: architecture-only; no external weights loaded.
            resolved_arch = architecture
            resolved_weights = None
            resolved_preprocessing = (
                preprocessing
                if preprocessing is not None
                else default_preprocessing_for_architecture(
                    architecture if isinstance(architecture, str) else None
                )
            )
            resolved_warning = None

        else:
            # §2.2 step 4: bare weights file — architecture is required.
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
                else default_preprocessing_for_architecture(
                    architecture if isinstance(architecture, str) else None
                )
            )
            resolved_warning = None

        if resolved_warning:
            warnings.warn(resolved_warning, UserWarning, stacklevel=2)

        # Build the backbone; use the architecture's own pretrained weights
        # (e.g. ImageNet via timm) only when no external weights will be loaded.
        use_pretrained = resolved_weights is None
        backbone = build_architecture(resolved_arch, pretrained=use_pretrained)
        backbone.eval()

        if resolved_weights is not None:
            local_path = resolve_weights(resolved_weights)
            arch_name = resolved_arch if isinstance(resolved_arch, str) else ""
            report = load_state_dict_for_architecture(
                backbone,
                local_path,
                resolved_device,
                arch_name,
            )
            logger.info("ReIDModel weights (%s): %s", resolved_weights, report.summary())

        backbone.to(resolved_device)

        instance = cls(backbone, resolved_device, resolved_preprocessing)
        # Store the architecture name (string only) to enable save_pretrained.
        instance._architecture = resolved_arch if isinstance(resolved_arch, str) else None
        return instance

    def save_pretrained(self, directory: str) -> None:
        """Write ``weights.safetensors`` and ``reid_config.json`` to *directory*."""
        from safetensors.torch import save_file

        from trackers.core.reid.models.registry import ModelCard, save_model_config

        if self._architecture is None:
            raise ValueError(
                "Cannot save a model whose architecture name is unknown. "
                "Build the model via from_pretrained() with a named "
                "architecture (e.g. 'osnet_x1_0', 'timm:resnet50') to "
                "enable save_pretrained()."
            )

        import os

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
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Extract embeddings from pre-cropped image paths (evaluation use).

        For bbox crops from a video frame, use :meth:`extract_features`.

        Args:
            image_paths: Paths to RGB-ready crop images.
            batch_size: Images per forward pass.
            normalize: L2-normalise embeddings when ``True`` (default).

        Returns:
            Float32 array of shape ``(N, D)``.
        """
        if not image_paths:
            return np.empty((0, 0), dtype=np.float32)

        import torch
        from PIL import Image as PILImage

        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []
            for p in batch_paths:
                img = PILImage.open(p).convert("RGB")
                tensors.append(self._transforms(img))

            batch = torch.stack(tensors).to(self._device)
            with torch.inference_mode():
                embs = self._backbone(batch)
            if normalize:
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embeddings.append(embs.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def extract_features(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Extract L2-normalised appearance embeddings for each detection.

        Args:
            detections: Detections whose ``xyxy`` boxes define the crops.
            frame: BGR video frame the detections were produced on.

        Returns:
            Float32 array of shape ``(N, D)``, or ``(0, 0)`` when empty.
        """
        if len(detections) == 0:
            return np.empty((0, 0), dtype=np.float32)

        import torch
        from PIL import Image as PILImage

        frame_h, frame_w = frame.shape[:2]
        crops = []
        for box in detections.xyxy:
            safe_box = _clamp_xyxy_to_frame(box, frame_h, frame_w)
            crop = sv.crop_image(image=frame, xyxy=safe_box.astype(int))
            if crop.size == 0:
                crop = np.zeros((1, 1, 3), dtype=frame.dtype)
            if self._preprocessing.to_rgb:
                crop = crop[:, :, ::-1].copy()
            pil_crop = PILImage.fromarray(crop)
            crops.append(self._transforms(pil_crop))

        batch = torch.stack(crops).to(self._device)

        with torch.inference_mode():
            embeddings = self._backbone(batch)

        if self._preprocessing.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)
