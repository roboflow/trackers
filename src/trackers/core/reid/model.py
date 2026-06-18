# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Architecture-agnostic appearance feature extractor for re-identification.

A :class:`ReIDModel` is defined by three independent, swappable axes:

- **architecture** — selected by name from
  :mod:`trackers.core.reid.architectures` (or a timm model / a raw ``nn.Module``);
- **weights** — any local file or ``hf://`` URL, see
  :mod:`trackers.core.reid.weights`;
- **preprocessing** — an explicit :class:`~trackers.core.reid.preprocessing.ReIDPreprocessing`
  describing every transformation applied to a crop and to the output embedding.

Each axis is a parameter, so changing the backbone, the weights, or the
preprocessing never requires a new class. Adding a new architecture is a matter
of registering a builder in :mod:`trackers.core.reid.architectures`.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

from trackers.core.reid.preprocessing import ReIDPreprocessing
from trackers.core.reid.weights import load_state_dict_into, resolve_weights

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

logger = logging.getLogger(__name__)


def _require_reid_deps() -> None:
    """Raise a descriptive ImportError when the reid optional deps are absent."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The reid feature requires optional dependencies. "
            "Install them with:  pip install trackers[reid]"
        ) from exc


def _select_device(device: str) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Args:
        device: ``"auto"`` to pick the best available device, or any value
            accepted by :class:`torch.device` (``"cpu"``, ``"cuda"``, ``"mps"``, …).

    Returns:
        The resolved :class:`torch.device`.
    """
    import torch

    if device == "auto":
        from trackers.utils.device import _best_device

        return _best_device()
    return torch.device(device)


class ReIDModel:
    """Appearance feature extractor for object re-identification.

    Wraps a backbone neural network plus an explicit preprocessing pipeline and
    exposes a single inference method (:meth:`extract_features`) that accepts
    ``supervision.Detections`` and a video frame and returns embedding vectors —
    one per detection.

    The model is **class-agnostic and architecture-agnostic by design**: there
    is nothing person-specific in the API, and the backbone, weights, and
    preprocessing are all independent parameters (see :meth:`from_pretrained`).
    The *default* checkpoint (OSNet on MSMT17) was trained on pedestrian images;
    a ``UserWarning`` is emitted when that default is used so callers are aware.

    Preprocessing is never hidden: the active
    :class:`~trackers.core.reid.preprocessing.ReIDPreprocessing` is available as
    :attr:`preprocessing` and is logged on construction.

    Typical usage (inference only)::

        model = ReIDModel.from_pretrained()                     # default OSNet/MSMT17
        embeddings = model.extract_features(detections, frame)  # (N, 512)

    Args:
        backbone: A :class:`torch.nn.Module` that accepts a ``(B, 3, H, W)``
            float tensor and returns a ``(B, D)`` embedding tensor when in eval
            mode.
        device: The :class:`torch.device` the backbone lives on.
        preprocessing: The explicit input/output processing applied around the
            backbone.
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
        """The explicit preprocessing pipeline applied by this model."""
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
        """Build a re-ID model from a source, with optional overrides.

        This is the single, architecture-agnostic entry point. The three axes
        (architecture, weights, preprocessing) are resolved from *source* in
        this order, then any explicit keyword overrides are applied:

        1. ``source is None`` and ``architecture is None`` → default curated model.
        2. ``source in ALIASES`` → the alias's :class:`~trackers.core.reid.registry.ModelCard`.
        3. *source* is a directory or ``hf://`` repo that contains a
           ``reid_config.json`` → self-describing checkpoint (written by
           :meth:`save_pretrained`).
        4. *source* is a bare weights file (``*.pth`` / ``*.safetensors``) →
           ``architecture`` is **required**; raises :class:`ValueError` if missing.
        5. ``source is None`` and ``architecture is not None`` → build the
           architecture with its own pretrained weights (timm ImageNet) or
           random weights (OSNet); no external weights loaded.

        Args:
            source: A curated alias (e.g. ``"osnet_x1_0_msmt17_combineall"``),
                an ``"hf://org/repo"`` repo URL, an ``"hf://org/repo/file.pth"``
                file URL, a local ``.pth`` / ``.safetensors`` path, a local
                directory produced by :meth:`save_pretrained`, or ``None`` to
                use the default model.
            architecture: Override the resolved architecture. One of a
                registered name (``"osnet_x1_0"``, ``"timm:resnet50"``, …),
                a pre-built :class:`torch.nn.Module`, or ``None`` to keep the
                resolved value. Required when *source* is a bare weights file.
            preprocessing: Override the resolved preprocessing. ``None`` uses
                the card's preprocessing or the default.
            device: Compute device — ``"auto"`` selects the best available
                device, or pass any :class:`torch.device`-compatible string.

        Returns:
            A :class:`ReIDModel` ready for inference.

        Raises:
            ValueError: If *source* is a bare weights file and *architecture*
                is not provided.

        Examples:
            >>> model = ReIDModel.from_pretrained()  # doctest: +SKIP
            >>> model = ReIDModel.from_pretrained(
            ...     "/runs/osnet.pth", architecture="osnet_x1_0"
            ... )  # doctest: +SKIP
            >>> model = ReIDModel.from_pretrained(
            ...     architecture="timm:resnet50"
            ... )  # doctest: +SKIP
        """
        from trackers.core.reid.architectures import build_architecture
        from trackers.core.reid.registry import DEFAULT_MODEL, resolve_model_card

        _require_reid_deps()
        resolved_device = _select_device(device)

        # §2.2 step 1: no source and no architecture → use the default alias.
        using_default = False
        if source is None and architecture is None:
            source = DEFAULT_MODEL
            using_default = True

        # §2.2 steps 2-3: resolve a ModelCard from an alias or config-bearing
        # directory/repo.
        card = None
        if source is not None:
            card = resolve_model_card(source)

        if card is not None:
            resolved_arch = architecture if architecture is not None else card.architecture
            resolved_weights = card.weights
            resolved_preprocessing = (
                preprocessing if preprocessing is not None else card.preprocessing
            )
            resolved_warning = card.domain_warning if using_default else None

        elif source is None:
            # §2.2 step 5: architecture-only; no external weights loaded.
            resolved_arch = architecture
            resolved_weights = None
            resolved_preprocessing = (
                preprocessing if preprocessing is not None else ReIDPreprocessing()
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
                preprocessing if preprocessing is not None else ReIDPreprocessing()
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
            report = load_state_dict_into(backbone, local_path, resolved_device)
            logger.info(
                "ReIDModel weights (%s): %s", resolved_weights, report.summary()
            )

        backbone.to(resolved_device)

        instance = cls(backbone, resolved_device, resolved_preprocessing)
        # Store the architecture name (string only) to enable save_pretrained.
        instance._architecture = (
            resolved_arch if isinstance(resolved_arch, str) else None
        )
        return instance

    def save_pretrained(self, directory: str) -> None:
        """Write the model to *directory* as a self-describing checkpoint.

        Produces two files that :meth:`from_pretrained` can reload without any
        manual architecture hint:

        - ``weights.safetensors`` — the backbone's state dict (classifier head
          excluded; it is stripped at load time via
          :func:`~trackers.core.reid.weights.load_state_dict_into`).
        - ``reid_config.json`` — architecture name + preprocessing config.

        Args:
            directory: Target directory; created automatically if absent.

        Raises:
            ValueError: If the architecture name is unknown (e.g. the model was
                built directly from a raw ``nn.Module`` without going through
                :meth:`from_pretrained`).

        Examples:
            >>> model = ReIDModel.from_pretrained(
            ...     architecture="osnet_x1_0"
            ... )  # doctest: +SKIP
            >>> model.save_pretrained("/runs/my_reid")  # doctest: +SKIP
        """
        from safetensors.torch import save_file

        from trackers.core.reid.registry import ModelCard, save_model_config

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
        """Extract embeddings from a list of image file paths.

        Designed for the re-ID evaluation workflow where each image is already
        a cropped identity photograph (e.g. MSMT17 / Market-1501 samples).
        Images are read as RGB via Pillow, so the preprocessing ``to_rgb`` flag
        (which concerns BGR OpenCV crops) does not apply here.
        For the tracking use case where crops are derived from bounding boxes
        in a video frame, use :meth:`extract_features` instead.

        Args:
            image_paths: Absolute or relative paths to image files.
            batch_size: Number of images to process in a single forward pass.
                Reduce if running out of GPU memory.
            normalize: If ``True`` (default), L2-normalise each embedding so
                cosine similarity equals the dot product. Set ``False`` to
                return the raw backbone features (needed for Euclidean distance).

        Returns:
            Float32 array of shape ``(N, D)`` where *N* is ``len(image_paths)``
            and *D* is the backbone's embedding dimension.

        Examples:
            >>> model = None  # doctest: +SKIP
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
        """Extract appearance embeddings for each detection.

        Each bounding box in *detections* is cropped from *frame*, then passed
        through the explicit preprocessing pipeline (:attr:`preprocessing`) and
        the backbone. Crops are assumed BGR (OpenCV convention) and converted to
        RGB when ``preprocessing.to_rgb`` is set; the output is L2-normalised
        when ``preprocessing.normalize_embeddings`` is set.

        Args:
            detections: Detections whose bounding boxes define the crops.
                The frame must contain the full scene; boxes outside the frame
                boundaries are clamped automatically by :func:`sv.crop_image`.
            frame: BGR image array (e.g. from ``cv2.VideoCapture``). Must be
                the same frame the detections were produced on.

        Returns:
            Float32 array of shape ``(N, D)`` where *N* is ``len(detections)``
            and *D* is the backbone's embedding dimension. Returns an empty
            ``(0, 0)`` array when *detections* is empty.

        Examples:
            >>> import numpy as np
            >>> import supervision as sv
            >>> detections = sv.Detections.empty()
            >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> class _FakeModel:
            ...     def extract_features(self, d, f): return np.empty((0, 0))
            >>> _FakeModel().extract_features(detections, frame).shape
            (0, 0)
        """
        if len(detections) == 0:
            return np.empty((0, 0), dtype=np.float32)

        import torch
        from PIL import Image as PILImage

        crops = []
        for box in detections.xyxy:
            crop = sv.crop_image(image=frame, xyxy=box.astype(int))
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
