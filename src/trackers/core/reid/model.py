# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

_DEFAULT_HF_REPO = "kaiyangzhou/osnet"
_DEFAULT_HF_FILENAME = (
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
_DEFAULT_DOMAIN_WARNING = (
    "The default ReIDModel weights were trained on MSMT17 pedestrian images. "
    "Performance may degrade significantly on non-person objects (vehicles, animals, products, etc.). "
    "Pass a domain-specific checkpoint via from_pretrained() or from_timm() for other use cases."
)

# Input resolution expected by OSNet (height × width).
_OSNET_INPUT_SIZE = (256, 128)
# ImageNet normalisation used by all OSNet checkpoints.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_reid_deps() -> None:
    """Raise a descriptive ImportError when the reid optional deps are absent."""
    try:
        import timm  # noqa: F401
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


def _build_osnet_transforms(input_size: tuple[int, int] = _OSNET_INPUT_SIZE):
    """Return the standard OSNet inference transform pipeline.

    Crops arrive as BGR numpy arrays from OpenCV; this pipeline converts to
    RGB, resizes, converts to a float tensor, and applies ImageNet normalisation.

    Args:
        input_size: ``(height, width)`` to resize each crop to.

    Returns:
        A ``torchvision.transforms.Compose`` instance.
    """
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    # ToPILImage is applied first inside extract_features (crop is BGR ndarray).
    return Compose([
        Resize(input_size),
        ToTensor(),
        Normalize(mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD)),
    ])


def _load_osnet_checkpoint(model: nn.Module, path: str, device: torch.device) -> None:
    """Load pretrained weights into *model*, ignoring the classifier head.

    The classifier weights differ across checkpoints (different ``num_classes``
    per dataset) and are not needed for inference — OSNet returns the embedding
    vector at ``model.eval()`` without passing through the classifier.

    Args:
        model: The :class:`~trackers.core.reid.osnet.OSNet` instance to load into.
        path: Local filesystem path to a ``.pth`` checkpoint.
        device: Device to map tensors to during load.
    """
    import torch

    state_dict = torch.load(path, map_location=device, weights_only=False)

    # Some checkpoints are wrapped in a {"state_dict": ...} dict.
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Strip module. prefix from DataParallel-saved checkpoints.
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    # Drop classifier weights — shapes differ across datasets and are not used.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}

    model_dict = model.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(matched)
    model.load_state_dict(model_dict)


class ReIDModel:
    """Appearance feature extractor for object re-identification.

    Wraps a backbone neural network and exposes a single inference method
    (:meth:`extract_features`) that accepts ``supervision.Detections`` and a
    video frame and returns L2-normalised embedding vectors — one per detection.

    The model is **class-agnostic by design**: there is nothing person-specific
    in the API. The *default* pretrained checkpoint (OSNet on MSMT17) was trained
    on pedestrian images; a ``UserWarning`` is emitted when that default is used
    so callers are aware. Pass any HF repository or timm model name for other
    domains.

    Typical usage (inference only)::

        model = ReIDModel.from_pretrained()            # default OSNet/MSMT17
        embeddings = model.extract_features(detections, frame)  # (N, 512)

    Args:
        backbone: A :class:`torch.nn.Module` that accepts a ``(B, 3, H, W)``
            float tensor and returns a ``(B, D)`` embedding tensor when in eval
            mode.
        device: The :class:`torch.device` the backbone lives on.
        transforms: A callable that maps a PIL Image to a ``(3, H, W)`` float
            tensor (standard torchvision transform pipeline).
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        transforms,
    ) -> None:
        self._backbone = backbone
        self._device = device
        self._transforms = transforms

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = _DEFAULT_HF_REPO,
        filename: str = _DEFAULT_HF_FILENAME,
        device: str = "auto",
        variant: str = "x1_0",
    ) -> ReIDModel:
        """Load an OSNet backbone with weights from a Hugging Face repository.

        The default ``repo_id`` and ``filename`` point to an OSNet x1.0 model
        trained on **MSMT17** (pedestrian re-ID). A :class:`UserWarning` is
        emitted when the defaults are used to remind callers that the weights
        are domain-specific.

        Args:
            repo_id: Hugging Face model repository ID
                (e.g. ``"kaiyangzhou/osnet"``).
            filename: Filename of the ``.pth`` checkpoint inside the repository.
            device: Compute device — ``"auto"`` selects the best available
                device, or pass any :class:`torch.device`-compatible string.
            variant: OSNet width variant: ``"x1_0"`` (default), ``"x0_75"``,
                ``"x0_5"``, or ``"x0_25"``.

        Returns:
            A :class:`ReIDModel` ready for inference.

        Examples:
            >>> model = ReIDModel.from_pretrained()  # doctest: +SKIP
        """
        _require_reid_deps()

        from huggingface_hub import hf_hub_download

        from trackers.core.reid.osnet import build_osnet

        if repo_id == _DEFAULT_HF_REPO and filename == _DEFAULT_HF_FILENAME:
            warnings.warn(_DEFAULT_DOMAIN_WARNING, UserWarning, stacklevel=2)

        resolved_device = _select_device(device)

        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        backbone = build_osnet(variant=variant)
        backbone.eval()
        _load_osnet_checkpoint(backbone, local_path, resolved_device)
        backbone.to(resolved_device)

        return cls(backbone, resolved_device, _build_osnet_transforms())

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "auto",
        variant: str = "x1_0",
    ) -> ReIDModel:
        """Load an OSNet backbone from a local ``.pth`` checkpoint.

        Use this for checkpoints that are not hosted on the Hugging Face Hub —
        for example the torchreid model-zoo weights distributed via Google Drive.
        The classifier head is dropped on load (see :func:`_load_osnet_checkpoint`),
        so only the embedding backbone needs to match.

        Args:
            checkpoint_path: Local filesystem path to a ``.pth`` checkpoint.
            device: Compute device — ``"auto"`` selects the best available
                device, or pass any :class:`torch.device`-compatible string.
            variant: OSNet width variant: ``"x1_0"`` (default), ``"x0_75"``,
                ``"x0_5"``, or ``"x0_25"``.

        Returns:
            A :class:`ReIDModel` ready for inference.

        Examples:
            >>> model = ReIDModel.from_checkpoint("osnet_x1_0_market1501.pth")  # doctest: +SKIP
        """
        _require_reid_deps()

        from trackers.core.reid.osnet import build_osnet

        resolved_device = _select_device(device)

        backbone = build_osnet(variant=variant)
        backbone.eval()
        _load_osnet_checkpoint(backbone, checkpoint_path, resolved_device)
        backbone.to(resolved_device)

        return cls(backbone, resolved_device, _build_osnet_transforms())

    @classmethod
    def from_timm(
        cls,
        model_name: str,
        device: str = "auto",
        input_size: tuple[int, int] = _OSNET_INPUT_SIZE,
        pretrained: bool = True,
    ) -> ReIDModel:
        """Load any `timm <https://huggingface.co/docs/timm>`_ backbone.

        Use this when you want a ResNet50, EfficientNet, or any other
        timm-supported architecture instead of OSNet. The backbone is
        instantiated with ``num_classes=0`` so it returns pooled features
        directly.

        Args:
            model_name: A timm model name (e.g. ``"resnet50"``).
            device: Compute device.
            input_size: ``(height, width)`` to resize crops to before feeding
                the backbone. Default matches the OSNet standard (256×128).
            pretrained: Whether to load ImageNet-pretrained timm weights.

        Returns:
            A :class:`ReIDModel` ready for inference.

        Examples:
            >>> model = ReIDModel.from_timm("resnet50")  # doctest: +SKIP
        """
        _require_reid_deps()

        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from torchvision.transforms import Compose, ToPILImage

        resolved_device = _select_device(device)

        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        backbone.eval().to(resolved_device)

        cfg = resolve_data_config(backbone.pretrained_cfg)
        timm_transforms = create_transform(**cfg)
        transforms = Compose([ToPILImage(), timm_transforms])

        return cls(backbone, resolved_device, transforms)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def extract_features_from_paths(
        self,
        image_paths: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Extract L2-normalised embeddings from a list of image file paths.

        Designed for the re-ID evaluation workflow where each image is already
        a cropped identity photograph (e.g. MSMT17 / Market-1501 samples).
        For the tracking use case where crops are derived from bounding boxes
        in a video frame, use :meth:`extract_features` instead.

        Args:
            image_paths: Absolute or relative paths to image files. Images are
                opened as RGB via Pillow.
            batch_size: Number of images to process in a single forward pass.
                Reduce if running out of GPU memory.

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
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embeddings.append(embs.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def extract_features(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Extract L2-normalised appearance embeddings for each detection.

        Each bounding box in *detections* is cropped from *frame*, resized,
        and passed through the backbone. The resulting vectors are L2-normalised
        so cosine similarity equals the dot product.

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
            crop_bgr = sv.crop_image(image=frame, xyxy=box.astype(int))
            # sv.crop_image returns BGR; convert to RGB before the transform.
            crop_rgb = crop_bgr[:, :, ::-1].copy()
            pil_crop = PILImage.fromarray(crop_rgb)
            crops.append(self._transforms(pil_crop))

        batch = torch.stack(crops).to(self._device)

        with torch.inference_mode():
            embeddings = self._backbone(batch)

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)
