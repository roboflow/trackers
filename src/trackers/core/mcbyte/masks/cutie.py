# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch

from trackers.core.mcbyte.masks.base import MaskOutput, MaskPropagator
from trackers.utils.downloader import _download_file

logger = logging.getLogger(__name__)

CUTIE_CHECKPOINT_URLS = {
    "base-mega": "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth",
}

CUTIE_CHECKPOINT_MD5S = {
    "base-mega": "a6071de6136982e396851903ab4c083a",
}

CUTIE_DEFAULT_WEIGHTS_PATHS = {
    "base-mega": Path("models/cutie/cutie-base-mega.pth"),
}


def _ensure_weights_exist(
    weights_path: Path,
    model_type: str,
) -> None:
    """Download the default Cutie weights if they are not available locally."""
    if weights_path.exists():
        return

    checkpoint_url = CUTIE_CHECKPOINT_URLS.get(model_type)
    if checkpoint_url is None:
        raise ValueError(f"No default Cutie checkpoint URL for model_type={model_type!r}.")

    parsed_url = urlparse(checkpoint_url)
    if parsed_url.scheme != "https":
        raise ValueError(f"Unsupported checkpoint URL scheme: {parsed_url.scheme!r}")

    weights_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Cutie checkpoint to %s", weights_path)
    _download_file(
        checkpoint_url,
        weights_path,
        md5=CUTIE_CHECKPOINT_MD5S.get(model_type),
    )


def _get_cutie_package_dir(cutie_module: Any) -> Path:
    """Return the root directory of the installed Cutie package."""

    if getattr(cutie_module, "__file__", None) is not None:
        return Path(cutie_module.__file__).resolve().parent

    module_paths = list(getattr(cutie_module, "__path__", []))
    if len(module_paths) == 0:
        raise RuntimeError("Could not resolve Cutie package directory.")

    return Path(module_paths[0]).resolve()


def _image_to_torch(
    frame: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Convert an RGB frame from ``(H, W, 3)`` NumPy format to ``(3, H, W)`` Cutie
    tensor format.
    """
    return (
        torch.from_numpy(frame.transpose(2, 0, 1))
        .float()
        .to(
            device,
            non_blocking=True,
        )
        / 255.0
    )


def _torch_prob_to_numpy_mask(prob: torch.Tensor) -> np.ndarray:
    """Convert Cutie probability output to an indexed NumPy mask."""
    return torch.max(prob, dim=0).indices.cpu().numpy().astype(np.uint8)


def _binary_masks_to_non_overlapping_torch(
    masks: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Convert binary masks to non-overlapping Cutie mask channels.

    Input masks have shape ``(N, H, W)``. If masks overlap, earlier masks keep
    priority, matching the original McByte initialization behavior. The returned
    tensor has shape ``(N, H, W)``, dtype ``float32``, and values ``0.0`` or ``1.0``.
    """
    occupied = np.zeros(masks.shape[1:], dtype=bool)
    non_overlapping_masks = np.zeros_like(masks, dtype=np.float32)

    for mask_index, mask in enumerate(masks):
        valid_mask = mask & ~occupied
        non_overlapping_masks[mask_index] = valid_mask.astype(np.float32)
        occupied |= valid_mask

    return torch.from_numpy(non_overlapping_masks).to(device)


def _indexed_mask_to_binary_masks(
    indexed_mask: np.ndarray,
    object_ids: list[int],
) -> np.ndarray:
    """Convert an indexed mask to one binary mask per Cutie object ID."""
    return np.stack(
        [indexed_mask == object_id for object_id in object_ids],
        axis=0,
    )


def _build_tracklet_object_dict(
    tracklet_mask_dict: dict[int, int],
) -> dict[int, int]:
    """Map tracklet IDs to Cutie object IDs.

    Mask generators return local mask array indices starting from 0. Cutie uses
    positive object IDs, with 0 reserved for background.
    """
    return {tracklet_id: local_mask_index + 1 for tracklet_id, local_mask_index in tracklet_mask_dict.items()}


def _compute_mask_avg_prob_dict(
    prob: torch.Tensor,
    object_ids: list[int],
) -> dict[int, float]:
    """Compute average Cutie confidence for each predicted object region.

    The input probability tensor has shape ``(num_objects + 1, H, W)``, where
    channel 0 is background. For each object ID, confidence is averaged only over
    pixels where that object wins the argmax prediction.
    """
    mask_maxes = torch.max(prob, dim=0).indices
    mask_avg_prob_dict: dict[int, float] = {}

    for object_id in object_ids:
        object_prob = prob[object_id][mask_maxes == object_id]
        if object_prob.numel() == 0:
            continue

        avg_prob = object_prob.mean().item()
        if not np.isnan(avg_prob):
            mask_avg_prob_dict[object_id] = avg_prob

    return mask_avg_prob_dict


class CutieMaskPropagator(MaskPropagator):
    """Propagate McByte masks over time using Cutie.

    This class wraps an externally installed Cutie package. It does not vendor
    Cutie source code into Trackers.

    The first version supports the core original McByte behavior:

    1. initialize Cutie memory from masks on a reference frame,
    2. propagate masks to the current frame.

    Adding new masks, removing masks, overlap-aware mask creation, and color
    preservation are intentionally left for later MaskManager-level integration.

    Args:
        weights_path: Optional path to Cutie checkpoint weights. If not provided,
            the default checkpoint path for ``model_type`` is used. If the file is
            missing, it is downloaded automatically.
        config_path: Optional path to Cutie's Hydra config directory. If not
            provided, the path is inferred from the installed ``cutie`` package.
        config_name: Hydra config name used by Cutie.
        device: Device used by Cutie, for example ``"cpu"`` or ``"cuda"``.
        use_amp: Whether to use CUDA automatic mixed precision during Cutie calls.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        model_type: str = "base-mega",
        config_path: str | Path | None = None,
        config_name: str = "eval_config",
        device: str = "cuda",
        use_amp: bool = True,
    ) -> None:
        try:
            import cutie
            from cutie.inference.inference_core import InferenceCore
            from cutie.inference.utils.args_utils import get_dataset_cfg
            from cutie.model.cutie import CUTIE
            from hydra import compose, initialize_config_dir
            from omegaconf import open_dict
        except ImportError as exc:
            msg = (
                "Cutie support requires Cutie to be installed. "
                "Install Cutie separately, for example by cloning the Cutie "
                "repository and running `pip install -e .` inside it."
            )
            raise ImportError(msg) from exc

        self.device = torch.device(device)

        self.use_amp = use_amp and self.device.type == "cuda"

        default_weights_path = CUTIE_DEFAULT_WEIGHTS_PATHS.get(model_type)
        if default_weights_path is None:
            raise ValueError(
                f"Unsupported model_type={model_type!r}. Supported types: {sorted(CUTIE_DEFAULT_WEIGHTS_PATHS)}"
            )

        self.weights_path = Path(weights_path) if weights_path is not None else default_weights_path

        _ensure_weights_exist(
            weights_path=self.weights_path,
            model_type=model_type,
        )

        if config_path is None:
            cutie_package_dir = _get_cutie_package_dir(cutie)
            self.config_path = cutie_package_dir / "config"
        else:
            self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Cutie config directory not found: {self.config_path}. "
                "Pass config_path explicitly if Cutie is installed in a custom location."
            )

        with initialize_config_dir(
            version_base="1.3.2",
            config_dir=str(self.config_path.resolve()),
            job_name="eval_config",
        ):
            cfg = compose(config_name=config_name)

        with open_dict(cfg):
            cfg["weights"] = self.weights_path

        # Cutie mutates/augments the config through this helper in the original code.
        _ = get_dataset_cfg(cfg)

        self.cutie = CUTIE(cfg).to(self.device).eval()
        model_weights = torch.load(self.weights_path, map_location=self.device)
        self.cutie.load_weights(model_weights)

        self.processor = InferenceCore(self.cutie, cfg=cfg)
        self._tracklet_object_dict: dict[int, int] = {}
        self._object_ids: list[int] = []
        self._initialized = False

    def reset(self) -> None:
        """Reset Cutie temporal memory and object mappings."""
        self.processor.clear_memory()
        self._tracklet_object_dict = {}
        self._object_ids = []
        self._initialized = False

    def initialize(
        self,
        frame: np.ndarray,
        mask_output: MaskOutput,
    ) -> None:
        """Initialize Cutie memory from masks on the given frame."""
        if mask_output.masks is None:
            self.reset()
            return

        if mask_output.masks.ndim != 3:
            raise ValueError(
                f"CutieMaskPropagator expects masks with shape (N, H, W). Got shape {mask_output.masks.shape}."
            )

        num_masks = mask_output.masks.shape[0]
        if len(mask_output.tracklet_mask_dict) != num_masks:
            raise ValueError(
                "Number of tracklet-mask mappings must match number of masks. "
                f"Got {len(mask_output.tracklet_mask_dict)} mappings for {num_masks} masks."
            )

        if num_masks == 0:
            self.reset()
            return

        expected_indices = list(range(num_masks))
        actual_indices = sorted(mask_output.tracklet_mask_dict.values())
        if actual_indices != expected_indices:
            raise ValueError(
                "MaskOutput tracklet_mask_dict must map tracklet IDs to local mask "
                f"indices {expected_indices}. Got {actual_indices}."
            )

        self._tracklet_object_dict = _build_tracklet_object_dict(mask_output.tracklet_mask_dict)

        # For the first masks (initializing Cutie), these will be 1,2,...,N
        # In case of adding new masks and removing the redundant ones, this list
        # will be evolving
        self._object_ids = [
            object_id
            for _, object_id in sorted(
                self._tracklet_object_dict.items(),
                key=lambda item: item[1],
            )
        ]

        frame_torch = _image_to_torch(frame, self.device)
        masks_torch = _binary_masks_to_non_overlapping_torch(mask_output.masks, self.device)

        with torch.inference_mode():
            with self._autocast_context():
                _ = self.processor.step(
                    frame_torch,
                    masks_torch,
                    objects=self._object_ids,
                    idx_mask=False,
                )

        self._initialized = True

    def propagate(
        self,
        frame: np.ndarray,
    ) -> MaskOutput | None:
        """Propagate initialized masks to the given frame."""
        if not self._initialized:
            return None

        frame_torch = _image_to_torch(frame, self.device)

        with torch.inference_mode():
            with self._autocast_context():
                prob = self.processor.step(frame_torch)

        indexed_mask = _torch_prob_to_numpy_mask(prob)

        masks = _indexed_mask_to_binary_masks(
            indexed_mask=indexed_mask,
            object_ids=self._object_ids,
        )
        mask_avg_prob_dict = _compute_mask_avg_prob_dict(
            prob=prob,
            object_ids=self._object_ids,
        )

        return MaskOutput(
            masks=masks,
            tracklet_mask_dict=self._tracklet_object_dict.copy(),
            mask_avg_prob_dict=mask_avg_prob_dict,
        )

    def _autocast_context(self) -> Any:
        if self.use_amp:
            if hasattr(torch, "amp"):
                return torch.amp.autocast("cuda", enabled=True)
            return torch.cuda.amp.autocast(enabled=True)
        return nullcontext()
