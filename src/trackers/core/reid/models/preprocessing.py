# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Re-ID crop and embedding preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# ImageNet statistics — shared by every OSNet / timm ImageNet-pretrained backbone.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Standard person re-ID input geometry (height, width). OSNet and most backbones
# train at 256×128; BoT-SORT FastReID SBS uses 384×128 (see registry overrides).
REID_INPUT_SIZE = (256, 128)

# Gray padding used by YOLO / FastReID letterbox helpers (BoT-SORT ``preprocess()``).
FASTREID_PAD_VALUE = 114

ResizeMode = Literal["stretch", "letterbox"]


@dataclass(frozen=True)
class ReIDPreprocessing:
    """Crop resize/normalisation and optional L2-normalised embeddings."""

    input_size: tuple[int, int] = REID_INPUT_SIZE
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    interpolation: str = "bilinear"
    to_rgb: bool = True
    normalize_embeddings: bool = True
    resize_mode: ResizeMode = "stretch"
    pad_value: int = FASTREID_PAD_VALUE

    def describe(self) -> str:
        """Human-readable one-line summary."""
        h, w = self.input_size
        colour = "BGR→RGB" if self.to_rgb else "RGB (no swap)"
        norm = "L2" if self.normalize_embeddings else "none"
        return (
            f"ReIDPreprocessing(resize={h}x{w} [{self.resize_mode}, {self.interpolation}], "
            f"{colour}, mean={self.mean}, std={self.std}, embed_norm={norm})"
        )

    def resize_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize an ``H×W×C`` crop to :attr:`input_size` using OpenCV.

        ``stretch`` matches BoT-SORT ``FastReIDInterface`` inference
        (``cv2.resize`` to ``SIZE_TEST``, aspect ratio not preserved).
        ``letterbox`` is optional (aspect-preserving pad); BoT-SORT leaves it
        commented out upstream, so registered FastReID defaults use stretch.
        """
        import cv2

        if crop.size == 0:
            target_h, target_w = self.input_size
            channels = crop.shape[2] if crop.ndim == 3 else 1
            shape = (target_h, target_w, channels) if crop.ndim == 3 else (target_h, target_w)
            return np.zeros(shape, dtype=crop.dtype)

        target_h, target_w = self.input_size
        interpolation = _opencv_interpolation(self.interpolation)

        if self.resize_mode == "letterbox":
            height, width = crop.shape[:2]
            scale = min(target_h / height, target_w / width)
            new_w = max(int(round(width * scale)), 1)
            new_h = max(int(round(height * scale)), 1)
            resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
            if crop.ndim == 3:
                padded = np.full((target_h, target_w, crop.shape[2]), self.pad_value, dtype=crop.dtype)
            else:
                padded = np.full((target_h, target_w), self.pad_value, dtype=crop.dtype)
            padded[:new_h, :new_w] = resized
            return padded

        return cv2.resize(crop, (target_w, target_h), interpolation=interpolation)

    def build_transform(self):
        """Tensor normalisation for a crop already at :attr:`input_size` (RGB PIL in)."""
        from torchvision.transforms import Compose, Normalize, ToTensor

        return Compose(
            [
                ToTensor(),
                Normalize(mean=list(self.mean), std=list(self.std)),
            ]
        )

    def to_dict(self) -> dict:
        """Serialise for ``reid_config.json``."""
        return {
            "input_size": list(self.input_size),
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": self.interpolation,
            "to_rgb": self.to_rgb,
            "normalize_embeddings": self.normalize_embeddings,
            "resize_mode": self.resize_mode,
            "pad_value": self.pad_value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReIDPreprocessing:
        """Reconstruct from a ``reid_config.json`` preprocessing dict."""
        kwargs: dict = {}
        if "input_size" in data:
            kwargs["input_size"] = tuple(data["input_size"])
        if "mean" in data:
            kwargs["mean"] = tuple(data["mean"])
        if "std" in data:
            kwargs["std"] = tuple(data["std"])
        if "interpolation" in data:
            kwargs["interpolation"] = data["interpolation"]
        if "to_rgb" in data:
            kwargs["to_rgb"] = data["to_rgb"]
        if "normalize_embeddings" in data:
            kwargs["normalize_embeddings"] = data["normalize_embeddings"]
        if "resize_mode" in data:
            kwargs["resize_mode"] = data["resize_mode"]
        if "pad_value" in data:
            kwargs["pad_value"] = data["pad_value"]
        return cls(**kwargs)


def _opencv_interpolation(name: str) -> int:
    import cv2

    modes = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
    }
    if name not in modes:
        raise ValueError(f"Unknown interpolation {name!r}. Choose from: {sorted(modes)}")
    return modes[name]
