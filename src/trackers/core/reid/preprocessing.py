# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Re-ID crop and embedding preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

# ImageNet statistics — shared by every OSNet / timm ImageNet-pretrained backbone.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Standard person re-ID input geometry (height, width). OSNet, BoT, FastReID, …
# all train at 256x128; this is intentionally portrait, not square.
REID_INPUT_SIZE = (256, 128)


@dataclass(frozen=True)
class ReIDPreprocessing:
    """Crop resize/normalisation and optional L2-normalised embeddings."""

    input_size: tuple[int, int] = REID_INPUT_SIZE
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    interpolation: str = "bilinear"
    to_rgb: bool = True
    normalize_embeddings: bool = True

    def describe(self) -> str:
        """Human-readable one-line summary."""
        h, w = self.input_size
        colour = "BGR→RGB" if self.to_rgb else "RGB (no swap)"
        norm = "L2" if self.normalize_embeddings else "none"
        return (
            f"ReIDPreprocessing(resize={h}x{w} [{self.interpolation}], {colour}, "
            f"mean={self.mean}, std={self.std}, embed_norm={norm})"
        )

    def build_transform(self):
        """Torchvision resize → tensor → normalise (RGB PIL image in)."""
        from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

        modes = {
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
            "nearest": InterpolationMode.NEAREST,
        }
        if self.interpolation not in modes:
            raise ValueError(
                f"Unknown interpolation {self.interpolation!r}. "
                f"Choose from: {sorted(modes)}"
            )

        return Compose([
            Resize(self.input_size, interpolation=modes[self.interpolation]),
            ToTensor(),
            Normalize(mean=list(self.mean), std=list(self.std)),
        ])

    def to_dict(self) -> dict:
        """Serialise for ``reid_config.json``."""
        return {
            "input_size": list(self.input_size),
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": self.interpolation,
            "to_rgb": self.to_rgb,
            "normalize_embeddings": self.normalize_embeddings,
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
        return cls(**kwargs)
