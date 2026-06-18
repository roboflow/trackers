# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Explicit, inspectable preprocessing for re-ID encoders.

Every transformation applied to a crop **before** it reaches the backbone — and
the single post-processing step applied to the embedding **after** — is declared
here in one place, as data, so it is never hidden inside the model code.

A :class:`ReIDPreprocessing` instance is attached to every
:class:`~trackers.core.reid.model.ReIDModel` and is logged on construction, so
the exact resize / colour-order / normalisation used for a given checkpoint is
always visible and reproducible. This matters because re-ID checkpoints are
sensitive to preprocessing: feeding the wrong input size or channel statistics
silently degrades retrieval quality without raising any error.

Pipeline (in order):

1. **colour order** — crops captured with OpenCV are BGR; ``to_rgb`` converts
   them to RGB (the order every ImageNet-pretrained backbone expects).
2. **resize** — to ``input_size`` ``(height, width)`` using ``interpolation``
   (exact resize, aspect ratio is **not** preserved — the re-ID convention).
3. **to tensor** — ``uint8 [0, 255]`` → ``float32 [0, 1]``, channels-first.
4. **normalise** — ``(x - mean) / std`` per channel.
5. **embedding normalisation** (post-backbone) — if ``normalize_embeddings`` is
   ``True`` the output vector is L2-normalised so cosine similarity equals the
   dot product. This is the only step applied *after* the backbone.
"""

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
    """Declarative description of a re-ID encoder's input/output processing.

    Attributes:
        input_size: Target crop size as ``(height, width)``. Resized exactly
            (aspect ratio not preserved), matching the re-ID convention.
        mean: Per-channel mean subtracted during normalisation (RGB order).
        std: Per-channel standard deviation used during normalisation.
        interpolation: Resize interpolation mode. One of ``"bilinear"``,
            ``"bicubic"``, or ``"nearest"``.
        to_rgb: If ``True`` (default), input crops are assumed to be **BGR**
            (OpenCV convention) and converted to RGB before the backbone.
        normalize_embeddings: If ``True`` (default), L2-normalise the output
            embedding so cosine similarity equals the dot product.
    """

    input_size: tuple[int, int] = REID_INPUT_SIZE
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    interpolation: str = "bilinear"
    to_rgb: bool = True
    normalize_embeddings: bool = True

    def describe(self) -> str:
        """Return a one-line, human-readable summary of the pipeline."""
        h, w = self.input_size
        colour = "BGR→RGB" if self.to_rgb else "RGB (no swap)"
        norm = "L2" if self.normalize_embeddings else "none"
        return (
            f"ReIDPreprocessing(resize={h}x{w} [{self.interpolation}], {colour}, "
            f"mean={self.mean}, std={self.std}, embed_norm={norm})"
        )

    def build_transform(self):
        """Build the torchvision transform mapping a PIL **RGB** image → tensor.

        Colour conversion (``to_rgb``) and embedding normalisation
        (``normalize_embeddings``) are applied by the model around this
        transform; this pipeline covers only the resize → tensor → normalise
        steps that operate on an already-RGB PIL image.

        Returns:
            A ``torchvision.transforms.Compose`` callable.
        """
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
        """Serialise to the ``reid_config.json`` preprocessing sub-schema.

        Returns:
            A plain-dict representation that ``from_dict`` can reconstruct.

        Examples:
            >>> ReIDPreprocessing().to_dict()["interpolation"]
            'bilinear'
        """
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
        """Reconstruct from a ``reid_config.json`` preprocessing sub-dict.

        Unknown keys are silently ignored; missing keys fall back to the
        class defaults, so partial configs work as overrides.

        Args:
            data: Dict produced by :meth:`to_dict` or read from JSON.

        Returns:
            A :class:`ReIDPreprocessing` instance.

        Examples:
            >>> p = ReIDPreprocessing()
            >>> ReIDPreprocessing.from_dict(p.to_dict()) == p
            True
        """
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
