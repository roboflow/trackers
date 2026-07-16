# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Dataset, PK sampler, and train transforms for re-ID fine-tuning."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from torch.utils.data import Dataset, Sampler

from trackers.core.reid.models.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    REID_INPUT_SIZE,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def build_train_transform(
    input_size: tuple[int, int] = REID_INPUT_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
):
    """Training augmentations matching inference geometry (resize, flip, jitter, erase)."""
    from torchvision.transforms import (
        ColorJitter,
        Compose,
        InterpolationMode,
        Normalize,
        RandomErasing,
        RandomHorizontalFlip,
        Resize,
        ToTensor,
    )

    return Compose(
        [
            Resize(input_size, interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.0),
            ToTensor(),
            Normalize(mean=list(mean), std=list(std)),
            RandomErasing(p=0.5, value="random"),
        ]
    )


class ReIDCropDataset(Dataset):
    """Identity-folder crop dataset (``<root>/<identity>/<crop>.jpg``)."""

    def __init__(
        self,
        root: str | Path,
        transform=None,
        include_identities: list[str] | None = None,
    ) -> None:
        """Scan ``root`` and map each identity folder to a contiguous label.

        Args:
            root: Dataset root with one subdirectory per identity.
            transform: Optional image transform (defaults to :func:`build_train_transform`).
            include_identities: Optional identity folder whitelist.
        """
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Crop dataset root not found: {self.root}")

        self.transform = transform if transform is not None else build_train_transform()

        allowed = set(include_identities) if include_identities is not None else None
        identities: list[str] = []
        samples: list[tuple[Path, int]] = []
        for identity_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if allowed is not None and identity_dir.name not in allowed:
                continue
            images = sorted(p for p in identity_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
            if not images:
                continue
            label = len(identities)
            identities.append(identity_dir.name)
            samples.extend((image_path, label) for image_path in images)

        if not samples:
            raise FileNotFoundError(
                f"No identity crops found under {self.root}. Expected <root>/<identity>/<crop>.jpg."
            )

        self.identities = identities
        self.samples = samples
        self.labels = [label for _, label in samples]

    @property
    def num_classes(self) -> int:
        """Number of distinct identities."""
        return len(self.identities)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


class PKSampler(Sampler[list[int]]):
    """Sample ``P`` identities x ``K`` instances per batch for triplet mining."""

    def __init__(
        self,
        labels: list[int],
        p: int,
        k: int,
        seed: int | None = None,
    ) -> None:
        """Initialise the PK sampler.

        Args:
            labels: Per-sample identity labels.
            p: Identities per batch.
            k: Instances per identity per batch.
            seed: Optional shuffle seed.
        """
        if p <= 0 or k <= 0:
            raise ValueError(f"p and k must be positive, got p={p}, k={k}.")

        self.p = p
        self.k = k
        self.batch_size = p * k
        self.seed = seed
        self._epoch = 0

        self._indices_by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self._indices_by_label[label].append(index)
        self._labels = list(self._indices_by_label)

        if len(self._labels) < p:
            raise ValueError(f"PKSampler needs at least p={p} identities, got {len(self._labels)}.")

        self._num_batches = len(self._labels) // p

    def __len__(self) -> int:
        return self._num_batches

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so shuffling differs across epochs (deterministically)."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(None if self.seed is None else self.seed + self._epoch)  # noqa: S311
        self._epoch += 1

        per_label_pool: dict[int, list[int]] = {}
        for label, indices in self._indices_by_label.items():
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            per_label_pool[label] = shuffled

        labels = self._labels.copy()
        rng.shuffle(labels)

        cursor = {label: 0 for label in labels}
        for batch_start in range(self._num_batches):
            chosen = labels[batch_start * self.p : (batch_start + 1) * self.p]
            batch: list[int] = []
            for label in chosen:
                pool = per_label_pool[label]
                picks: list[int] = []
                for _ in range(self.k):
                    if cursor[label] >= len(pool):
                        rng.shuffle(pool)
                        cursor[label] = 0
                    picks.append(pool[cursor[label]])
                    cursor[label] += 1
                batch.extend(picks)
            yield batch
