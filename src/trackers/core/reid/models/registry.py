# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Curated ReID recipes: aliases, ``ModelCard``, and ``reid_config.json``.

This module answers *which pretrained model to load*, not *how to build a
backbone*.

- ``ALIASES`` maps short names (for example
  ``osnet_x1_0_msmt17_combineall``) to a :class:`ModelCard`.
- A :class:`ModelCard` holds the four load axes: architecture name, weights
  source (``hf://`` / ``gd://`` / local path), :class:`ReIDPreprocessing`, and
  an optional domain warning.
- :func:`resolve_model_card` looks up an alias, a local
  ``save_pretrained`` directory, or an HF repo that ships ``reid_config.json``.
- :func:`save_model_config` / :func:`load_model_config` round-trip the card for
  self-describing checkpoints.

Architecture construction lives in ``trackers.core.reid.architectures``.
Checkpoint I/O lives in ``trackers.core.reid.models.loaders``.
``ReIDModel.from_pretrained`` ties those pieces together.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

from trackers.core.reid.architectures import list_architectures
from trackers.core.reid.models.preprocessing import ReIDPreprocessing

# OSNet x1.0 trained on MSMT17 with combineall (train + query + gallery).
# Produces the strongest general-purpose pedestrian features, which is why it
# is the library default — but because it trains on the MSMT17 test identities
# it MUST NOT be used to benchmark MSMT17.
_DEFAULT_OSNET_WEIGHTS = (
    "hf://kaiyangzhou/osnet/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10"
    "_softmax_labelsmooth_flip_jitter.pth"
)

_DOMAIN_WARNING = (
    "The default ReIDModel weights were trained on MSMT17 pedestrian images. "
    "Performance may degrade significantly on non-person objects (vehicles, "
    "animals, products, etc.). Pass a domain-specific checkpoint via "
    "`ReIDModel.from_pretrained(source, architecture=...)` for other use cases."
)

# ---------------------------------------------------------------------------
# Public constants and types
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "osnet_x1_0_msmt17_combineall"
"""Curated alias used when :meth:`ReIDModel.from_pretrained` gets no ``source``."""


# Default preprocessing for registered architectures (bare `.pth` loads, no ModelCard).
DEFAULT_ARCHITECTURE_PREPROCESSING = ReIDPreprocessing()


def _build_architecture_default_preprocessing() -> dict[str, ReIDPreprocessing]:
    return {name: DEFAULT_ARCHITECTURE_PREPROCESSING for name in list_architectures()}


ARCHITECTURE_DEFAULT_PREPROCESSING = _build_architecture_default_preprocessing()


def default_preprocessing_for_architecture(architecture: str | None) -> ReIDPreprocessing:
    """Return default preprocessing for a named architecture (bare weight loads)."""
    if architecture is None:
        return DEFAULT_ARCHITECTURE_PREPROCESSING
    return ARCHITECTURE_DEFAULT_PREPROCESSING.get(architecture, DEFAULT_ARCHITECTURE_PREPROCESSING)


@dataclass
class ModelCard:
    """One pretrained ReID recipe (architecture + weights + preprocessing).

    Produced by curated :data:`ALIASES` entries or by loading
    ``reid_config.json`` from a ``save_pretrained`` directory / HF repo.
    """

    architecture: str
    weights: str | None
    preprocessing: ReIDPreprocessing
    domain_warning: str | None = None


ALIASES: dict[str, ModelCard] = {
    DEFAULT_MODEL: ModelCard(
        architecture="osnet_x1_0",
        weights=_DEFAULT_OSNET_WEIGHTS,
        preprocessing=DEFAULT_ARCHITECTURE_PREPROCESSING,
        domain_warning=_DOMAIN_WARNING,
    ),
}

# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_model_card(source: str) -> ModelCard | None:
    """Return a card for a curated alias or a directory/repo with ``reid_config.json``."""
    if source in ALIASES:
        return ALIASES[source]

    # Local directory with a reid_config.json (save_pretrained output).
    if os.path.isdir(source):
        config_path = os.path.join(source, "reid_config.json")
        if os.path.exists(config_path):
            return load_model_config(source)
        return None

    # HF repo URL (no trailing filename) — try to fetch reid_config.json.
    # hf://org/repo has exactly 2 non-empty parts after "hf://".
    # hf://org/repo/file.pth has 3+ parts and is a bare weights file.
    if source.startswith("hf://"):
        rest = source[len("hf://") :]
        parts = [p for p in rest.split("/") if p]
        if len(parts) == 2:
            try:
                return _load_hf_repo_config(source)
            except (HfHubHTTPError, EntryNotFoundError, OSError, ValueError) as exc:
                raise RuntimeError(f"Failed to resolve Hugging Face repo config for {source!r}.") from exc

    return None


def load_model_config(directory_or_repo: str) -> ModelCard:
    """Load a :class:`ModelCard` from ``reid_config.json`` (and local ``weights.safetensors`` if present)."""
    if directory_or_repo.startswith("hf://"):
        return _load_hf_repo_config(directory_or_repo)

    config_path = os.path.join(directory_or_repo, "reid_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No reid_config.json found in {directory_or_repo!r}. "
            "Use ReIDModel.save_pretrained() to create a self-describing "
            "model directory."
        )
    weights_path = os.path.join(directory_or_repo, "weights.safetensors")
    weights_source = weights_path if os.path.exists(weights_path) else None
    return _parse_config_file(config_path, weights_source=weights_source)


def save_model_config(card: ModelCard, directory: str) -> None:
    """Write ``reid_config.json`` to *directory*."""
    if not isinstance(card.architecture, str):
        raise ValueError(
            "Cannot save a ModelCard whose architecture is an nn.Module. "
            "Use a named architecture string (e.g. 'osnet_x1_0', "
            "'timm:resnet50') to enable save_pretrained()."
        )

    config = {
        "architecture": card.architecture,
        "preprocessing": card.preprocessing.to_dict(),
    }
    config_path = os.path.join(directory, "reid_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_hf_repo_config(hf_repo_url: str) -> ModelCard:
    """Download and parse ``reid_config.json`` from an HF repo URL."""
    rest = hf_repo_url[len("hf://") :]
    parts = [p for p in rest.split("/") if p]
    repo_id = "/".join(parts[:2])
    revision = None
    if "@" in repo_id:
        repo_id, revision = repo_id.split("@", 1)

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="reid_config.json",
        revision=revision,
    )
    weights_source = hf_repo_url.rstrip("/") + "/weights.safetensors"
    return _parse_config_file(config_path, weights_source=weights_source)


def _parse_config_file(config_path: str, *, weights_source: str | None) -> ModelCard:
    """Parse a ``reid_config.json`` file into a :class:`ModelCard`."""
    with open(config_path) as f:
        data = json.load(f)

    architecture = data["architecture"]
    preprocessing = ReIDPreprocessing.from_dict(data.get("preprocessing", {}))
    return ModelCard(
        architecture=architecture,
        weights=weights_source,
        preprocessing=preprocessing,
    )
