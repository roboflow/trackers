# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Model cards, the tiny curated alias map, and config (de)serialization.

A :class:`ModelCard` bundles the three independent axes that define a pretrained
re-ID identity — *architecture*, *weights*, and *preprocessing* — plus an
optional ``domain_warning`` for checkpoints that are domain-specific (e.g.
trained only on pedestrian images).

**Scaling is via self-describing repos, not this map.** Any model published with
a ``reid_config.json`` (exactly what
:meth:`~trackers.core.reid.model.ReIDModel.save_pretrained` writes) loads with
``from_pretrained("hf://org/repo")`` and **zero registration** —
:func:`resolve_model_card` reads the config straight from the directory / Hub
repo. Community and user-trained models are these self-describing repos; they do
**not** belong in :data:`ALIASES`.

:data:`ALIASES` is therefore a *deliberately tiny* curated layer with only two
jobs: (1) provide the no-arg default model, and (2) adapt external checkpoints
that lack a ``reid_config.json`` (a bare ``.pth`` cannot state its architecture
or preprocessing, so the card records that triple once). It is a convenience,
never a gate. The code-level extension point for new backbones is
:mod:`trackers.core.reid.architectures`, not this map.

All ``huggingface_hub`` imports are lazy so that importing this module does not
require network access or the optional ``[reid]`` extra at import time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from trackers.core.reid.preprocessing import ReIDPreprocessing

# ---------------------------------------------------------------------------
# Default checkpoint URL
# ---------------------------------------------------------------------------

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


@dataclass
class ModelCard:
    """A complete, overridable description of a pretrained re-ID identity.

    Attributes:
        architecture: Architecture selector string — a registered name such as
            ``"osnet_x1_0"``, a timm model as ``"timm:resnet50"``, etc. (see
            :func:`~trackers.core.reid.architectures.list_architectures`).
        weights: Weight source string — a local path or ``"hf://<repo>/<file>"``
            URL, or ``None`` if no external weights are needed.
        preprocessing: Explicit input/output preprocessing pipeline.
        domain_warning: Optional human-readable warning emitted (as a
            :class:`UserWarning`) when this card is used as the *default*
            checkpoint, flagging that it is domain-specific.
    """

    architecture: str
    weights: str | None
    preprocessing: ReIDPreprocessing
    domain_warning: str | None = None


ALIASES: dict[str, ModelCard] = {
    DEFAULT_MODEL: ModelCard(
        architecture="osnet_x1_0",
        weights=_DEFAULT_OSNET_WEIGHTS,
        preprocessing=ReIDPreprocessing(),
        domain_warning=_DOMAIN_WARNING,
    ),
}

# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_model_card(source: str) -> ModelCard | None:
    """Return a :class:`ModelCard` for an alias or a config-bearing source.

    Args:
        source: A curated alias name (e.g. ``"osnet_x1_0_msmt17_combineall"``),
            a local directory path that contains a ``reid_config.json``, or an
            ``"hf://org/repo"`` URL (no file suffix) whose repo holds
            ``reid_config.json``.

    Returns:
        A :class:`ModelCard` if the source is resolved, else ``None``
        (indicating the source is a bare weights file and requires an explicit
        ``architecture`` argument).
    """
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
        rest = source[len("hf://"):]
        parts = [p for p in rest.split("/") if p]
        if len(parts) == 2:
            try:
                return _load_hf_repo_config(source)
            except Exception:
                return None

    return None


def load_model_config(directory_or_repo: str) -> ModelCard:
    """Load a :class:`ModelCard` from a ``reid_config.json`` in a directory or HF repo.

    Also discovers ``weights.safetensors`` alongside the config file and sets
    the card's ``weights`` field accordingly.

    Args:
        directory_or_repo: Local directory path or ``"hf://org/repo"`` URL.

    Returns:
        A :class:`ModelCard` populated from the config file.

    Raises:
        FileNotFoundError: If no ``reid_config.json`` is found in the directory.
    """
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
    """Write a ``reid_config.json`` to *directory* from a :class:`ModelCard`.

    Args:
        card: The :class:`ModelCard` to serialize. ``card.architecture`` must
            be a string (not an ``nn.Module``).
        directory: Target directory (must exist).

    Raises:
        ValueError: If ``card.architecture`` is not a string.
    """
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
    from huggingface_hub import hf_hub_download

    rest = hf_repo_url[len("hf://"):]
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
