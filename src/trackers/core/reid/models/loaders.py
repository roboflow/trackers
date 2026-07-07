# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Checkpoint path resolution and state-dict loading."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

_HF_PREFIX = "hf://"
_GD_PREFIX = "gd://"
_FASTREID_SBS_ARCH = "fastreid_sbs_resnest50"

# State-dict key prefixes dropped before matching. Classification heads differ
# per dataset (num_classes) and are unused at inference — re-ID reads the
# pre-classifier embedding.
_DEFAULT_DROP_PREFIXES = ("classifier",)


@dataclass
class KeyReport:
    """Summary of a state-dict load (matched/missing/unexpected keys)."""

    matched: int
    total: int
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    @property
    def matched_fraction(self) -> float:
        return self.matched / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"loaded {self.matched}/{self.total} params "
            f"({self.matched_fraction:.0%}); "
            f"{len(self.missing)} missing, {len(self.unexpected)} unused"
        )


def resolve_weights(source: str) -> str:
    """Resolve a local path, ``hf://``, or ``gd://`` URL to a local weights file."""
    if source.startswith(_HF_PREFIX):
        return _resolve_hf(source)
    if source.startswith(_GD_PREFIX):
        return _resolve_gd(source)

    import os

    if not os.path.exists(source):
        raise FileNotFoundError(f"Weights file not found: {source}")
    return source


def _resolve_hf(source: str) -> str:
    from huggingface_hub import hf_hub_download

    rest = source[len(_HF_PREFIX) :]
    parts = rest.split("/")
    if len(parts) < 3:
        raise ValueError(f"Malformed hf:// weights URL {source!r}. Expected 'hf://<org>/<name>/<filename>'.")

    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])

    revision = None
    if "@" in repo_id:
        repo_id, revision = repo_id.split("@", 1)

    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def _resolve_gd(source: str) -> str:
    """Download a Google Drive file once and cache under ``~/.cache/trackers/weights``."""
    import os

    rest = source[len(_GD_PREFIX) :]
    file_id, _, filename = rest.partition("/")
    if not file_id:
        raise ValueError(f"Malformed gd:// weights URL {source!r}. Expected 'gd://<file_id>/<filename>'.")
    if not filename:
        filename = f"{file_id}.pth"

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "trackers", "weights")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        return path

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Google Drive weights (gd://...) require gdown. Install with:  pip install gdown"
        ) from exc

    gdown.download(id=file_id, output=path, quiet=False)
    return path


def remap_fastreid_sbs_state_dict(state_dict: dict) -> dict:
    """Map BoT-SORT / FastReID SBS checkpoint keys onto :class:`FastReIDSBSResNeSt50`.

    Renames FastReID ``heads.*`` keys to ``pool.*`` / ``bottleneck.*`` and passes
    ``backbone.*`` through unchanged. Skips ``heads.weight`` (classifier). GeM
    ``heads.pool_layer.p`` becomes ``pool.p`` and replaces the 3.0 init default.
    """
    mapped: dict = {}
    for key, value in state_dict.items():
        key = key[7:] if key.startswith("module.") else key
        if key.startswith("backbone."):
            mapped[key] = value
        elif key == "heads.pool_layer.p":
            mapped["pool.p"] = value  # GeM exponent; overwrites GeneralizedMeanPooling default
        elif key.startswith("heads.bottleneck.0."):
            mapped["bottleneck." + key[len("heads.bottleneck.0.") :]] = value
        # Skip heads.weight (identity classifier; unused at inference).
    return mapped


def load_fastreid_sbs_state_dict_into(
    module: nn.Module,
    path: str,
    device: torch.device,
    *,
    warn_threshold: float = 0.5,
) -> KeyReport:
    """Load a BoT-SORT / FastReID SBS ``.pth`` checkpoint into :class:`FastReIDSBSResNeSt50`."""
    state_dict = _read_state_dict(path, device)
    cleaned = remap_fastreid_sbs_state_dict(state_dict)

    target = module.state_dict()
    matched: dict = {}
    unexpected: list[str] = []
    for key, value in cleaned.items():
        if key in target and target[key].shape == value.shape:
            matched[key] = value
        else:
            unexpected.append(key)

    target.update(matched)
    module.load_state_dict(target)

    missing = [key for key in target if key not in matched]
    report = KeyReport(
        matched=len(matched),
        total=len(target),
        missing=missing,
        unexpected=unexpected,
    )

    if report.matched_fraction < warn_threshold:
        warnings.warn(
            f"Only {report.summary()} from {path!r}. The weights likely do not "
            f"match {_FASTREID_SBS_ARCH!r}.",
            UserWarning,
            stacklevel=2,
        )

    return report


def load_state_dict_for_architecture(
    module: nn.Module,
    path: str,
    device: torch.device,
    architecture: str,
    *,
    warn_threshold: float = 0.5,
) -> KeyReport:
    """Load *path* using the loader appropriate for *architecture*."""
    if architecture == _FASTREID_SBS_ARCH:
        return load_fastreid_sbs_state_dict_into(module, path, device, warn_threshold=warn_threshold)
    return load_state_dict_into(module, path, device, warn_threshold=warn_threshold)


def _read_state_dict(path: str, device: torch.device) -> dict:
    """Read a raw state dict from a ``.pth`` or ``.safetensors`` file."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device=str(device))

    import torch

    state_dict = torch.load(path, map_location=device, weights_only=False)
    # Some checkpoints wrap the tensors in {"state_dict": ...} (or "model").
    for wrapper_key in ("state_dict", "model"):
        if isinstance(state_dict, dict) and wrapper_key in state_dict:
            state_dict = state_dict[wrapper_key]
            break
    return state_dict


def load_state_dict_into(
    module: nn.Module,
    path: str,
    device: torch.device,
    *,
    drop_prefixes: tuple[str, ...] = _DEFAULT_DROP_PREFIXES,
    warn_threshold: float = 0.5,
) -> KeyReport:
    """Load *path* into *module* by name and shape (classifier keys skipped)."""
    state_dict = _read_state_dict(path, device)

    cleaned: dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        if any(key.startswith(p) for p in drop_prefixes):
            continue
        cleaned[key] = v

    target = module.state_dict()
    matched: dict = {}
    unexpected: list[str] = []
    for k, v in cleaned.items():
        if k in target and target[k].shape == v.shape:
            matched[k] = v
        else:
            unexpected.append(k)

    target.update(matched)
    module.load_state_dict(target)

    missing = [k for k in target if k not in matched]
    report = KeyReport(
        matched=len(matched),
        total=len(target),
        missing=missing,
        unexpected=unexpected,
    )

    if report.matched_fraction < warn_threshold:
        warnings.warn(
            f"Only {report.summary()} from {path!r}. The weights likely do not "
            f"match this architecture — check the model/weights pairing.",
            UserWarning,
            stacklevel=2,
        )

    return report
