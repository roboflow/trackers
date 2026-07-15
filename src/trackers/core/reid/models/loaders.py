# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Checkpoint path resolution and state-dict loading."""

from __future__ import annotations

import os
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from safetensors.torch import load_file

from trackers.core.reid.architectures import checkpoint_remap_for_architecture

_HF_PREFIX = "hf://"
_GD_PREFIX = "gd://"

_DEFAULT_DROP_PREFIXES = ("classifier",)
_COMMON_KEY_PREFIXES = ("module.", "model.", "encoder.")


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

    if not os.path.exists(source):
        raise FileNotFoundError(f"Weights file not found: {source}")
    return source


def _resolve_hf(source: str) -> str:
    rest = source[len(_HF_PREFIX) :]
    parts = rest.split("/")
    if len(parts) < 3:
        raise ValueError(f"Malformed hf:// weights URL {source!r}. Expected 'hf://<org>/<name>/<filename>'.")

    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])

    revision = None
    if "@" in repo_id:
        repo_id, revision = repo_id.split("@", 1)

    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    except (HfHubHTTPError, EntryNotFoundError, OSError) as exc:
        raise RuntimeError(
            f"Failed to download Hugging Face weights from {source!r} (repo_id={repo_id!r}, filename={filename!r})."
        ) from exc


def _validate_downloaded_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Downloaded weights file is missing: {path}")
    if os.path.getsize(path) < 1:
        raise OSError(f"Downloaded weights file is empty: {path}")
    return path


def _resolve_gd(source: str) -> str:
    """Download a Google Drive file once and cache under ``~/.cache/trackers/weights``."""
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
        return _validate_downloaded_file(path)

    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Google Drive weights (gd://...) require gdown. Install with: pip install gdown") from exc

    fd, tmp_path = tempfile.mkstemp(prefix=f"{filename}.", suffix=".part", dir=cache_dir)
    os.close(fd)
    try:
        gdown.download(id=file_id, output=tmp_path, quiet=False)
        _validate_downloaded_file(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return _validate_downloaded_file(path)


def load_state_dict_for_architecture(
    module: nn.Module,
    path: str,
    device: torch.device,
    architecture: str,
    *,
    warn_threshold: float = 0.5,
    required_match_fraction: float | None = None,
) -> KeyReport:
    """Load *path* using the loader appropriate for *architecture*."""
    remap = checkpoint_remap_for_architecture(architecture)
    report = load_state_dict_into(
        module,
        path,
        device,
        remap=remap,
        warn_threshold=warn_threshold,
    )
    if required_match_fraction is not None and report.matched_fraction < required_match_fraction:
        raise ValueError(
            f"Checkpoint {path!r} matched only {report.matched_fraction:.0%} of "
            f"{architecture!r} parameters (required >= {required_match_fraction:.0%}). "
            f"{report.summary()}"
        )
    return report


def _strip_common_prefixes(key: str) -> str:
    stripped = key
    changed = True
    while changed:
        changed = False
        for prefix in _COMMON_KEY_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                changed = True
    return stripped


def _read_state_dict(path: str, device: torch.device) -> dict:
    """Read a raw state dict from a ``.pth`` or ``.safetensors`` file."""
    if path.endswith(".safetensors"):
        return load_file(path, device=str(device))

    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=device, weights_only=False)

    for wrapper_key in ("state_dict", "model", "encoder"):
        if isinstance(state_dict, dict) and wrapper_key in state_dict and isinstance(state_dict[wrapper_key], dict):
            state_dict = state_dict[wrapper_key]
            break
    return state_dict


def load_state_dict_into(
    module: nn.Module,
    path: str,
    device: torch.device,
    *,
    drop_prefixes: tuple[str, ...] = _DEFAULT_DROP_PREFIXES,
    remap: Callable[[dict], dict] | None = None,
    warn_threshold: float = 0.5,
) -> KeyReport:
    """Load *path* into *module* by name and shape (classifier keys skipped)."""
    state_dict = _read_state_dict(path, device)

    if remap is not None:
        cleaned = remap(state_dict)
    else:
        cleaned: dict = {}
        for k, v in state_dict.items():
            key = _strip_common_prefixes(k)
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
