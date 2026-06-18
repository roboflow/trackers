# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Weight sourcing and architecture-agnostic loading for re-ID backbones.

This module decouples *where weights come from* from *what architecture they
are loaded into*. A weight source is a single string:

- ``"/path/to/weights.pth"`` (or ``.safetensors``) — a local file;
- ``"hf://<repo_id>/<filename>"`` — a file on the Hugging Face Hub, with an
  optional ``"@<revision>"`` suffix on the repo id, e.g.
  ``"hf://org/model@main/weights.safetensors"``.

Loading is name + shape matched against the target module and produces a
:class:`KeyReport`, so a mismatched architecture/weights pairing fails *loudly*
(a near-empty match is an error, not a silent accuracy regression).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

_HF_PREFIX = "hf://"

# State-dict key prefixes dropped before matching. Classification heads differ
# per dataset (num_classes) and are unused at inference — re-ID reads the
# pre-classifier embedding.
_DEFAULT_DROP_PREFIXES = ("classifier",)


@dataclass
class KeyReport:
    """Summary of a state-dict load, for transparency and loud failures.

    Attributes:
        matched: Number of target parameters filled from the checkpoint.
        total: Total number of target parameters.
        missing: Target keys that were *not* found in the checkpoint.
        unexpected: Checkpoint keys that did not map to any target parameter
            (after prefix stripping/dropping), including shape mismatches.
    """

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
    """Resolve a weight source string to a local filesystem path.

    Args:
        source: A local path or an ``"hf://<repo_id>/<filename>"`` URL (with an
            optional ``"@<revision>"`` on the repo id).

    Returns:
        Absolute local path to the weights file.

    Raises:
        FileNotFoundError: If a local path does not exist.
        ValueError: If an ``hf://`` URL is malformed.
    """
    if source.startswith(_HF_PREFIX):
        return _resolve_hf(source)

    import os

    if not os.path.exists(source):
        raise FileNotFoundError(f"Weights file not found: {source}")
    return source


def _resolve_hf(source: str) -> str:
    from huggingface_hub import hf_hub_download

    rest = source[len(_HF_PREFIX):]
    parts = rest.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Malformed hf:// weights URL {source!r}. "
            f"Expected 'hf://<org>/<name>/<filename>'."
        )

    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])

    revision = None
    if "@" in repo_id:
        repo_id, revision = repo_id.split("@", 1)

    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


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
    """Load weights from *path* into *module*, matching by name and shape.

    Keys are matched against the module's own ``state_dict``; only entries whose
    name **and** shape agree are loaded. ``module.`` prefixes (DataParallel) are
    stripped and *drop_prefixes* (classification heads by default) are removed.

    Args:
        module: Target module to load weights into.
        path: Local path to a ``.pth`` / ``.safetensors`` checkpoint.
        device: Device tensors are mapped to during load.
        drop_prefixes: Source key prefixes to discard before matching.
        warn_threshold: Emit a :class:`UserWarning` if the matched fraction
            falls below this value — a strong signal that the weights do not
            belong to this architecture.

    Returns:
        A :class:`KeyReport` describing what was loaded.
    """
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
