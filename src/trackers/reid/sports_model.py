# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Prepare the pretrained sports re-ID model used by the original McByte++ for loading through Roboflow ReID.

The model is GTA-link's OSNet sports checkpoint, ``sports_model.pth.tar-60``:
https://github.com/sjc042/gta-link/blob/main/reid_checkpoints/sports_model.pth.tar-60

Preparation verifies the original checkpoint and preserves its inference tensors and preprocessing in a reusable cache.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_SOURCE_URL = "https://raw.githubusercontent.com/sjc042/gta-link/main/reid_checkpoints/sports_model.pth.tar-60"
_SOURCE_SHA256 = "8d5b2fd8763db34c2aad69810466adf413f0426d9f8119d322227e0e639c5fbd"
_SOURCE_SIZE = 30_393_613
_REID_VERSION = "0.1.0"
_INFERENCE_TENSOR_COUNT = 565
_EXCLUDED_KEYS = ("classifier.bias", "classifier.weight")
_PREPROCESSING = {
    "input_size": [256, 128],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "interpolation": "bilinear",
    "to_rgb": True,
    "resize_mode": "stretch",
    "pad_value": 114,
}


def _sha256(path: Path) -> str:
    """Hash a cached artifact without loading it as a model."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(data: bytes) -> None:
    """Reject every checkpoint except the known original before unpickling."""
    if len(data) != _SOURCE_SIZE:
        raise ValueError(f"Sports checkpoint size mismatch: expected {_SOURCE_SIZE} bytes, received {len(data)}.")
    if hashlib.sha256(data).hexdigest() != _SOURCE_SHA256:
        raise ValueError("Sports checkpoint SHA256 mismatch; only the verified original GTA-link file is supported.")


def _download_checkpoint() -> bytes:
    """Download the fixed source with a bounded size and verify its content."""
    import requests

    output = io.BytesIO()
    with requests.get(_SOURCE_URL, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if output.tell() + len(chunk) > _SOURCE_SIZE:
                raise ValueError("Sports checkpoint download exceeds the verified source size.")
            output.write(chunk)
    data = output.getvalue()
    _verify_source(data)
    return data


def _model_provenance() -> dict[str, Any]:
    """Describe the fixed source and conversion contract checked on cache reuse."""
    return {
        "schema_version": 1,
        "source_url": _SOURCE_URL,
        "source_sha256": _SOURCE_SHA256,
        "source_size": _SOURCE_SIZE,
        "algorithm_reference_commit": "be1bbc03f18e33e93e0a359bbcbfdfc4dc4ab6b9",
        "architecture": "osnet_x1_0",
        "source_tensor_count": _INFERENCE_TENSOR_COUNT + len(_EXCLUDED_KEYS),
        "exact_inference_tensor_count": _INFERENCE_TENSOR_COUNT,
        "excluded_training_head_keys": list(_EXCLUDED_KEYS),
        "preprocessing": _PREPROCESSING,
        "safe_loading": "weights_only=True with four explicit known NumPy globals",
    }


def _verify_cache(directory: Path) -> None:
    """Require matching provenance and intact model and preprocessing artifacts."""
    try:
        provenance = json.loads((directory / "provenance.json").read_text())
        config = json.loads((directory / "reid_config.json").read_text())
        expected_config = {"architecture": "osnet_x1_0", "preprocessing": _PREPROCESSING}
        if not isinstance(provenance, dict) or config != expected_config:
            raise ValueError("model configuration differs from the verified sports model")
        if any(provenance.get(key) != value for key, value in _model_provenance().items()):
            raise ValueError("source or conversion provenance does not match")
        packages = provenance.get("packages")
        if not isinstance(packages, dict) or packages.get("reid") != _REID_VERSION:
            raise ValueError("conversion package version does not match")
        for filename, key in (("weights.safetensors", "converted_sha256"), ("reid_config.json", "config_sha256")):
            if _sha256(directory / filename) != provenance.get(key):
                raise ValueError(f"{filename} SHA256 mismatch")
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Invalid sports ReID cache at {directory}: {exc}. Remove this directory to prepare it again."
        ) from exc


def _convert_checkpoint(data: bytes | None, directory: Path) -> None:
    """Safely convert verified tensors using the supported ReID package version."""
    try:
        import numpy as np
        import torch
        from numpy._core.multiarray import scalar as numpy_scalar
        from reid import ReIDModel
        from reid.preprocessing import ReIDPreprocessing
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("Sports ReID preparation requires the optional extra: pip install 'trackers[reid]'.") from exc

    reid_version = importlib.metadata.version("reid")
    if reid_version != _REID_VERSION:
        raise ValueError(f"Sports conversion requires reid=={_REID_VERSION}; found {reid_version}.")
    if data is None:
        data = _download_checkpoint()

    # These four globals are needed only by the verified file's NumPy training
    # metadata. Keep weights_only=True; never fall back to unrestricted pickle.
    allowed = [
        (numpy_scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        np.dtypes.Float32DType,
        np.dtypes.Float64DType,
    ]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    weights = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(weights, dict) or len(weights) != _INFERENCE_TENSOR_COUNT + len(_EXCLUDED_KEYS):
        raise ValueError("Sports checkpoint does not contain the expected inference and classifier tensors.")
    if not all(key in weights for key in _EXCLUDED_KEYS):
        raise ValueError("Sports checkpoint is missing the expected classifier tensors.")
    inference = {key: value for key, value in weights.items() if key not in _EXCLUDED_KEYS}
    model = ReIDModel("osnet_x1_0", preprocessing=ReIDPreprocessing.from_dict(_PREPROCESSING), device="cpu")
    # ReID 0.1.0 has no public tensor-state import API; this narrow, version-checked
    # conversion uses its backbone once and publishes its supported save format.
    model._backbone.load_state_dict(inference, strict=True)
    model.save(str(directory))
    converted = load_file(str(directory / "weights.safetensors"), device="cpu")
    if converted.keys() != inference.keys():
        raise ValueError("Converted sports model has different inference tensor keys.")
    for key, tensor in converted.items():
        if tensor.dtype != inference[key].dtype or not torch.equal(tensor, inference[key]):
            raise ValueError(f"Converted sports model changed inference tensor {key}.")

    provenance = _model_provenance()
    provenance.update(
        packages={
            name: importlib.metadata.version(name)
            for name in ("reid", "torch", "torchvision", "numpy", "supervision", "safetensors", "Pillow")
        },
        converted_sha256=_sha256(directory / "weights.safetensors"),
        config_sha256=_sha256(directory / "reid_config.json"),
    )
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def prepare_sports_reid_model(checkpoint: str | Path | None = None, *, cache_dir: str | Path | None = None) -> Path:
    """Prepare the original McByte++ sports OSNet model for the shared ReID loader.

    Only the exact GTA-link sports checkpoint is accepted. Its 565 inference
    tensors and RGB preprocessing at 256 pixels high and 128 pixels wide are
    preserved in a model directory containing ``weights.safetensors``,
    ``reid_config.json`` and provenance.
    Conversion requires ``trackers[reid]`` with ``reid==0.1.0`` and uses CPU
    without changing thread or precision settings. A valid cache needs no
    optional imports or network access.

    Args:
        checkpoint: Local original checkpoint. If omitted and no prepared cache
            exists, download the fixed GitHub source and verify its SHA256 before
            safe tensor loading. A valid prepared cache avoids this download.
        cache_dir: Cache root. Defaults to ``$XDG_CACHE_HOME/trackers/reid``
            or ``~/.cache/trackers/reid`` when that variable is unset.

    Returns:
        Path to the verified model directory. Pass ``str(path)`` as a tracker's
        ``reid_model`` or as ``source`` to ``ReIDModel.from_pretrained``.

    Raises:
        ValueError: If the checkpoint, conversion, package version or cache
            fails verification. A corrupt cache must be removed explicitly.
        ImportError: If the optional conversion dependencies are unavailable.
        OSError: If local files cannot be read or written.
    """
    data = None
    if checkpoint is not None:
        with Path(checkpoint).expanduser().open("rb") as source:
            data = source.read(_SOURCE_SIZE + 1)
        _verify_source(data)

    if cache_dir is None:
        root = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache") / "trackers" / "reid"
    else:
        root = Path(cache_dir)
    root = root.expanduser().resolve()
    directory = root / f"sports-osnet-{_SOURCE_SHA256[:12]}-reid-{_REID_VERSION}-v1"
    if directory.exists():
        _verify_cache(directory)
        return directory

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sports-", dir=root) as temporary:
        staged = Path(temporary) / "model"
        _convert_checkpoint(data, staged)
        _verify_cache(staged)
        try:
            staged.rename(directory)
        except OSError:
            # Another process may have completed the same conversion first.
            if not directory.exists():
                raise
            _verify_cache(directory)
    return directory
