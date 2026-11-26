# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Verified preparation of the original McByte++ sports encoder."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import requests
from numpy._core.multiarray import scalar as numpy_scalar

from trackers.reid import sports_model
from trackers.reid.sports_model import prepare_sports_reid_model


@pytest.fixture
def conversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Provide tiny tensors and fake optional packages for offline API tests."""
    data = b"verified miniature checkpoint"
    checkpoint = tmp_path / "original.pth"
    checkpoint.write_bytes(data)
    monkeypatch.setattr(sports_model, "_SOURCE_SIZE", len(data))
    monkeypatch.setattr(sports_model, "_SOURCE_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(sports_model, "_INFERENCE_TENSOR_COUNT", 3)
    monkeypatch.setattr(sports_model.importlib.metadata, "version", lambda name: "0.1.0" if name == "reid" else "test")
    state = {
        "conv1.weight": np.array([[1.25, -2.5]], dtype=np.float32),
        "bn1.running_mean": np.array([0.25, 0.75], dtype=np.float32),
        "bn1.num_batches_tracked": np.array(12, dtype=np.int64),
        "classifier.weight": np.array([[10.0, 20.0]], dtype=np.float32),
        "classifier.bias": np.array([0.5], dtype=np.float32),
    }
    calls: dict[str, Any] = {"loads": [], "allowlists": [], "models": []}

    @contextmanager
    def safe_globals(allowed: list[Any]):
        """Record the bounded allowlist used while loading the source."""
        calls["allowlists"].append(allowed)
        yield

    def load(source: io.BytesIO, *, map_location: str, weights_only: bool) -> dict[str, Any]:
        """Make the loader's CPU and safe-loading arguments observable."""
        calls["loads"].append({"data": source.read(), "map_location": map_location, "weights_only": weights_only})
        return {"state_dict": state}

    class FakePreprocessing:
        """Store the public preprocessing configuration serialized by ReID."""

        def __init__(self, values: dict[str, Any]) -> None:
            self.values = values

        @classmethod
        def from_dict(cls, values: dict[str, Any]) -> FakePreprocessing:
            return cls(values)

    class FakeBackbone:
        """Copy imported tensors so the save path exercises the loaded model."""

        def load_state_dict(self, tensors: dict[str, np.ndarray], *, strict: bool) -> None:
            assert strict is True
            self.state = {key: value.copy() for key, value in tensors.items()}

    class FakeModel:
        """Save tiny NumPy artifacts in place of the optional ReID package."""

        def __init__(self, architecture: str, *, preprocessing: FakePreprocessing, device: str) -> None:
            self.architecture = architecture
            self.preprocessing = preprocessing
            self.device = device
            self._backbone = FakeBackbone()
            calls["models"].append(self)

        def save(self, directory: str) -> None:
            root = Path(directory)
            root.mkdir()
            with (root / "weights.safetensors").open("wb") as handle:
                np.savez(handle, allow_pickle=False, **self._backbone.state)
            (root / "reid_config.json").write_text(
                json.dumps({"architecture": self.architecture, "preprocessing": self.preprocessing.values})
            )

    def load_file(path: str, *, device: str) -> dict[str, np.ndarray]:
        """Read tensors back through the mocked safetensors boundary."""
        assert device == "cpu"
        with np.load(path, allow_pickle=False) as saved:
            return {key: saved[key] for key in saved.files}

    torch = SimpleNamespace(load=load, equal=np.array_equal, serialization=SimpleNamespace(safe_globals=safe_globals))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "reid", SimpleNamespace(ReIDModel=FakeModel))
    monkeypatch.setitem(sys.modules, "reid.preprocessing", SimpleNamespace(ReIDPreprocessing=FakePreprocessing))
    monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(load_file=load_file))
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: pytest.fail("Unexpected network request"))
    return SimpleNamespace(checkpoint=checkpoint, data=data, state=state, calls=calls, torch=torch, model=FakeModel)


def test_prepare_sports_rejects_unrecognized_checkpoint(tmp_path: Path) -> None:
    """A local checkpoint must match the original before any conversion."""
    checkpoint = tmp_path / "sports.pth"
    checkpoint.write_bytes(b"not the original sports checkpoint")

    with pytest.raises(ValueError, match=r"SHA256|size"):
        prepare_sports_reid_model(checkpoint, cache_dir=tmp_path / "cache")

    assert not (tmp_path / "cache").exists()


def test_prepare_sports_preserves_inference_tensors_and_preprocessing(
    tmp_path: Path, conversion: SimpleNamespace
) -> None:
    """A user-supplied checkpoint becomes a reusable, documented model directory."""
    directory = prepare_sports_reid_model(conversion.checkpoint, cache_dir=tmp_path / "cache")

    assert directory.is_dir()
    assert {path.name for path in directory.iterdir()} == {
        "weights.safetensors",
        "reid_config.json",
        "provenance.json",
    }
    model = conversion.calls["models"][0]
    assert model.architecture == "osnet_x1_0"
    assert model.device == "cpu"
    expected_keys = {"conv1.weight", "bn1.running_mean", "bn1.num_batches_tracked"}
    with np.load(directory / "weights.safetensors", allow_pickle=False) as saved:
        assert set(saved.files) == expected_keys
        for key in expected_keys:
            np.testing.assert_array_equal(saved[key], conversion.state[key])
            assert saved[key].dtype == conversion.state[key].dtype
    preprocessing = json.loads((directory / "reid_config.json").read_text())["preprocessing"]
    expected_preprocessing = {
        "input_size": [256, 128],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "bilinear",
        "to_rgb": True,
        "resize_mode": "stretch",
        "pad_value": 114,
    }
    assert preprocessing == expected_preprocessing
    assert conversion.calls["loads"] == [{"data": conversion.data, "map_location": "cpu", "weights_only": True}]
    expected_allowed = {
        (numpy_scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        np.dtypes.Float32DType,
        np.dtypes.Float64DType,
    }
    assert set(conversion.calls["allowlists"][0]) == expected_allowed
    provenance = json.loads((directory / "provenance.json").read_text())
    assert provenance["source_sha256"] == hashlib.sha256(conversion.data).hexdigest()
    assert provenance["excluded_training_head_keys"] == ["classifier.bias", "classifier.weight"]
    assert provenance["packages"]["reid"] == "0.1.0"
    for filename, key in (("weights.safetensors", "converted_sha256"), ("reid_config.json", "config_sha256")):
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == provenance[key]


def test_prepare_sports_reuses_verified_cache_without_optional_packages(
    tmp_path: Path, conversion: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache hit neither imports optional packages nor downloads the source."""
    cache = tmp_path / "cache"
    directory = prepare_sports_reid_model(conversion.checkpoint, cache_dir=cache)
    monkeypatch.setitem(sys.modules, "reid", None)
    monkeypatch.setitem(sys.modules, "torch", None)

    assert prepare_sports_reid_model(cache_dir=cache) == directory
    assert len(conversion.calls["loads"]) == 1


@pytest.mark.parametrize("filename", ["weights.safetensors", "reid_config.json", "provenance.json"])
def test_prepare_sports_rejects_corrupt_cache(tmp_path: Path, conversion: SimpleNamespace, filename: str) -> None:
    """A changed artifact is never returned as a valid prepared model."""
    cache = tmp_path / "cache"
    directory = prepare_sports_reid_model(conversion.checkpoint, cache_dir=cache)
    with (directory / filename).open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(ValueError, match="Invalid sports ReID cache"):
        prepare_sports_reid_model(cache_dir=cache)
    assert len(conversion.calls["loads"]) == 1


@pytest.mark.parametrize("mutation", ["source", "package", "preprocessing", "missing"])
def test_prepare_sports_rejects_invalid_provenance(tmp_path: Path, conversion: SimpleNamespace, mutation: str) -> None:
    """Artifact hashes alone cannot validate a different conversion contract."""
    cache = tmp_path / "cache"
    directory = prepare_sports_reid_model(conversion.checkpoint, cache_dir=cache)
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if mutation == "source":
        provenance["source_sha256"] = "0" * 64
    elif mutation == "package":
        provenance["packages"]["reid"] = "0.2.0"
    elif mutation == "preprocessing":
        provenance["preprocessing"]["to_rgb"] = False
    provenance_path.write_text(json.dumps(provenance))
    if mutation == "missing":
        provenance_path.unlink()

    with pytest.raises(ValueError, match="Invalid sports ReID cache"):
        prepare_sports_reid_model(cache_dir=cache)


def test_prepare_sports_verifies_explicit_source_even_with_valid_cache(
    tmp_path: Path, conversion: SimpleNamespace
) -> None:
    """An existing cache cannot mask the caller passing an incorrect checkpoint."""
    cache = tmp_path / "cache"
    prepare_sports_reid_model(conversion.checkpoint, cache_dir=cache)
    conversion.checkpoint.write_bytes(b"x" * len(conversion.data))

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare_sports_reid_model(conversion.checkpoint, cache_dir=cache)


def test_prepare_sports_requires_optional_extra_for_conversion(
    tmp_path: Path, conversion: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing optional dependencies produce an actionable installation error."""
    monkeypatch.setitem(sys.modules, "reid", None)

    with pytest.raises(ImportError, match=r"trackers\[reid\]"):
        prepare_sports_reid_model(conversion.checkpoint, cache_dir=tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []


def test_prepare_sports_never_retries_unsafe_pickle_loading(
    tmp_path: Path, conversion: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A safe unpickler refusal aborts conversion rather than broadening trust."""
    attempts = []

    def reject(source: Any, **kwargs: Any) -> None:
        attempts.append(kwargs)
        raise pickle.UnpicklingError("Unexpected training metadata")

    monkeypatch.setattr(conversion.torch, "load", reject)
    with pytest.raises(pickle.UnpicklingError, match="Unexpected training metadata"):
        prepare_sports_reid_model(conversion.checkpoint, cache_dir=tmp_path / "cache")

    assert attempts == [{"map_location": "cpu", "weights_only": True}]
    assert list((tmp_path / "cache").iterdir()) == []


def test_prepare_sports_failed_save_does_not_publish_partial_cache(
    tmp_path: Path, conversion: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partially saved files stay private and are removed after conversion fails."""

    def fail_save(self: Any, directory: str) -> None:
        root = Path(directory)
        root.mkdir()
        (root / "weights.safetensors").write_bytes(b"incomplete")
        raise OSError("save interrupted")

    monkeypatch.setattr(conversion.model, "save", fail_save)
    with pytest.raises(OSError, match="save interrupted"):
        prepare_sports_reid_model(conversion.checkpoint, cache_dir=tmp_path / "cache")

    assert list((tmp_path / "cache").iterdir()) == []


@pytest.mark.parametrize("corrupt_download", [False, True])
def test_prepare_sports_downloads_verified_fixed_source_into_default_cache(
    tmp_path: Path, conversion: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, corrupt_download: bool
) -> None:
    """Default preparation downloads only the fixed URL and checks its digest."""

    class Response:
        """Small streamed response containing either the expected or wrong bytes."""

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, *, chunk_size: int):
            assert chunk_size > 0
            yield b"x" * len(conversion.data) if corrupt_download else conversion.data

    def get(url: str, **kwargs: Any) -> Response:
        assert url == "https://raw.githubusercontent.com/sjc042/gta-link/main/reid_checkpoints/sports_model.pth.tar-60"
        assert kwargs["stream"] is True
        timeout = kwargs["timeout"]
        assert np.isscalar(timeout) or (isinstance(timeout, tuple) and len(timeout) == 2)
        values = np.asarray(timeout, dtype=float)
        assert np.isfinite(values).all()
        assert (values > 0).all()
        return Response()

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "user-cache"))
    cache = tmp_path / "user-cache" / "trackers" / "reid"
    if corrupt_download:
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            prepare_sports_reid_model()
        assert list(cache.iterdir()) == []
        assert conversion.calls["loads"] == []
    else:
        directory = prepare_sports_reid_model()
        assert directory.parent == cache
        assert len(conversion.calls["loads"]) == 1
