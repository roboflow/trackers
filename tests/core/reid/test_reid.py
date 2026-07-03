# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Minimal test suite for the re-ID model loading refactor (RFC 0002).

Tests are grouped by the module they exercise:
  - preprocessing (ReIDPreprocessing)
  - registry (resolve_model_card, load_model_config, save_model_config)
  - loaders (resolve_weights, load_state_dict_into)
  - model smoke (ReIDModel.from_pretrained, extract_features)  [torch required]
  - round-trip (save_pretrained → from_pretrained)              [torch required]

torch/timm-dependent tests are guarded with pytest.importorskip.
No network downloads are performed.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from trackers.core.reid.models.loaders import resolve_weights
from trackers.core.reid.models.preprocessing import ReIDPreprocessing
from trackers.core.reid.models.registry import (
    DEFAULT_MODEL,
    FASTREID_MOT17_SBS50,
    resolve_model_card,
)

# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------


class TestReIDPreprocessing:
    def test_build_transform_callable(self) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("torchvision")
        t = ReIDPreprocessing().build_transform()
        from PIL import Image

        img = Image.new("RGB", (64, 128))
        out = t(img)
        assert out.shape == (3, 256, 128)

    def test_unknown_interpolation_raises(self) -> None:
        p = ReIDPreprocessing(interpolation="lanczos")
        pytest.importorskip("torchvision")
        with pytest.raises(ValueError, match="lanczos"):
            p.build_transform()

    def test_to_dict_from_dict_roundtrip(self) -> None:
        p = ReIDPreprocessing(input_size=(128, 64), interpolation="bicubic", to_rgb=False)
        d = p.to_dict()
        assert d["interpolation"] == "bicubic"
        assert d["input_size"] == [128, 64]
        assert d["to_rgb"] is False
        p2 = ReIDPreprocessing.from_dict(d)
        assert p2 == p


# ---------------------------------------------------------------------------
# 2. Registry / resolution
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_resolve_default_alias(self) -> None:
        card = resolve_model_card(DEFAULT_MODEL)
        assert card is not None
        assert card.architecture == "osnet_x1_0"
        assert card.weights is not None
        assert card.domain_warning is not None

    def test_resolve_fastreid_mot17_alias(self) -> None:
        card = resolve_model_card(FASTREID_MOT17_SBS50)
        assert card is not None
        assert card.architecture == "fastreid_sbs_resnest50"
        assert card.weights is not None
        assert card.preprocessing.input_size == (384, 128)
        assert card.domain_warning is not None

    def test_local_dir_with_config(self, tmp_path) -> None:
        import json

        config = {
            "architecture": "osnet_x1_0",
            "preprocessing": ReIDPreprocessing().to_dict(),
        }
        (tmp_path / "reid_config.json").write_text(json.dumps(config))
        card = resolve_model_card(str(tmp_path))
        assert card is not None
        assert card.architecture == "osnet_x1_0"

    def test_bare_path_without_architecture_raises(self) -> None:
        """from_pretrained on a bare weights file without architecture → ValueError."""
        pytest.importorskip("torch")
        from trackers.core.reid.model import ReIDModel

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            tmp_path = f.name
        try:
            # Write a minimal torch save so the file exists
            import torch

            torch.save({}, tmp_path)
            with pytest.raises(ValueError, match="architecture"):
                ReIDModel.from_pretrained(tmp_path)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 3. Loaders
# ---------------------------------------------------------------------------


class TestLoaders:
    def test_missing_local_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_weights("/nonexistent/path/to/weights.pth")

    def test_malformed_hf_url_raises(self) -> None:
        from trackers.core.reid.models.loaders import _resolve_hf

        with pytest.raises(ValueError, match="hf://"):
            _resolve_hf("hf://only_one_part")

    def test_load_state_dict_full_match(self) -> None:
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from trackers.core.reid.models.loaders import load_state_dict_into

        model = nn.Linear(4, 2)
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(model.state_dict(), tmp_path)
            report = load_state_dict_into(model, tmp_path, torch.device("cpu"))
            assert report.matched == report.total
            assert report.matched_fraction == 1.0
        finally:
            os.unlink(tmp_path)

    def test_load_state_dict_mismatch_warns(self) -> None:
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from trackers.core.reid.models.loaders import load_state_dict_into

        source = nn.Linear(4, 2)
        target = nn.Linear(8, 4)
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(source.state_dict(), tmp_path)
            with pytest.warns(UserWarning, match="weights likely do not match"):
                load_state_dict_into(target, tmp_path, torch.device("cpu"))
        finally:
            os.unlink(tmp_path)

    def test_remap_fastreid_sbs_keys(self) -> None:
        pytest.importorskip("torch")
        import torch

        from trackers.core.reid.models.loaders import remap_fastreid_sbs_state_dict

        raw = {
            "backbone.conv1.0.weight": torch.zeros(1),
            "heads.pool_layer.p": torch.tensor([1.5]),
            "heads.bottleneck.0.weight": torch.zeros(2048),
            "heads.weight": torch.zeros(487, 2048),
        }
        mapped = remap_fastreid_sbs_state_dict(raw)
        assert "backbone.conv1.0.weight" in mapped
        assert "pool.p" in mapped
        assert "bottleneck.weight" in mapped
        assert "heads.weight" not in mapped

    def test_load_fastreid_sbs_from_synthetic_checkpoint(self) -> None:
        pytest.importorskip("torch")
        import torch

        from trackers.core.reid.architectures import build_architecture
        from trackers.core.reid.models.loaders import load_fastreid_sbs_state_dict_into

        model = build_architecture("fastreid_sbs_resnest50")
        source = model.state_dict()
        wrapped: dict = {}
        for key, value in source.items():
            if key.startswith("backbone."):
                wrapped[key] = value
            elif key == "pool.p":
                wrapped["heads.pool_layer.p"] = value
            elif key.startswith("bottleneck."):
                wrapped[f"heads.bottleneck.0.{key[len('bottleneck.') :]}"] = value

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(wrapped, tmp_path)
            report = load_fastreid_sbs_state_dict_into(model, tmp_path, torch.device("cpu"))
            assert report.matched == report.total
            assert report.matched_fraction == 1.0
        finally:
            os.unlink(tmp_path)

    def test_fastreid_sbs_forward_shape(self) -> None:
        pytest.importorskip("torch")
        import torch

        from trackers.core.reid.architectures import build_architecture

        model = build_architecture("fastreid_sbs_resnest50")
        model.eval()
        with torch.inference_mode():
            out = model(torch.randn(2, 3, 384, 128))
        assert out.shape == (2, 2048)


# ---------------------------------------------------------------------------
# 4. Model smoke (torch required)
# ---------------------------------------------------------------------------


class TestModelSmoke:
    def test_extract_features_tiny_module(self) -> None:
        pytest.importorskip("torch")
        import numpy as np
        import supervision as sv
        import torch
        import torch.nn as nn

        from trackers.core.reid.model import ReIDModel
        from trackers.core.reid.models.preprocessing import ReIDPreprocessing

        class _TinyEncoder(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.flatten(1)[:, :16]

        device = torch.device("cpu")
        preprocessing = ReIDPreprocessing()
        model = ReIDModel(_TinyEncoder(), device, preprocessing)

        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        dets = sv.Detections(xyxy=np.array([[0, 0, 64, 64]], dtype=np.float32))
        embs = model.extract_features(dets, frame)
        assert embs.shape[0] == 1
        assert embs.dtype == np.float32
        # Verify L2 normalization (default normalize_embeddings=True)
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_extract_features_clamps_out_of_frame_boxes(self) -> None:
        pytest.importorskip("torch")
        import numpy as np
        import supervision as sv
        import torch
        import torch.nn as nn

        from trackers.core.reid.model import ReIDModel
        from trackers.core.reid.models.preprocessing import ReIDPreprocessing

        class _TinyEncoder(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.flatten(1)[:, :16]

        device = torch.device("cpu")
        model = ReIDModel(_TinyEncoder(), device, ReIDPreprocessing())
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        dets = sv.Detections(xyxy=np.array([[-40.0, -40.0, 10.0, 10.0]], dtype=np.float32))
        embs = model.extract_features(dets, frame)
        assert embs.shape[0] == 1

    def test_empty_detections_returns_empty(self) -> None:
        pytest.importorskip("torch")
        import numpy as np
        import supervision as sv
        import torch
        import torch.nn as nn

        from trackers.core.reid.model import ReIDModel
        from trackers.core.reid.models.preprocessing import ReIDPreprocessing

        class _TinyEncoder(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.flatten(1)[:, :16]

        device = torch.device("cpu")
        model = ReIDModel(_TinyEncoder(), device, ReIDPreprocessing())
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        embs = model.extract_features(sv.Detections.empty(), frame)
        assert embs.shape == (0, 0)

    def test_from_pretrained_architecture_only(self) -> None:
        """from_pretrained(architecture=...) builds an OSNet model with no downloads."""
        pytest.importorskip("torch")
        import numpy as np
        import supervision as sv

        from trackers.core.reid.model import ReIDModel

        # osnet_x0_25 is a clean-room architecture with no external weights, so
        # this builds entirely offline (random init).
        model = ReIDModel.from_pretrained(architecture="osnet_x0_25", device="cpu")
        frame = np.zeros((256, 128, 3), dtype=np.uint8)
        dets = sv.Detections(xyxy=np.array([[0, 0, 128, 256]], dtype=np.float32))
        embs = model.extract_features(dets, frame)
        assert embs.ndim == 2
        assert embs.shape[0] == 1
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Round-trip: save_pretrained → from_pretrained (torch required)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_save_and_reload_produces_identical_embeddings(self, tmp_path) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("safetensors")
        import numpy as np
        import supervision as sv

        from trackers.core.reid.model import ReIDModel

        # osnet_x0_25 builds offline; its random init is fixed for this process,
        # so the saved weights reload to byte-identical parameters.
        model_a = ReIDModel.from_pretrained(architecture="osnet_x0_25", device="cpu")
        save_dir = str(tmp_path / "saved_model")
        model_a.save_pretrained(save_dir)

        assert os.path.exists(os.path.join(save_dir, "weights.safetensors"))
        assert os.path.exists(os.path.join(save_dir, "reid_config.json"))

        model_b = ReIDModel.from_pretrained(save_dir, device="cpu")
        assert model_b.preprocessing == model_a.preprocessing

        frame = np.zeros((256, 128, 3), dtype=np.uint8)
        dets = sv.Detections(xyxy=np.array([[0, 0, 128, 256]], dtype=np.float32))
        embs_a = model_a.extract_features(dets, frame)
        embs_b = model_b.extract_features(dets, frame)
        np.testing.assert_allclose(embs_a, embs_b, atol=1e-5)
