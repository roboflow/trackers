# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID model / architecture / loader tests.

Requires ``trackers[reid]`` (torch, timm, huggingface_hub, safetensors).
Installed and run in CI via ``uv sync --group dev --extra reid``.
Curated weight downloads live in the ``@pytest.mark.slow`` test below.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, cast

import numpy as np
import pytest

from trackers.core.reid.models.loaders import resolve_weights
from trackers.core.reid.models.preprocessing import ReIDPreprocessing
from trackers.core.reid.models.registry import (
    DEFAULT_MODEL,
    FASTREID_MOT17_SBS50,
    default_preprocessing_for_architecture,
    resolve_model_card,
)

# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------


class TestReIDPreprocessing:
    def test_build_transform_callable(self) -> None:
        t = ReIDPreprocessing().build_transform()
        from PIL import Image

        img = Image.new("RGB", (128, 256))  # PIL (width, height) = (W, H)
        out = t(img)
        assert out.shape == (3, 256, 128)

    def test_stretch_resize_matches_target_size(self) -> None:
        crop = np.zeros((100, 50, 3), dtype=np.uint8)
        out = ReIDPreprocessing(input_size=(384, 128), resize_mode="stretch").resize_crop(crop)
        assert out.shape == (384, 128, 3)

    def test_letterbox_preserves_aspect_and_pads(self) -> None:
        crop = np.full((200, 100, 3), 255, dtype=np.uint8)
        out = ReIDPreprocessing(input_size=(384, 128), resize_mode="letterbox").resize_crop(crop)
        assert out.shape == (384, 128, 3)
        assert out[0, 0, 0] == 255
        assert out[-1, 0, 0] == 114

    def test_fastreid_default_uses_stretch(self) -> None:
        from trackers.core.reid.architectures.fastreid_sbs import FASTREID_SBS_ARCHITECTURE

        fastreid = default_preprocessing_for_architecture(FASTREID_SBS_ARCHITECTURE)
        assert fastreid.input_size == (384, 128)
        assert fastreid.resize_mode == "stretch"

    def test_unknown_interpolation_raises(self) -> None:
        p = ReIDPreprocessing(interpolation="lanczos")
        crop = np.zeros((32, 16, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="lanczos"):
            p.resize_crop(crop)

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
        from trackers.core.reid.architectures.fastreid_sbs import FASTREID_SBS_ARCHITECTURE

        card = resolve_model_card(FASTREID_MOT17_SBS50)
        assert card is not None
        assert card.architecture == FASTREID_SBS_ARCHITECTURE
        assert card.weights is not None
        assert card.preprocessing.input_size == (384, 128)
        assert card.domain_warning is not None

    def test_default_preprocessing_for_architecture(self) -> None:
        from trackers.core.reid.architectures.fastreid_sbs import FASTREID_SBS_ARCHITECTURE

        osnet = default_preprocessing_for_architecture("osnet_x1_0")
        assert osnet.input_size == (256, 128)
        fastreid = default_preprocessing_for_architecture(FASTREID_SBS_ARCHITECTURE)
        assert fastreid.input_size == (384, 128)
        assert default_preprocessing_for_architecture("timm:resnet50").input_size == (256, 128)

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

    def test_generic_loader_preserves_backbone_prefix(self) -> None:
        import torch
        import torch.nn as nn

        from trackers.core.reid.models.loaders import load_state_dict_into

        class _BackboneModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = nn.Linear(4, 2)

        model = _BackboneModel()
        wrapped = {
            "backbone.weight": model.backbone.weight.detach().clone(),
            "backbone.bias": model.backbone.bias.detach().clone(),
        }
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(wrapped, tmp_path)
            report = load_state_dict_into(model, tmp_path, torch.device("cpu"))
            assert report.matched == report.total
            assert report.matched_fraction == 1.0
        finally:
            os.unlink(tmp_path)

    def test_osnet_backbone_loads_with_full_match(self, tmp_path) -> None:
        """Inference OSNet has no classifier, so strict checkpoint loads succeed."""
        import torch

        from trackers.core.reid.architectures import build_architecture
        from trackers.core.reid.models.loaders import load_state_dict_for_architecture

        source = build_architecture("osnet_x0_25")
        assert not any(key.startswith("classifier.") for key in source.state_dict())

        path = tmp_path / "osnet.pth"
        torch.save(source.state_dict(), path)

        target = build_architecture("osnet_x0_25")
        report = load_state_dict_for_architecture(
            target,
            str(path),
            torch.device("cpu"),
            "osnet_x0_25",
            required_match_fraction=1.0,
        )
        assert report.matched_fraction == 1.0

    def test_remap_fastreid_sbs_keys(self) -> None:
        import torch

        from trackers.core.reid.architectures.fastreid_sbs import remap_fastreid_sbs_state_dict

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
        import torch

        from trackers.core.reid.architectures import build_architecture
        from trackers.core.reid.architectures.fastreid_sbs import FASTREID_SBS_ARCHITECTURE
        from trackers.core.reid.models.loaders import load_state_dict_for_architecture

        model = build_architecture(FASTREID_SBS_ARCHITECTURE)
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
            report = load_state_dict_for_architecture(
                model, tmp_path, torch.device("cpu"), FASTREID_SBS_ARCHITECTURE
            )
            assert report.matched == report.total
            assert report.matched_fraction == 1.0
        finally:
            os.unlink(tmp_path)

    def test_fastreid_sbs_loads_gem_p_from_checkpoint(self) -> None:
        """GeM ``p`` must come from ``heads.pool_layer.p``, not the 3.0 constructor default."""
        import torch

        from trackers.core.reid.architectures import build_architecture
        from trackers.core.reid.architectures.fastreid_sbs import (
            FASTREID_SBS_ARCHITECTURE,
            FastReIDSBSResNeSt50,
        )
        from trackers.core.reid.models.loaders import load_state_dict_for_architecture

        mot17_gem_p = 1.7194522619247437

        model = cast(FastReIDSBSResNeSt50, build_architecture(FASTREID_SBS_ARCHITECTURE))
        assert model.pool.p.item() == pytest.approx(3.0)

        wrapped: dict = {"heads.pool_layer.p": torch.tensor([mot17_gem_p])}
        for key, value in model.state_dict().items():
            if key.startswith("backbone."):
                wrapped[key] = value
            elif key.startswith("bottleneck."):
                wrapped[f"heads.bottleneck.0.{key[len('bottleneck.') :]}"] = value

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save(wrapped, tmp_path)
            report = load_state_dict_for_architecture(
                model, tmp_path, torch.device("cpu"), FASTREID_SBS_ARCHITECTURE
            )
            assert report.matched == report.total
            assert model.pool.p.item() == pytest.approx(mot17_gem_p)
        finally:
            os.unlink(tmp_path)

    def test_fastreid_sbs_last_stride_patch(self) -> None:
        """``layer4[0]`` must match FastReID LAST_STRIDE=1 (not raw timm output_stride=16)."""
        import torch.nn as nn

        from trackers.core.reid.architectures import build_architecture
        from trackers.core.reid.architectures.fastreid_sbs import (
            FASTREID_SBS_ARCHITECTURE,
            FastReIDSBSResNeSt50,
        )

        model = cast(FastReIDSBSResNeSt50, build_architecture(FASTREID_SBS_ARCHITECTURE))
        block = cast(Any, cast(Any, model.backbone).layer4[0])
        assert isinstance(block.downsample[0], nn.AvgPool2d)
        assert block.downsample[0].kernel_size == (1, 1) or block.downsample[0].kernel_size == 1
        assert block.downsample[0].stride == (1, 1) or block.downsample[0].stride == 1
        assert isinstance(block.avd_last, nn.AvgPool2d)
        assert block.avd_last.kernel_size == (3, 3) or block.avd_last.kernel_size == 3
        assert block.avd_last.stride == (1, 1) or block.avd_last.stride == 1

    def test_fastreid_sbs_forward_shape(self) -> None:
        import torch

        from trackers.core.reid.architectures import build_architecture
        from trackers.core.reid.architectures.fastreid_sbs import FASTREID_SBS_ARCHITECTURE

        model = build_architecture(FASTREID_SBS_ARCHITECTURE)
        model.eval()
        with torch.inference_mode():
            out = model(torch.randn(2, 3, 384, 128))
        assert out.shape == (2, 2048)


# ---------------------------------------------------------------------------
# 4. Model smoke (torch required)
# ---------------------------------------------------------------------------


class TestModelSmoke:
    def test_extract_features_tiny_module(self) -> None:
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
        # Default: embeddings are not L2-normalised
        norms = np.linalg.norm(embs, axis=1)
        assert not np.allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_extract_features_optional_l2_normalize(self) -> None:
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
        preprocessing = ReIDPreprocessing(normalize_embeddings=True)
        model = ReIDModel(_TinyEncoder(), device, preprocessing)

        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        dets = sv.Detections(xyxy=np.array([[0, 0, 64, 64]], dtype=np.float32))
        embs = model.extract_features(dets, frame)
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_extract_features_clamps_out_of_frame_boxes(self) -> None:
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
        assert embs.dtype == np.float32
        norms = np.linalg.norm(embs, axis=1)
        assert not np.allclose(norms, np.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Round-trip: save_pretrained → from_pretrained (torch required)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_save_and_reload_produces_identical_embeddings(self, tmp_path) -> None:
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


# ---------------------------------------------------------------------------
# 6. Slow: curated FastReID weight download
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_fastreid_mot17_alias_loads_with_finite_normalized_output() -> None:
    """Download ``fastreid_mot17_sbs50`` and smoke the BoT-SORT encoder path."""
    import supervision as sv

    from trackers.core.reid.model import ReIDModel

    model = ReIDModel.from_pretrained("fastreid_mot17_sbs50", device="cpu")
    frame = np.zeros((384, 128, 3), dtype=np.uint8)
    dets = sv.Detections(xyxy=np.array([[0.0, 0.0, 128.0, 384.0]], dtype=np.float32))
    embs = model.extract_features(dets, frame)
    assert embs.shape == (1, 2048)
    assert np.isfinite(embs).all()
    np.testing.assert_allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-4)
    assert model.preprocessing.input_size == (384, 128)
