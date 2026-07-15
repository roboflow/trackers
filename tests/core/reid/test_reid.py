# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Minimal test suite for the re-ID package.

Tests are grouped by the module they exercise:
  - preprocessing / registry / loaders / model / round-trip (RFC 0002)
  - feature bank + appearance distance
  - eval metrics + dataset loaders

torch/timm-dependent tests are guarded with pytest.importorskip.
No network downloads are performed (see test_fastreid_integration.py).
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, cast

import numpy as np
import pytest

# Heavy ReID deps live behind the optional extra; skip collection when absent.
pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("timm")
pytest.importorskip("huggingface_hub")
pytest.importorskip("safetensors")

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
        pytest.importorskip("torch")
        pytest.importorskip("torchvision")
        t = ReIDPreprocessing().build_transform()
        from PIL import Image

        img = Image.new("RGB", (128, 256))  # PIL (width, height) = (W, H)
        out = t(img)
        assert out.shape == (3, 256, 128)

    def test_stretch_resize_matches_target_size(self) -> None:
        pytest.importorskip("cv2")
        crop = np.zeros((100, 50, 3), dtype=np.uint8)
        out = ReIDPreprocessing(input_size=(384, 128), resize_mode="stretch").resize_crop(crop)
        assert out.shape == (384, 128, 3)

    def test_letterbox_preserves_aspect_and_pads(self) -> None:
        pytest.importorskip("cv2")
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
        pytest.importorskip("cv2")
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

    def test_generic_loader_preserves_backbone_prefix(self) -> None:
        pytest.importorskip("torch")
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

    def test_remap_fastreid_sbs_keys(self) -> None:
        pytest.importorskip("torch")
        import torch

        from trackers.core.reid.architectures.fastreid_sbs import (
            remap_fastreid_sbs_state_dict,
        )

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
            report = load_state_dict_for_architecture(model, tmp_path, torch.device("cpu"), FASTREID_SBS_ARCHITECTURE)
            assert report.matched == report.total
            assert report.matched_fraction == 1.0
        finally:
            os.unlink(tmp_path)

    def test_fastreid_sbs_loads_gem_p_from_checkpoint(self) -> None:
        """GeM ``p`` must come from ``heads.pool_layer.p``, not the 3.0 constructor default."""
        pytest.importorskip("torch")
        import torch

        from trackers.core.reid.architectures import build_architecture
        from trackers.core.reid.architectures.fastreid_sbs import (
            FASTREID_SBS_ARCHITECTURE,
            FastReIDSBSResNeSt50,
        )
        from trackers.core.reid.models.loaders import load_state_dict_for_architecture

        mot17_gem_p = 1.7194522619247437  # heads.pool_layer.p in mot17_sbs_S50.pth

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
            report = load_state_dict_for_architecture(model, tmp_path, torch.device("cpu"), FASTREID_SBS_ARCHITECTURE)
            assert report.matched == report.total
            assert model.pool.p.item() == pytest.approx(mot17_gem_p)
            assert model.pool.p.item() != pytest.approx(3.0)
        finally:
            os.unlink(tmp_path)

    def test_fastreid_sbs_last_stride_patch(self) -> None:
        """``layer4[0]`` must match FastReID LAST_STRIDE=1 (not raw timm output_stride=16)."""
        pytest.importorskip("torch")
        pytest.importorskip("timm")
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
        pytest.importorskip("torch")
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


# ---------------------------------------------------------------------------
# 6. Feature bank + appearance distance
# ---------------------------------------------------------------------------


class TestFeatureBank:
    def test_first_update_normalizes(self) -> None:
        from trackers.core.reid.feature_bank import FeatureBank

        bank = FeatureBank(alpha=0.9)
        assert bank.update(np.array([3.0, 4.0], dtype=np.float32))
        feature = bank.feature
        assert feature is not None
        np.testing.assert_allclose(np.linalg.norm(feature), 1.0, atol=1e-6)

    def test_zero_and_non_finite_embeddings_are_skipped(self) -> None:
        from trackers.core.reid.feature_bank import FeatureBank

        bank = FeatureBank()
        assert bank.update(np.zeros(8, dtype=np.float32)) is False
        assert bank.update(np.array([1.0, np.nan], dtype=np.float32)) is False
        assert not bank.is_initialized

    def test_shape_change_is_rejected(self) -> None:
        from trackers.core.reid.feature_bank import FeatureBank

        bank = FeatureBank()
        bank.update(np.array([1.0, 0.0], dtype=np.float32))
        before = bank.feature
        assert before is not None
        assert bank.update(np.array([1.0, 0.0, 0.0], dtype=np.float32)) is False
        after = bank.feature
        assert after is not None
        np.testing.assert_allclose(before, after)


class TestAppearanceSimilarity:
    def test_non_finite_detection_rows_are_sanitized(self) -> None:
        from trackers.core.reid.distance import appearance_similarity

        sim = appearance_similarity(
            [np.array([1.0, 0.0], dtype=np.float32)],
            np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32),
        )
        assert np.isfinite(sim).all()
        assert sim[0, 0] == pytest.approx(1.0)
        assert sim[0, 1] == pytest.approx(0.0)

    def test_skips_incompatible_track_dimensions(self) -> None:
        from trackers.core.reid.distance import appearance_similarity

        sim = appearance_similarity(
            [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        assert sim.shape == (1, 1)
        assert sim[0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. Eval metrics + dataset loaders
# ---------------------------------------------------------------------------


class TestComputeReidMetrics:
    def test_market1501_pid_zero_is_junk_in_gallery(self) -> None:
        from trackers.core.reid.eval.datasets import MARKET1501_GALLERY_JUNK_PIDS
        from trackers.core.reid.eval.metrics import compute_reid_metrics

        metrics = compute_reid_metrics(
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            q_pids=np.array([1]),
            g_pids=np.array([0, 1, 2]),
            q_camids=np.array([0]),
            g_camids=np.array([0, 1, 0]),
            max_rank=3,
            gallery_junk_pids=MARKET1501_GALLERY_JUNK_PIDS,
        )
        assert metrics.rank1 == pytest.approx(100.0)

    def test_cmc_pads_when_valid_gallery_shorter_than_max_rank(self) -> None:
        from trackers.core.reid.eval.metrics import compute_reid_metrics

        metrics = compute_reid_metrics(
            np.array([[0.2, 0.1]], dtype=np.float32),
            q_pids=np.array([3]),
            g_pids=np.array([3, 8]),
            q_camids=np.array([0]),
            g_camids=np.array([1, 0]),
            max_rank=5,
        )
        assert metrics.rank1 == pytest.approx(0.0)
        assert metrics.rank5 == pytest.approx(100.0)

    def test_reid_metrics_map_alias(self) -> None:
        from trackers.core.reid.eval.metrics import ReIDMetrics

        metrics = ReIDMetrics(
            mean_average_precision=42.0,
            rank1=1.0,
            rank5=2.0,
            rank10=3.0,
            minp=4.0,
            num_queries=1,
        )
        assert metrics.map == pytest.approx(42.0)


class TestMarket1501Loader:
    def test_load_market1501_from_temp_tree(self, tmp_path) -> None:
        from trackers.core.reid.eval.datasets import (
            MARKET1501_GALLERY_JUNK_PIDS,
            load_market1501,
        )

        query_dir = tmp_path / "query"
        gallery_dir = tmp_path / "bounding_box_test"
        query_dir.mkdir()
        gallery_dir.mkdir()
        (query_dir / "0001_c1s1_001051_00.jpg").write_bytes(b"jpeg")
        (gallery_dir / "0000_c1s1_000151_01.jpg").write_bytes(b"jpeg")
        (gallery_dir / "0002_c2s1_000851_01.jpg").write_bytes(b"jpeg")

        query, gallery = load_market1501(tmp_path)
        assert len(query) == 1
        assert query.pids.tolist() == [1]
        assert gallery.pids.tolist() == [0, 2]
        assert gallery.gallery_junk_pids == MARKET1501_GALLERY_JUNK_PIDS


class TestMSMT17Loader:
    def test_load_msmt17_from_temp_lists(self, tmp_path) -> None:
        from pathlib import Path

        from trackers.core.reid.eval.datasets import load_msmt17

        root = Path(tmp_path)
        test_root = root / "test"
        test_root.mkdir()
        rel = "0001/0001_019_07_0303morning_0020_1.jpg"
        image_path = test_root / rel
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(b"jpeg")
        (root / "list_query.txt").write_text(f"{rel} 42\n")
        (root / "list_gallery.txt").write_text(f"{rel} 42 6\n")

        query, gallery = load_msmt17(root)
        assert query.pids.tolist() == [42]
        assert query.camids.tolist() == [6]
        assert gallery.pids.tolist() == [42]
        assert gallery.gallery_junk_pids == frozenset({-1})
