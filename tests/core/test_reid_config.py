# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Appearance ReID fusion settings tests."""

from __future__ import annotations

import pytest

from trackers.core.reid.config import ReIDConfig


class TestReIDConfig:
    """Validation of the standalone ReID fusion settings."""

    @pytest.mark.parametrize("value", [-0.01, 1.01])
    def test_rejects_invalid_reid_ema_alpha(self, value: float) -> None:
        with pytest.raises(ValueError, match="ema_alpha"):
            ReIDConfig(ema_alpha=value)

    @pytest.mark.parametrize("parameter", ["appearance_threshold", "proximity_threshold"])
    @pytest.mark.parametrize("value", [-0.01, 1.01])
    def test_rejects_invalid_association_thresholds(self, parameter: str, value: float) -> None:
        with pytest.raises(ValueError, match=parameter):
            # mypy cannot narrow a dynamic dict against typed kwargs.
            ReIDConfig(**{parameter: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_accepts_ema_alpha_boundaries(self, value: float) -> None:
        assert ReIDConfig(ema_alpha=value).ema_alpha == value

    @pytest.mark.parametrize("parameter", ["appearance_threshold", "proximity_threshold"])
    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_accepts_association_threshold_boundaries(self, parameter: str, value: float) -> None:
        # mypy cannot narrow a dynamic dict against typed kwargs.
        config = ReIDConfig(**{parameter: value})  # type: ignore[arg-type]

        assert getattr(config, parameter) == value

    def test_defaults_match_botsort(self) -> None:
        config = ReIDConfig()

        assert (config.ema_alpha, config.appearance_threshold, config.proximity_threshold) == (0.9, 0.25, 0.5)

    def test_adaptive_defaults_match_deep_ocsort(self) -> None:
        config = ReIDConfig(fusion="adaptive")

        assert (config.appearance_weight, config.adaptive_weight_cap) == (0.75, 0.5)

    def test_unknown_fusion_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown fusion"):
            ReIDConfig(fusion="deepocsort")  # type: ignore[arg-type]

    @pytest.mark.parametrize("parameter", ["appearance_weight", "adaptive_weight_cap"])
    @pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
    def test_rejects_unusable_adaptive_weights(self, parameter: str, value: float) -> None:
        with pytest.raises(ValueError, match=parameter):
            ReIDConfig(**{parameter: value})  # type: ignore[arg-type]
