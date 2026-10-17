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

    @pytest.mark.parametrize("parameter", ["ema_alpha", "appearance_threshold", "proximity_threshold"])
    @pytest.mark.parametrize("value", [-0.01, 1.01])
    def test_rejects_values_outside_unit_range(self, parameter: str, value: float) -> None:
        with pytest.raises(ValueError, match=parameter):
            # mypy cannot narrow a dynamic dict against typed kwargs.
            ReIDConfig(**{parameter: value})  # type: ignore[arg-type]

    @pytest.mark.parametrize("parameter", ["ema_alpha", "appearance_threshold", "proximity_threshold"])
    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_accepts_unit_range_boundaries(self, parameter: str, value: float) -> None:
        # mypy cannot narrow a dynamic dict against typed kwargs.
        config = ReIDConfig(**{parameter: value})  # type: ignore[arg-type]

        assert getattr(config, parameter) == value

    def test_defaults_match_botsort(self) -> None:
        # BoT-SORT's appearance_thresh, proximity_thresh and feature EMA momentum.
        config = ReIDConfig()

        assert (config.ema_alpha, config.appearance_threshold, config.proximity_threshold) == (0.9, 0.25, 0.5)
