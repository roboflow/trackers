# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for the shared mask-pipeline primitives in ``trackers.core.masks.base``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from trackers.core.masks.base import _resolve_auto_device  # noqa: E402


class TestResolveAutoDevice:
    """``device="auto"`` is what every mask component and CLI command defaults to."""

    @pytest.mark.parametrize(
        ("cuda_available", "expected"),
        [
            pytest.param(True, "cuda", id="cuda-present"),
            pytest.param(False, "cpu", id="cuda-absent"),
        ],
    )
    def test_auto_follows_cuda_availability(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cuda_available: bool,
        expected: str,
    ) -> None:
        """CUDA is taken when it is there, and CPU is the fallback, never a failure."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

        assert _resolve_auto_device() == expected

    def test_mps_is_never_auto_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MPS is measurably slower than CPU for this pipeline, so ``auto`` must skip it.

        Availability is forced on for both accelerators; only an explicit ``device="mps"`` may opt in.
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)

        assert _resolve_auto_device() == "cpu"
