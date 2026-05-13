# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Smoke tests that CMC symbols are reachable from documented import paths."""

from __future__ import annotations


def test_cmc_importable_from_trackers_utils() -> None:
    """CMC, CMCConfig, CMCMethod must be importable from trackers.utils.cmc."""
    from trackers.utils.cmc import CMC, CMCConfig, CMCMethod  # noqa: F401


def test_cmc_importable_from_top_level() -> None:
    """CMC, CMCConfig, CMCMethod must be importable from the top-level trackers package."""
    from trackers import CMC, CMCConfig, CMCMethod  # noqa: F401


def test_compat_shim_emits_deprecation_warning() -> None:
    """Importing from trackers.core.botsort.cmc must emit DeprecationWarning."""
    import importlib
    import sys
    import warnings

    # Evict cached module so __getattr__ fires fresh
    sys.modules.pop("trackers.core.botsort.cmc", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod = importlib.import_module("trackers.core.botsort.cmc")
        _ = mod.CMC  # triggers __getattr__

    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "Expected DeprecationWarning when importing CMC from trackers.core.botsort.cmc"
    )
