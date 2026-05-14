# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated shim — CMC symbols moved to ``trackers.utils.cmc``.

.. deprecated::
    Import from ``trackers.utils.cmc`` or the top-level ``trackers`` package
    directly.  This module will be removed in v3.0.

Example migration::

    # old (deprecated)
    from trackers.core.botsort.cmc import CMC, CMCConfig

    # new
    from trackers.utils.cmc import CMC, CMCConfig
    # or
    from trackers import CMC, CMCConfig
"""

from __future__ import annotations

import warnings


def __getattr__(name: str) -> object:
    """Provide back-compat access with deprecation warnings for every symbol."""
    from trackers.utils.cmc import CMC, CMCConfig, CMCMethod

    _symbols: dict[str, object] = {
        "CMC": CMC,
        "CMCConfig": CMCConfig,
        "CMCMethod": CMCMethod,
        "CMCTMethod": CMCMethod,  # CMCTMethod was the pre-refactor name
    }
    if name in _symbols:
        warnings.warn(
            f"Importing {name!r} from 'trackers.core.botsort.cmc' is deprecated and will "
            "be removed in v3.0.  Use 'from trackers.utils.cmc import ...' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return _symbols[name]
    raise AttributeError(f"module 'trackers.core.botsort.cmc' has no attribute {name!r}")
