# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Lazy boundary to the optional ``roboflow-reid`` package.

Trackers ships only numpy-only association glue. The appearance encoder,
weights, preprocessing, and catalog live in the standalone ``reid`` package,
installed via the ``trackers[reid]`` extra. This module is the single seam that
resolves ``reid.ReIDModel`` on demand so importing trackers never pulls torch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reid import ReIDModel as ReIDModel

REID_INSTALL_HINT = (
    "ReID features require the optional `trackers[reid]` extra. Install with: pip install 'trackers[reid]'"
)

_REID_PACKAGE = "reid"


def import_reid_model() -> Any:
    """Return ``reid.ReIDModel``, rewriting the missing-extra error.

    Raises:
        ImportError: With an install hint when ``roboflow-reid`` (or one of its
            heavy dependencies) is not installed.
    """
    try:
        module = importlib.import_module(_REID_PACKAGE)
    except ImportError as exc:
        raise ImportError(REID_INSTALL_HINT) from exc
    return module.ReIDModel
