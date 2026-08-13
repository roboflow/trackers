# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated shim — ``MaskManager`` moved to ``trackers.core.masks.manager``.

The mask stack is no longer McByte-specific, so it was hoisted out of
``trackers.core.mcbyte``. Only ``MaskManager``, the single name this module
exported in 2.6.0, is forwarded.

**When it warns**: ``MaskManager`` here is a pyDeprecate ``deprecated_class``
proxy, so it warns when the name is *used* — instantiated, or an attribute read
— not when it is imported. Instantiating it returns a real
:class:`trackers.core.masks.manager.MaskManager`, and ``isinstance`` forwards to
the real class without warning, but ``proxy is real_class`` is ``False``. The
two abstract base classes in :mod:`trackers.core.mcbyte.masks` warn at import
instead; see that module for why they cannot be proxies.

.. deprecated::
    Import from ``trackers.core.masks.manager`` instead. This module will be
    removed in v3.0.

Example migration::

    # old (deprecated)
    from trackers.core.mcbyte.mask_manager import MaskManager

    # new
    from trackers.core.masks.manager import MaskManager
"""

# TODO(v3.0): remove this shim.

from __future__ import annotations

from deprecate import deprecated_class

from trackers.core.masks.manager import MaskManager as _MaskManager

__all__ = ["MaskManager"]


# ``num_warns=-1`` so every consumer is warned, not only the first one imported
# in a process; see ``trackers.core.mcbyte.masks`` for the full rationale.
@deprecated_class(
    target=_MaskManager,
    deprecated_in="2.7.0",
    remove_in="3.0",
    num_warns=-1,
    # Spelled out because pyDeprecate's built-in template names the class only,
    # which does not tell the reader which import to fix.
    template_mgs=(
        "The `trackers.core.mcbyte.mask_manager.%(source_name)s` was deprecated since v%(deprecated_in)s"
        " in favor of `%(target_path)s`. It will be removed in v%(remove_in)s."
    ),
)
class MaskManager:
    """Deprecated alias for :class:`trackers.core.masks.manager.MaskManager`."""
