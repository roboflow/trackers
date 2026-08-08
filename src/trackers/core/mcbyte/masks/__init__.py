# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated shim — mask primitives moved to ``trackers.core.masks``.

The mask stack is no longer McByte-specific, so it was hoisted out of
``trackers.core.mcbyte``. Only the four names this subpackage exported in 2.6.0
are forwarded; the deep submodule paths (``.base``, ``.sam``, ``.cutie``,
``.dummy``) are not recreated and must be imported from their new home.

**When each name warns**, because the two halves use different mechanisms:

- ``MaskOutput`` and ``TrackletSnapshot`` are pyDeprecate ``deprecated_class``
  proxies. They warn when the name is *used* — instantiated, or an attribute
  read — not when it is imported. ``isinstance`` forwards to the real class but
  does not warn, and ``proxy is real_class`` is ``False``.
- ``MaskGenerator`` and ``MaskPropagator`` are served by the module
  ``__getattr__`` below and warn at *import*, handing back the real class.
  They cannot be proxies: both are abstract base classes, and a proxy raises
  ``TypeError`` when subclassed, which is the only thing those two are for.

.. deprecated::
    Import from ``trackers.core.masks`` instead. This module will be removed in
    v3.0.

Example migration::

    # old (deprecated)
    from trackers.core.mcbyte.masks import MaskGenerator, MaskOutput

    # new
    from trackers.core.masks import MaskGenerator, MaskOutput
"""

# TODO(v3.0): remove this shim.

from __future__ import annotations

from typing import TYPE_CHECKING

from deprecate import deprecated_class

# pyDeprecate's own emitter, the documented default ``stream`` of ``deprecated``
# and ``deprecated_class``. Used so the ``__getattr__`` half warns through the
# same machinery as the proxied half rather than a hand-rolled ``warnings.warn``.
from deprecate.deprecation import deprecation_warning

from trackers.core.masks.base import MaskOutput as _MaskOutput
from trackers.core.masks.base import TrackletSnapshot as _TrackletSnapshot

if TYPE_CHECKING:
    # Bound for type checkers and for ruff's ``__all__`` resolution only. At
    # runtime these two names are served by ``__getattr__``, so that they warn.
    from trackers.core.masks.base import MaskGenerator, MaskPropagator

__all__ = [
    "MaskGenerator",
    "MaskOutput",
    "MaskPropagator",
    "TrackletSnapshot",
]

# Names this module forwards through ``__getattr__`` rather than through a proxy.
_FORWARDED_ABCS = ("MaskGenerator", "MaskPropagator")

# ``num_warns=-1`` rather than the default 1: a shim that warns once per process
# would stay silent for every consumer imported after the first, which is the
# opposite of what a migration aid is for. Python's own filter still collapses
# the warning to once per call site.
_DEPRECATION = {
    "deprecated_in": "2.7.0",
    "remove_in": "3.0",
    "num_warns": -1,
    # pyDeprecate's built-in template names the class only, which does not say
    # which import to fix. The prefix is spelled out so the proxied names read
    # the same as the ``__getattr__`` ones below.
    "template_mgs": (
        "The `trackers.core.mcbyte.masks.%(source_name)s` was deprecated since v%(deprecated_in)s"
        " in favor of `%(target_path)s`. It will be removed in v%(remove_in)s."
    ),
}


@deprecated_class(target=_MaskOutput, **_DEPRECATION)  # type: ignore[arg-type]
class MaskOutput:
    """Deprecated alias for :class:`trackers.core.masks.base.MaskOutput`."""


@deprecated_class(target=_TrackletSnapshot, **_DEPRECATION)  # type: ignore[arg-type]
class TrackletSnapshot:
    """Deprecated alias for :class:`trackers.core.masks.base.TrackletSnapshot`."""


def __getattr__(name: str) -> object:
    """Forward a moved abstract base class, warning at import time.

    Args:
        name: Attribute requested from this module.

    Returns:
        The real class of the same name from :mod:`trackers.core.masks.base`.

    Raises:
        AttributeError: If ``name`` was not exported by the 2.6.0 subpackage.
    """
    if name not in _FORWARDED_ABCS:
        raise AttributeError(f"module 'trackers.core.mcbyte.masks' has no attribute {name!r}")

    from trackers.core.masks import base

    deprecation_warning(
        f"The `trackers.core.mcbyte.masks.{name}` was deprecated since v2.7.0"
        f" in favor of `trackers.core.masks.{name}`. It will be removed in v3.0.",
        stacklevel=3,
    )
    return getattr(base, name)
