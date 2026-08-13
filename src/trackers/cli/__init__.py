# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Command-line interface for the trackers package.

Layout rule
-----------

The module tree mirrors the command tree, so a command name tells you the file
to open and vice versa:

- **Leaf command → module.** ``trackers track`` is ``cli/track.py``, ``trackers
  eval`` is ``cli/eval.py``.
- **Command group → package.** ``trackers inspect sam`` is
  ``cli/inspect/sam.py``; the package ``__init__`` holds the name→callable table
  the dispatcher reads.
- **Underscore prefix → infrastructure, not a command.** ``_parser``,
  ``_legacy``, ``_progress``, ``_annotate``, and ``_detections`` support the
  commands without being one. Nothing without an underscore is anything but a
  command.

Top-level command names are verbs. A tracker name is an argument to a verb, so
benchmarking McByte is ``trackers benchmark mcbyte`` rather than ``trackers
mcbyte``, and the group member names what is acted on.

Deliberate divergences
----------------------

``inspect``'s members name the component under inspection rather than a tracker,
because the mask stack they exercise is tracker-agnostic — ``sam``, ``cutie``,
and ``mask-manager`` live in :mod:`trackers.core.masks` and reference no tracker
at all. Only ``inspect mcbyte`` inspects a tracker. A ``trackers inspect
<tracker> <component>`` shape was rejected for implying a per-tracker variation
that does not exist: two trackers using SAM would run identical code.

Shared helpers sit at the level of the audience that consumes them.
``_annotate`` and ``_detections`` are at ``cli/`` because both ``benchmark`` and
``inspect`` (and, for the palette, ``track``) use them; helpers only ``inspect``
needs stay in ``cli/inspect/_common.py``.
"""
