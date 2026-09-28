# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Trackers benchmarked by ``trackers benchmark``.

Each key in :data:`BENCHMARK_COMMANDS` names a tracker run over a complete benchmark test set. Only McByte has a
benchmark harness today; the group exists so the top-level CLI holds verbs alone.

These commands need the ``mask`` extra (``pip install 'trackers[mask]'``). The heavy imports are deferred into the
command bodies so that importing the CLI stays cheap for everyone else.
"""

from trackers.cli.benchmark.mcbyte import benchmark_command

__all__ = ["BENCHMARK_COMMANDS", "benchmark_command"]

BENCHMARK_COMMANDS = {
    "mcbyte": benchmark_command,
}
