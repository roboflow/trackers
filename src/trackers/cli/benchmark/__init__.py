# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Trackers benchmarked by ``trackers benchmark``.

Each key in :data:`BENCHMARK_COMMANDS` names a tracker run over a benchmark test set. McByte supports several datasets;
McByte++ currently supports SoccerNet.

McByte needs the ``mask`` extra and McByte++ needs ``edgetam``. Heavy imports are deferred until execution so CLI
discovery does not load inference dependencies.
"""

from trackers.cli.benchmark.mcbyte import benchmark_command
from trackers.cli.benchmark.mcbyteplusplus import benchmark_command as mcbyteplusplus_benchmark_command

__all__ = ["BENCHMARK_COMMANDS", "benchmark_command", "mcbyteplusplus_benchmark_command"]

BENCHMARK_COMMANDS = {
    "mcbyte": benchmark_command,
    "mcbyteplusplus": mcbyteplusplus_benchmark_command,
}
