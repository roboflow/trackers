# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Components behind ``trackers inspect``.

Each entry in :data:`INSPECT_COMPONENTS` names the thing it inspects, not the
tracker that happens to use it. The mask stack (``sam``, ``cutie``,
``mask-manager``) is tracker-agnostic and lives in :mod:`trackers.core.masks`;
only ``mcbyte`` inspects a tracker.

These commands need the ``mask`` extra (``pip install 'trackers[mask]'``). The
heavy imports are deferred into the command bodies so that importing the CLI
stays cheap for everyone else.
"""

from trackers.cli.inspect.cutie import cutie_command
from trackers.cli.inspect.mask_manager import mask_manager_command
from trackers.cli.inspect.mcbyte import compare_mcbyte_command
from trackers.cli.inspect.sam import sam_command

__all__ = [
    "INSPECT_COMPONENTS",
    "compare_mcbyte_command",
    "cutie_command",
    "mask_manager_command",
    "sam_command",
]

INSPECT_COMPONENTS = {
    "sam": sam_command,
    "cutie": cutie_command,
    "mask-manager": mask_manager_command,
    "mcbyte": compare_mcbyte_command,
}
