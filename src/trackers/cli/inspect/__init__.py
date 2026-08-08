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

from trackers.cli.inspect.cutie import cutie_inspection
from trackers.cli.inspect.mask_manager import mask_manager_inspection
from trackers.cli.inspect.mcbyte import mcbyte_inspection
from trackers.cli.inspect.sam import sam_inspection

__all__ = [
    "INSPECT_COMPONENTS",
    "cutie_inspection",
    "mask_manager_inspection",
    "mcbyte_inspection",
    "sam_inspection",
]

INSPECT_COMPONENTS = {
    "sam": sam_inspection,
    "cutie": cutie_inspection,
    "mask-manager": mask_manager_inspection,
    "mcbyte": mcbyte_inspection,
}
