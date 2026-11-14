# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Components behind ``trackers inspect``.

Each entry in :data:`INSPECT_COMPONENTS` names the thing it inspects, not the tracker that happens to use it. The mask
stack (``sam``, ``cutie``, ``edgetam``, ``mask-manager``) is tracker-agnostic and lives in :mod:`trackers.core.masks`;
only ``mcbyte`` inspects a tracker.

EdgeTAM needs the ``edgetam`` extra; the other commands use ``mask``. Heavy inference imports are deferred into the
command bodies so that importing the CLI stays cheap for everyone else.
"""

from trackers.cli.inspect.cutie import cutie_inspection
from trackers.cli.inspect.edgetam import edgetam_inspection
from trackers.cli.inspect.mask_manager import mask_manager_inspection
from trackers.cli.inspect.mcbyte import mcbyte_inspection
from trackers.cli.inspect.sam import sam_inspection

__all__ = [
    "INSPECT_COMPONENTS",
    "cutie_inspection",
    "edgetam_inspection",
    "mask_manager_inspection",
    "mcbyte_inspection",
    "sam_inspection",
]

INSPECT_COMPONENTS = {
    "sam": sam_inspection,
    "cutie": cutie_inspection,
    "edgetam": edgetam_inspection,
    "mask-manager": mask_manager_inspection,
    "mcbyte": mcbyte_inspection,
}
