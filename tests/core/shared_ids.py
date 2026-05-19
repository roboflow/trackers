# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared test constants for tracker IDs used across test/core files."""

ALL_TRACKER_IDS = ["sort", "bytetrack", "ocsort", "botsort", "cbiou"]

# Trackers that accept a user-supplied ``iou=`` constructor argument.
# C-BIoU is intentionally excluded: it is opinionated and always uses BIoU.
IOU_TRACKER_IDS = [tid for tid in ALL_TRACKER_IDS if tid != "cbiou"]
