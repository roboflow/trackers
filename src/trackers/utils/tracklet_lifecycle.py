# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared tracklet lifecycle helpers."""

from __future__ import annotations

from trackers.utils.base_tracklet import BaseTracklet


def within_lost_track_budget(
    tracklet: BaseTracklet,
    *,
    maximum_frames_without_update: int,
    maximum_time_without_update: float | None = None,
) -> bool:
    """Return whether a tracklet is still within its lost-track budget.

    Args:
        tracklet: Tracklet whose miss counters are checked.
        maximum_frames_without_update: Frame-count budget when ``maximum_time_without_update``
            is ``None``.
        maximum_time_without_update: Seconds budget. When provided, frame-count
            budget is ignored.
    """
    if maximum_time_without_update is not None:
        return tracklet.time_since_update_seconds < maximum_time_without_update
    return tracklet.time_since_update < maximum_frames_without_update
