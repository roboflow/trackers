# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections.abc import Sequence

from trackers.core.botsort.tracklet import BoTSORTTracklet


def get_alive_trackers(
    trackers: Sequence[BoTSORTTracklet],
    minimum_consecutive_frames: int,
    maximum_frames_without_update: int,
) -> list[BoTSORTTracklet]:
    """
    Remove dead or immature lost tracklets and return alive ones.

    A tracklet is kept if it is within ``maximum_frames_without_update`` **and**
    it is either mature (enough successful updates) or was just updated this
    frame.

    Args:
        trackers: List of BoTSORTTracklet objects.
        minimum_consecutive_frames: Number of successful updates that an object
            must have before it is considered a 'valid' track.
        maximum_frames_without_update: Maximum number of frames without update
            before a track is considered dead.

    Returns:
        List of alive tracklets.
    """
    alive_trackers = []
    for tracker in trackers:
        is_mature = tracker.number_of_successful_updates >= minimum_consecutive_frames
        is_active = tracker.time_since_update == 0
        if tracker.time_since_update < maximum_frames_without_update and (
            is_mature or is_active
        ):
            alive_trackers.append(tracker)
    return alive_trackers
