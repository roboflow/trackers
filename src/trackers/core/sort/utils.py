# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections.abc import Sequence
from typing import TypeVar

from trackers.core.sort.tracklet import SORTTracklet

T_SORTTracklet = TypeVar("T_SORTTracklet", bound="SORTTracklet")


def _get_alive_tracklets(
    tracklets: Sequence[T_SORTTracklet],
    minimum_consecutive_frames: int,
    maximum_frames_without_update: int,
) -> list[T_SORTTracklet]:
    """
    Remove dead or immature lost tracklets and get alive trackers
    that are within `maximum_frames_without_update` AND (it's mature OR
    it was just updated).

    Note:
        SORT uses total `number_of_successful_updates` (cumulative) for maturity,
        unlike ByteTrack which uses `number_of_successful_consecutive_updates`.
        This means a briefly-lost-and-recovered track retains its maturity in SORT
        but resets to immature in ByteTrack.

    Args:
        tracklets: List of SORTTracklet objects.
        minimum_consecutive_frames: Number of consecutive frames that an object
            must be tracked before it is considered a 'valid' track.
        maximum_frames_without_update: Maximum number of frames without update
            before a track is considered dead.

    Returns:
        List of alive tracklets.
    """
    alive_tracklets = []
    for tracklet in tracklets:
        is_mature = tracklet.number_of_successful_updates >= minimum_consecutive_frames
        is_active = tracklet.time_since_update == 0
        if tracklet.time_since_update < maximum_frames_without_update and (is_mature or is_active):
            alive_tracklets.append(tracklet)
    return alive_tracklets
