# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections.abc import Sequence
from typing import TypeVar

from trackers.core.sort.tracklet import SORTTracklet
from trackers.utils.base_tracklet import BaseTracklet

T_SORTTracklet = TypeVar("T_SORTTracklet", bound="SORTTracklet")


def _get_alive_tracklets(
    tracklets: Sequence[T_SORTTracklet],
    minimum_consecutive_frames: int,
    maximum_frames_without_update: int,
    maximum_time_without_update: float | None = None,
) -> list[T_SORTTracklet]:
    """
    Remove dead or immature lost tracklets and return alive trackers
    that are within the time/frame budget AND (mature OR just updated).

    Note:
        SORT uses total `number_of_successful_updates` (cumulative) for maturity,
        unlike ByteTrack which uses `number_of_successful_consecutive_updates`.
        This means a briefly-lost-and-recovered track retains its maturity in SORT
        but resets to immature in ByteTrack.

    Args:
        tracklets: List of SORTTracklet objects.
        minimum_consecutive_frames: Number of consecutive frames an object must
            be tracked before it is considered a valid track.
        maximum_frames_without_update: Frame-count budget (used when
            ``maximum_time_without_update`` is ``None``).
        maximum_time_without_update: Seconds budget. When provided, this
            criterion is used **instead of** the frame-count budget, enabling
            correct pruning under variable frame rates.

    Returns:
        List of alive tracklets.
    """
    alive_tracklets = []
    for tracklet in tracklets:
        is_mature = tracklet.number_of_successful_updates >= minimum_consecutive_frames
        is_active = tracklet.time_since_update == 0
        within_budget = BaseTracklet.within_lost_track_budget(
            tracklet,
            maximum_frames_without_update=maximum_frames_without_update,
            maximum_time_without_update=maximum_time_without_update,
        )
        if within_budget and (is_mature or is_active):
            alive_tracklets.append(tracklet)
    return alive_tracklets
