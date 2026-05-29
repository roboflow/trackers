# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections.abc import Sequence
from typing import TypeVar

from trackers.utils.base_tracklet import BaseTracklet

T_ByteTrackTracklet = TypeVar("T_ByteTrackTracklet", bound="BaseTracklet")


def _get_alive_tracklets(
    tracklets: Sequence[T_ByteTrackTracklet],
    minimum_consecutive_frames: int,
    maximum_frames_without_update: int,
    maximum_time_without_update: float | None = None,
) -> list[T_ByteTrackTracklet]:
    """
    Remove dead or immature lost tracklets and return alive trackers
    that are within the time/frame budget AND (mature OR just updated).

    Note:
        Maturity is sticky: once a tracklet has reached
        `minimum_consecutive_frames` consecutive observations it is
        assigned a non-negative `tracker_id` (i.e. `tracker_id != -1`)
        and stays "confirmed" through subsequent missed frames until
        pruned. This matches the original ByteTrack paper's "confirmed
        track" semantics. For unconfirmed tracks (`tracker_id == -1`),
        maturity is determined by `number_of_successful_consecutive_updates`;
        this counter is reset to 0 in `predict()` on any missed frame.

    Args:
        tracklets: List of BaseTracklet objects.
        minimum_consecutive_frames: Number of consecutive frames an object
            must be tracked before it is considered a valid track.
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
        # Once a tracklet reaches consecutive-update maturity it gets a
        # non-negative tracker_id (assigned by the tracker), and that id
        # is never reset. So tracker_id != -1 is the sticky "confirmed"
        # signal we want here.
        is_mature = tracklet.tracker_id != -1 or (
            tracklet.number_of_successful_consecutive_updates >= minimum_consecutive_frames
        )
        is_active = tracklet.time_since_update == 0
        within_budget = BaseTracklet.within_lost_track_budget(
            tracklet,
            maximum_frames_without_update=maximum_frames_without_update,
            maximum_time_without_update=maximum_time_without_update,
        )
        if within_budget and (is_mature or is_active):
            alive_tracklets.append(tracklet)
    return alive_tracklets
