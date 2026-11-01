# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections.abc import Sequence

from trackers.core.mcbyteplusplus.tracklet import McBytePlusPlusTracklet
from trackers.utils.base_tracklet import BaseTracklet


def _get_alive_tracklets(
    tracklets: Sequence[McBytePlusPlusTracklet],
    minimum_consecutive_frames: int,
    maximum_frames_without_update: int,
    maximum_time_without_update: float | None = None,
) -> list[McBytePlusPlusTracklet]:
    """Remove dead or immature lost tracklets and return alive ones.

    A tracklet is kept if it is within ``maximum_frames_without_update`` **and**
    it satisfies at least one liveness condition:

    - it was updated this frame (``time_since_update == 0``), OR
    - it holds a real tracker ID (``tracker_id != -1``) — sticky maturity:
      a track instant-activated on the first frame is preserved on a miss
      rather than deleted, OR
    - it has accumulated at least ``minimum_consecutive_frames`` successful
      updates.

    Args:
        tracklets: List of McBytePlusPlusTracklet objects.
        minimum_consecutive_frames: Number of successful updates required for
            maturity. Ignored for tracklets that already hold a real
            ``tracker_id`` (sticky maturity path).
        maximum_frames_without_update: Maximum number of frames without update
            before a track is considered dead.
        maximum_time_without_update: Seconds budget. When provided, this
            criterion is used **instead of** the frame-count budget, enabling
            correct pruning under variable frame rates.
    Returns:
        List of alive tracklets.
    """
    alive_tracklets = []
    for tracker in tracklets:
        # Maturity is sticky: number_of_successful_updates is cumulative and
        # never decremented, but instant first-frame activation assigns
        # tracker_id before nsu reaches minimum_consecutive_frames. The
        # tracker_id != -1 guard keeps those instant-activated tracks alive
        # on a miss without waiting for nsu to catch up.
        is_mature = tracker.tracker_id != -1 or tracker.number_of_successful_updates >= minimum_consecutive_frames
        is_active = tracker.time_since_update == 0
        within_budget = BaseTracklet.within_lost_track_budget(
            tracker,
            maximum_frames_without_update=maximum_frames_without_update,
            maximum_time_without_update=maximum_time_without_update,
        )
        if within_budget and (is_mature or is_active):
            alive_tracklets.append(tracker)
    return alive_tracklets
