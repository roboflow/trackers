# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections.abc import Sequence
from typing import cast

from trackers.core.botsort.tracklet import BoTSORTTracklet
from trackers.core.botsort.utils import _fuse_score
from trackers.core.botsort.utils import get_alive_tracklets as _botsort_get_alive_tracklets
from trackers.core.mcbyte.tracklet import McByteTracklet

# `_fuse_score` is re-exported from BoTSORT so the identical implementation
# lives in a single place; `tracker.py` keeps importing both symbols from this
# local module.
__all__ = ["_fuse_score", "get_alive_tracklets"]


def get_alive_tracklets(
    tracklets: Sequence[McByteTracklet],
    minimum_consecutive_frames: int,
    maximum_frames_without_update: int,
    maximum_time_without_update: float | None = None,
) -> list[McByteTracklet]:
    """Remove dead or immature lost tracklets and return alive ones.

    Thin wrapper that delegates to
    :func:`trackers.core.botsort.utils.get_alive_tracklets`. The pruning logic
    (sticky maturity, the ``<=`` lost-track budget, and the optional seconds
    budget for variable frame rates) therefore lives in a single place shared
    with BoTSORT instead of being duplicated here. ``McByteTracklet`` and
    ``BoTSORTTracklet`` both derive from ``BaseTracklet``, so the shared
    implementation only touches the common tracklet interface.

    Args:
        tracklets: McByteTracklet objects to filter.
        minimum_consecutive_frames: Number of successful updates required for
            maturity. Ignored for tracklets that already hold a real
            ``tracker_id`` (sticky maturity path).
        maximum_frames_without_update: Maximum number of frames without update
            before a track is considered dead.
        maximum_time_without_update: Optional seconds budget. When provided, it
            is used instead of the frame-count budget, enabling correct pruning
            under variable frame rates.

    Returns:
        List of alive tracklets.
    """
    return cast(
        list[McByteTracklet],
        _botsort_get_alive_tracklets(
            tracklets=cast(Sequence[BoTSORTTracklet], tracklets),
            minimum_consecutive_frames=minimum_consecutive_frames,
            maximum_frames_without_update=maximum_frames_without_update,
            maximum_time_without_update=maximum_time_without_update,
        ),
    )
