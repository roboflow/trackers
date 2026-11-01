# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Track retention matches McByte's maturity and lost-budget contract."""

import numpy as np
import pytest

from trackers.core.mcbyteplusplus.tracklet import McBytePlusPlusTracklet
from trackers.core.mcbyteplusplus.utils import _get_alive_tracklets


@pytest.mark.parametrize(
    "misses,seconds,updates,tracker_id,time_budget,alive",
    [
        (0, 0.0, 1, -1, None, True),
        (1, 0.04, 1, -1, None, False),
        (1, 0.04, 2, -1, None, True),
        (1, 0.04, 1, 7, None, True),
        (2, 0.08, 2, 7, None, True),
        (3, 0.12, 2, 7, None, False),
        (10, 0.08, 2, 7, 0.08, True),
        (1, 0.09, 2, 7, 0.08, False),
    ],
)
def test_track_retention_budget_and_sticky_maturity(misses, seconds, updates, tracker_id, time_budget, alive) -> None:
    track = McBytePlusPlusTracklet(np.array([0.0, 0.0, 10.0, 20.0]))
    track.time_since_update = misses
    track.time_since_update_seconds = seconds
    track.number_of_successful_updates = updates
    track.tracker_id = tracker_id
    result = _get_alive_tracklets(
        [track], minimum_consecutive_frames=2, maximum_frames_without_update=2, maximum_time_without_update=time_budget
    )
    assert result == ([track] if alive else [])
