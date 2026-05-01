# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# tests/core/mcbyte/test_mcbyte_tracker.py

from __future__ import annotations

import numpy as np
import supervision as sv

from trackers.core.mcbyte.tracker import McByteTracker


def _detection(
    xyxy: tuple[float, float, float, float], conf: float = 0.9
) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([conf], dtype=np.float32),
    )


def _make_frame(h: int = 480, w: int = 640, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_mcbyte_instantiates_sets_frame_and_updates_with_sparse_opt_flow_cmc() -> None:
    """McByteTracker can run one basic CMC-enabled tracking sequence."""
    tracker = McByteTracker(
        enable_cmc=True,
        cmc_method="sparseOptFlow",
        minimum_consecutive_frames=2,
    )

    frame = _make_frame()

    for _ in range(5):
        result = tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame)

    assert len(result) == 1
    assert result.tracker_id is not None
    assert result.tracker_id[0] >= 0
    assert len(tracker.tracks) == 1
