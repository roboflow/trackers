# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""End-to-end smoke test for BoT-SORT with a real ``reid`` encoder.

Exercises the ``trackers`` -> ``reid`` boundary once, without re-testing the
``reid`` internals (those live in the ``reid`` package's own suite). Requires
the ``trackers[reid]`` extra; skipped when ``roboflow-reid`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

reid = pytest.importorskip("reid", reason="requires the optional trackers[reid] extra")

from trackers.core.botsort.tracker import BoTSORTTracker  # noqa: E402


@pytest.mark.integration
def test_botsort_with_real_reid_model_runs_over_frames() -> None:
    reid_model = reid.ReIDModel.from_pretrained(architecture="osnet_x0_25", device="cpu")

    tracker = BoTSORTTracker(enable_cmc=False, reid_model=reid_model)

    rng = np.random.default_rng(0)
    box = np.array([30.0, 30.0, 70.0, 90.0], dtype=np.float32)
    for _ in range(3):
        frame = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        detections = sv.Detections(
            xyxy=box[None, :].copy(),
            confidence=np.array([0.9], dtype=np.float32),
        )
        result = tracker.update(detections, frame=frame)
        assert result.tracker_id is not None
        box = box + np.array([2.0, 1.0, 2.0, 1.0], dtype=np.float32)

    assert len(tracker.tracks) == 1
    bank = tracker.tracks[0].feature_bank
    assert bank is not None and bank.is_initialized
