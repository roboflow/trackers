# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import supervision as sv

from trackers.core.mcbyte.masks.base import MaskOutput, TrackletSnapshot
from trackers.core.mcbyte.tracker import McByteTracker


class SpyMaskManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_updated_masks(
        self,
        frame: np.ndarray,
        previous_frame: np.ndarray | None,
        previous_tracklets: list[TrackletSnapshot],
        new_tracklets: list[TrackletSnapshot] | None = None,
        removed_tracklet_ids: list[int] | None = None,
    ) -> MaskOutput | None:
        self.calls.append(
            {
                "frame": frame,
                "previous_frame": previous_frame,
                "previous_tracklets": previous_tracklets,
                "new_tracklets": [] if new_tracklets is None else new_tracklets,
                "removed_tracklet_ids": [] if removed_tracklet_ids is None else removed_tracklet_ids,
            }
        )
        return None

    def reset(self) -> None:
        pass


def _detection(xyxy: tuple[float, float, float, float], conf: float = 0.9) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([conf], dtype=np.float32),
    )


def _make_frame(h: int = 480, w: int = 640, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_mcbyte_instantiates_and_updates_with_frame_and_sparse_opt_flow_cmc_returns_ids() -> None:
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


def test_mcbyte_reset_clears_mask_state() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=True,
        minimum_consecutive_frames=1,
    )

    frame = _make_frame()

    tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame)
    tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame)

    assert tracker._previous_frame is not None
    assert len(tracker._previous_tracklets) == 1
    assert tracker._last_mask_output is not None

    tracker.reset()

    assert tracker._previous_frame is None
    assert tracker._previous_tracklets == []
    assert tracker._last_mask_output is None


def test_mcbyte_does_not_store_previous_frame_without_mask_manager() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
        minimum_consecutive_frames=1,
    )

    frame = _make_frame()

    tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame)

    assert tracker._previous_frame is None
    assert tracker._previous_tracklets == []


def test_mcbyte_passes_new_tracklets_to_mask_manager_on_next_frame() -> None:
    mask_manager = SpyMaskManager()
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
        mask_manager=mask_manager,  # type: ignore[arg-type]
        minimum_consecutive_frames=1,
    )

    frame = _make_frame()

    tracker.update(_detection((100.0, 100.0, 200.0, 200.0)), frame)
    tracker.update(_detection((102.0, 102.0, 202.0, 202.0)), frame)

    assert len(mask_manager.calls) == 2

    first_call = mask_manager.calls[0]
    assert first_call["previous_frame"] is None
    assert first_call["previous_tracklets"] == []
    assert first_call["new_tracklets"] == []
    assert first_call["removed_tracklet_ids"] == []

    second_call = mask_manager.calls[1]
    assert second_call["previous_frame"] is frame

    previous_tracklets = second_call["previous_tracklets"]
    new_tracklets = second_call["new_tracklets"]

    assert isinstance(previous_tracklets, list)
    assert isinstance(new_tracklets, list)
    assert len(previous_tracklets) == 1
    assert len(new_tracklets) == 1
    assert previous_tracklets[0].tracker_id == new_tracklets[0].tracker_id
    assert second_call["removed_tracklet_ids"] == []


def test_mcbyte_mask_lifecycle_keeps_missing_tracklet_until_explicit_removal() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=True,
        mask_manager=None,
        minimum_consecutive_frames=1,
    )

    frame = _make_frame()

    visible_result = sv.Detections(
        xyxy=np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
    )
    visible_result.tracker_id = np.array([7], dtype=int)

    empty_result = sv.Detections.empty()
    empty_result.tracker_id = np.array([], dtype=int)

    tracker._store_previous_mask_inputs(
        frame=frame,
        detections=visible_result,
        removed_tracklet_ids=[],
    )

    assert tracker._previous_new_tracklets[0].tracker_id == 7
    assert tracker._mask_tracklet_ids == {7}

    tracker._store_previous_mask_inputs(
        frame=frame,
        detections=empty_result,
        removed_tracklet_ids=[],
    )

    assert tracker._previous_new_tracklets == []
    assert tracker._previous_removed_tracklet_ids == []
    assert tracker._mask_tracklet_ids == {7}

    tracker._store_previous_mask_inputs(
        frame=frame,
        detections=empty_result,
        removed_tracklet_ids=[7],
    )

    assert tracker._previous_new_tracklets == []
    assert tracker._previous_removed_tracklet_ids == [7]
    assert tracker._mask_tracklet_ids == set()
