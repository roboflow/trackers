# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
import supervision as sv
from pytest import MonkeyPatch

from trackers.core.mcbyte.mask_manager import MaskManager
from trackers.core.mcbyte.masks.base import MaskOutput, TrackletSnapshot
from trackers.core.mcbyte.masks.dummy import (
    DummyBoxMaskGenerator,
    DummyIdentityMaskPropagator,
)
from trackers.core.mcbyte.tracker import McByteMaskConfig, McByteTracker
from trackers.core.mcbyte.tracklet import McByteTracklet


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


def _tracklet_with_id(
    tracker_id: int,
    xyxy: tuple[float, float, float, float],
) -> McByteTracklet:
    """Create a McByte tracklet with a stable tracker ID for association tests."""
    tracklet = McByteTracklet(
        initial_bbox=np.array(xyxy, dtype=np.float32),
    )
    tracklet.tracker_id = tracker_id
    return tracklet


def _make_frame(h: int = 480, w: int = 640, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _dummy_mask_manager() -> MaskManager:
    """Create a lightweight mask manager for tracker unit tests."""
    return MaskManager(
        mask_generator=DummyBoxMaskGenerator(),
        mask_propagator=DummyIdentityMaskPropagator(),
    )


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
        mask_manager=_dummy_mask_manager(),
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
        mask_manager=_dummy_mask_manager(),
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


def test_mcbyte_mask_conditioned_association_combines_locked_and_reduced_matches() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
    )

    tracklets = [
        _tracklet_with_id(10, (0.0, 0.0, 5.0, 5.0)),
        _tracklet_with_id(20, (5.0, 5.0, 10.0, 10.0)),
        _tracklet_with_id(30, (10.0, 10.0, 15.0, 15.0)),
    ]
    detection_boxes = np.array(
        [
            [0.0, 0.0, 5.0, 5.0],
            [5.0, 5.0, 10.0, 10.0],
            [10.0, 10.0, 15.0, 15.0],
        ],
        dtype=np.float32,
    )

    similarity = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.7, 0.6],
            [0.0, 0.6, 0.7],
        ],
        dtype=np.float32,
    )

    matched, unmatched_tracks, unmatched_detections = tracker._get_mask_conditioned_associated_indices(
        similarity_matrix=similarity,
        raw_iou_similarity=similarity,
        tracklets=tracklets,
        detection_boxes=detection_boxes,
        min_similarity_thresh=0.5,
    )

    # Locked match: (0,0). Reduced matches: (1,1), (2,2) - mapped back from (0,0), (1,1)
    assert matched == [
        (0, 0),
        (1, 1),
        (2, 2),
    ]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_mcbyte_mask_conditioned_association_remaps_unmatched_indices() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
    )

    tracklets = [
        _tracklet_with_id(10, (0.0, 0.0, 5.0, 5.0)),
        _tracklet_with_id(20, (5.0, 5.0, 10.0, 10.0)),
        _tracklet_with_id(30, (10.0, 10.0, 15.0, 15.0)),
    ]
    detection_boxes = np.array(
        [
            [0.0, 0.0, 5.0, 5.0],
            [5.0, 5.0, 10.0, 10.0],
            [10.0, 10.0, 15.0, 15.0],
        ],
        dtype=np.float32,
    )

    similarity = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.4, 0.3],
            [0.0, 0.2, 0.1],
        ],
        dtype=np.float32,
    )

    matched, unmatched_tracks, unmatched_detections = tracker._get_mask_conditioned_associated_indices(
        similarity_matrix=similarity,
        raw_iou_similarity=similarity,
        tracklets=tracklets,
        detection_boxes=detection_boxes,
        min_similarity_thresh=0.5,
    )

    assert matched == [(0, 0)]
    assert unmatched_tracks == [1, 2]  # mapped back from [0, 1]
    assert unmatched_detections == [1, 2]  # mapped back from [0, 1]


def test_mcbyte_mask_conditioned_association_changes_ambiguous_assignment() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
    )

    tracklets = [
        _tracklet_with_id(10, (0.0, 0.0, 5.0, 5.0)),
        _tracklet_with_id(20, (5.0, 5.0, 10.0, 10.0)),
    ]
    detection_boxes = np.array(
        [
            [0.0, 0.0, 5.0, 5.0],
            [5.0, 5.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )

    similarity = np.array(
        [
            [0.6, 0.7],
            [0.7, 0.6],
        ],
        dtype=np.float32,
    )

    matches_without_masks, _, _ = tracker._get_mask_conditioned_associated_indices(
        similarity_matrix=similarity,
        raw_iou_similarity=similarity,
        tracklets=tracklets,
        detection_boxes=detection_boxes,
        min_similarity_thresh=0.5,
    )

    masks = np.zeros((2, 10, 10), dtype=bool)
    # it will result in fill-ratio bonus of 1.0 for first detection (0.6 + 1.0 = 1.6)
    masks[0, 0:5, 0:5] = True
    # it will result in fill-ratio bonus of 1.0 for second detection (0.6 + 1.0 = 1.6)
    masks[1, 5:10, 5:10] = True

    tracker._last_mask_output = MaskOutput(
        masks=masks,
        tracklet_mask_dict={
            10: 0,
            20: 1,
        },
        mask_avg_prob_dict={
            10: 0.9,
            20: 0.9,
        },
    )

    matches_with_masks, unmatched_tracks, unmatched_detections = tracker._get_mask_conditioned_associated_indices(
        similarity_matrix=similarity,
        raw_iou_similarity=similarity,
        tracklets=tracklets,
        detection_boxes=detection_boxes,
        min_similarity_thresh=0.5,
    )

    assert matches_without_masks == [(0, 1), (1, 0)]
    assert matches_with_masks == [(0, 0), (1, 1)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_mcbyte_mask_conditioned_association_rescues_isolated_pair_when_enabled() -> None:
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=False,
        enable_isolated_mask_matching=True,
    )

    tracklets = [
        _tracklet_with_id(10, (0.0, 0.0, 10.0, 10.0)),
    ]
    detection_boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )

    tracker._last_mask_output = MaskOutput(
        masks=np.ones((1, 10, 10), dtype=bool),
        tracklet_mask_dict={10: 0},
        mask_avg_prob_dict={10: 0.9},
    )

    matched, unmatched_tracks, unmatched_detections = tracker._get_mask_conditioned_associated_indices(
        similarity_matrix=np.array([[0.2]], dtype=np.float32),
        raw_iou_similarity=np.array([[0.2]], dtype=np.float32),
        tracklets=tracklets,
        detection_boxes=detection_boxes,
        min_similarity_thresh=0.5,
    )

    # Without isolation, match would be empty, unmatched_tracks would be [0],
    # unmatched_detections would be [0]
    assert matched == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_mcbyte_builds_real_mask_pipeline_when_enabled(
    monkeypatch: MonkeyPatch,
) -> None:
    created: dict[str, object] = {}

    class FakeSAMBoxMaskGenerator:
        def __init__(
            self,
            checkpoint_path: str | None = None,
            model_type: str = "vit_b",
            device: str = "cpu",
        ) -> None:
            created["sam"] = {
                "checkpoint_path": checkpoint_path,
                "model_type": model_type,
                "device": device,
            }

    class FakeCutieMaskPropagator:
        def __init__(
            self,
            weights_path: str | None = None,
            model_type: str = "base-mega",
            config_path: str | None = None,
            config_name: str = "eval_config",
            device: str = "cuda",
            use_amp: bool = True,
        ) -> None:
            created["cutie"] = {
                "weights_path": weights_path,
                "model_type": model_type,
                "config_path": config_path,
                "config_name": config_name,
                "device": device,
                "use_amp": use_amp,
            }

        def reset(self) -> None:
            pass

    # mcbyte_tracker_module.SAMBoxMaskGenerator and
    # mcbyte_tracker_module.CutieMaskPropagator are imported locally in
    # _build_default_mask_manager() in tracker.py
    fake_sam_module = ModuleType("trackers.core.mcbyte.masks.sam")
    fake_sam_module.SAMBoxMaskGenerator = FakeSAMBoxMaskGenerator  # type: ignore[attr-defined]

    fake_cutie_module = ModuleType("trackers.core.mcbyte.masks.cutie")
    fake_cutie_module.CutieMaskPropagator = FakeCutieMaskPropagator  # type: ignore[attr-defined]

    monkeypatch.setitem(
        sys.modules,
        "trackers.core.mcbyte.masks.sam",
        fake_sam_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "trackers.core.mcbyte.masks.cutie",
        fake_cutie_module,
    )

    # "cuda:1" only for testing of the passed values. No actual CUDA device is used,
    # because the classes are fake and do not initialize PyTorch or allocate anything.
    # enable_mask_manager=True must lead to creating MaskManager with
    # SAMBoxMaskGenerator and CutieMaskPropagator (being replaced by
    # FakeSAMBoxMaskGenerator and FakeCutieMaskPropagator)
    tracker = McByteTracker(
        enable_cmc=False,
        enable_mask_manager=True,
        mask_config=McByteMaskConfig(
            device="cuda:1",
            sam_model_type="vit_b",
            cutie_model_type="base-mega",
            cutie_config_name="eval_config",
            cutie_use_amp=False,
            mask_creation_bbox_overlap_threshold=0.7,
        ),
    )

    # Verify a real MaskManager was created
    assert isinstance(tracker.mask_manager, MaskManager)

    # Verify SAM received the right parameters (via McByteMaskConfig at McByteTracker)
    assert created["sam"] == {
        "checkpoint_path": None,
        "model_type": "vit_b",
        "device": "cuda:1",
    }

    # Verify Cutie received the right parameters (via McByteMaskConfig at McByteTracker)
    assert created["cutie"] == {
        "weights_path": None,
        "model_type": "base-mega",
        "config_path": None,
        "config_name": "eval_config",
        "device": "cuda:1",
        "use_amp": False,
    }

    # Verify the MaskManager-specific threshold
    assert tracker.mask_manager.mask_creation_bbox_overlap_threshold == 0.7


def test_mcbyte_uses_custom_mask_manager_without_real_model_construction() -> None:
    custom_manager = _dummy_mask_manager()

    tracker = McByteTracker(
        enable_cmc=False,
        mask_manager=custom_manager,
    )

    # There is some conditional flow in the MaskManager, hence we are ensuring the one
    # below here.
    assert tracker.mask_manager is custom_manager


def test_mcbyte_rejects_mask_config_with_custom_manager() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be used together",
    ):
        McByteTracker(
            enable_cmc=False,
            mask_manager=_dummy_mask_manager(),
            mask_config=McByteMaskConfig(),
        )


def test_mcbyte_rejects_unused_mask_config() -> None:
    with pytest.raises(
        ValueError,
        match="requires enable_mask_manager=True",
    ):
        McByteTracker(
            enable_cmc=False,
            enable_mask_manager=False,
            mask_config=McByteMaskConfig(),
        )
