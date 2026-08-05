# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for SAMBoxMaskGenerator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trackers.core.mcbyte.masks.base import TrackletSnapshot  # noqa: E402
from trackers.core.mcbyte.masks.sam import SAMBoxMaskGenerator  # noqa: E402


class _FakeTransform:
    """Records apply_boxes_torch() calls and returns the boxes unchanged."""

    def __init__(self) -> None:
        self.apply_boxes_torch_calls: list[dict[str, Any]] = []

    def apply_boxes_torch(self, boxes: Any, original_size: tuple[int, int]) -> Any:
        self.apply_boxes_torch_calls.append({"boxes": boxes, "original_size": original_size})
        return boxes


class _FakeSamPredictor:
    """Fake SamPredictor recording set_image()/predict_torch() calls without a real checkpoint."""

    def __init__(self, num_masks: int, height: int, width: int) -> None:
        self.set_image_calls: list[np.ndarray] = []
        self.predict_torch_calls: list[dict[str, Any]] = []
        self.transform = _FakeTransform()
        self._num_masks = num_masks
        self._height = height
        self._width = width

    def set_image(self, frame: np.ndarray) -> None:
        self.set_image_calls.append(frame)

    def predict_torch(
        self,
        point_coords: Any,
        point_labels: Any,
        boxes: Any,
        multimask_output: bool,
    ) -> tuple[Any, None, None]:
        self.predict_torch_calls.append(
            {
                "point_coords": point_coords,
                "point_labels": point_labels,
                "boxes": boxes,
                "multimask_output": multimask_output,
            }
        )
        masks = torch.zeros((self._num_masks, 1, self._height, self._width), dtype=torch.bool)
        masks[:, 0, 5:10, 5:10] = True
        return masks, None, None


def test_generate_returns_empty_mask_output_when_no_tracklets() -> None:
    """Scenario: generate() with an empty tracklet list returns an empty, all-None-avg-prob MaskOutput."""
    # We only need generate()'s empty-tracklets branch. Don't call __init__(), as it loads a SAM checkpoint.
    generator = object.__new__(SAMBoxMaskGenerator)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)

    output = generator.generate(frame=frame, tracklets=[])

    assert output.masks is not None
    assert output.masks.shape == (0, 100, 120)
    assert output.masks.dtype == bool
    assert output.tracklet_mask_dict == {}
    assert output.mask_avg_prob_dict is None


def test_generate_builds_tracklet_mask_dict_and_drives_predictor_plumbing() -> None:
    """Scenario: generate() with tracklets maps tracker IDs to local mask indices via SamPredictor calls."""
    # Bypass __init__() (loads a real SAM checkpoint) and inject a fake predictor instead.
    generator = object.__new__(SAMBoxMaskGenerator)
    generator.device = torch.device("cpu")
    generator.use_amp = False
    fake_predictor = _FakeSamPredictor(num_masks=2, height=100, width=120)
    generator.predictor = fake_predictor

    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    tracklets = [
        TrackletSnapshot(tracker_id=7, xyxy=np.array([0, 0, 10, 10], dtype=np.float32)),
        TrackletSnapshot(tracker_id=3, xyxy=np.array([20, 20, 40, 40], dtype=np.float32)),
    ]

    output = generator.generate(frame=frame, tracklets=tracklets)

    assert len(fake_predictor.set_image_calls) == 1
    assert fake_predictor.set_image_calls[0] is frame

    assert len(fake_predictor.predict_torch_calls) == 1
    predict_call = fake_predictor.predict_torch_calls[0]
    assert predict_call["point_coords"] is None
    assert predict_call["point_labels"] is None
    assert predict_call["multimask_output"] is False

    assert len(fake_predictor.transform.apply_boxes_torch_calls) == 1
    assert fake_predictor.transform.apply_boxes_torch_calls[0]["original_size"] == (100, 120)

    assert output.masks is not None
    assert output.masks.shape == (2, 100, 120)
    assert output.masks.dtype == bool
    # tracker_id 7 -> local mask index 0, tracker_id 3 -> local mask index 1 (enumeration order).
    assert output.tracklet_mask_dict == {7: 0, 3: 1}
    assert output.mask_avg_prob_dict is None


def test_sam_box_mask_generator_converts_sam_masks_to_expected_shape() -> None:
    # We only need _convert_masks(). Don't call __init__(), as it loads SAM checkpoint.
    generator = object.__new__(SAMBoxMaskGenerator)

    masks = torch.zeros((2, 1, 100, 120), dtype=torch.bool)
    masks[0, 0, 10:20, 30:40] = True
    masks[1, 0, 50:70, 80:90] = True

    converted_masks = generator._convert_masks(masks)

    assert converted_masks.shape == (2, 100, 120)
    assert converted_masks.dtype == bool

    assert converted_masks[0, 10:20, 30:40].all()
    # No extra True pixels elsewhere.
    assert converted_masks[0].sum() == 10 * 10

    assert converted_masks[1, 50:70, 80:90].all()
    assert converted_masks[1].sum() == 20 * 10
