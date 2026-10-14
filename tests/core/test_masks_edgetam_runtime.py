# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Exercise the optional model boundary independently of lifecycle fakes."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from trackers.core.masks.base import TrackletSnapshot
from trackers.core.masks.edgetam import EdgeTAMMaskBackend, _EdgeTAMRuntime


@pytest.mark.parametrize("dependency", ["torch", "transformers"])
def test_missing_dependency_explains_how_to_install(monkeypatch: pytest.MonkeyPatch, dependency: str) -> None:
    """Constructing EdgeTAM without its optional dependencies names the needed extra."""
    monkeypatch.setitem(sys.modules, dependency, None)
    with pytest.raises(ImportError, match=r"pip install 'trackers\[edgetam\]'"):
        EdgeTAMMaskBackend()


@pytest.fixture
def runtime() -> _EdgeTAMRuntime:
    """Use real tensor post-processing boundaries and a recording model/processor."""
    torch = pytest.importorskip("torch")
    runtime = object.__new__(_EdgeTAMRuntime)
    runtime._torch = torch
    runtime.device = torch.device("cpu")
    runtime._mask_threshold = 0.0
    runtime._model = MagicMock()
    runtime._processor = MagicMock()
    return runtime


def test_decoder_preserves_returned_ids_overlap_and_single_pixel_axis(runtime: _EdgeTAMRuntime) -> None:
    """Do not assume prompt order or squeeze away a spatial dimension of length one."""
    torch = runtime._torch
    runtime._model.return_value = SimpleNamespace(object_ids=[300, 0], pred_masks="logits")
    runtime._processor.post_process_masks.return_value = [torch.tensor([[[[2.0, 3.0, 0.0]]], [[[-1.0, 4.0, 2.0]]]])]
    session = SimpleNamespace(video_height=1, video_width=3)
    masks = runtime.infer(session, 7)
    expected_masks = {300: [[True, True, False]], 0: [[False, True, True]]}
    assert list(masks) == [300, 0]
    for tracker_id, expected in expected_masks.items():
        np.testing.assert_array_equal(masks[tracker_id], expected, err_msg=f"Wrong mask for tracklet {tracker_id}")
    assert (masks[300] & masks[0]).any()
    runtime._model.assert_called_once_with(inference_session=session, frame_idx=7)
    runtime._processor.post_process_masks.assert_called_once_with(["logits"], original_sizes=[[1, 3]], binarize=False)


def test_streamed_frame_is_preprocessed_and_forwarded(runtime: _EdgeTAMRuntime) -> None:
    """New images enter with an explicit local index; stored replay omits frame."""
    torch = runtime._torch
    pixels = torch.zeros((1, 3, 4, 4))
    runtime._processor.return_value.to.return_value = SimpleNamespace(pixel_values=pixels)
    runtime._model.return_value = SimpleNamespace(object_ids=[0], pred_masks="logits")
    runtime._processor.post_process_masks.return_value = [torch.ones((1, 1, 2, 3))]
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    session = SimpleNamespace(video_height=2, video_width=3)
    output = runtime.infer(session, 8, frame)
    assert output[0].shape == (2, 3)
    assert runtime._model.call_args.kwargs["frame_idx"] == 8
    assert torch.equal(runtime._model.call_args.kwargs["frame"], pixels[0])
    assert runtime._processor.call_args.kwargs["images"] is frame


def test_prompts_submit_complete_object_batch(runtime: _EdgeTAMRuntime) -> None:
    """Keep every object in the same processor call so no prompt IDs are replaced."""
    session = SimpleNamespace(video_height=8, video_width=12)
    snapshots = [TrackletSnapshot(20, np.array([1, 2, 3, 4])), TrackletSnapshot(0, np.array([5, 1, 8, 7]))]
    expected_box_batch = [[[1, 2, 3, 4], [5, 1, 8, 7]]]
    runtime.prompt_boxes(session, snapshots)
    prompt = runtime._processor.add_inputs_to_inference_session
    assert prompt.call_count == 1
    assert prompt.call_args.kwargs["obj_ids"] == [20, 0]
    assert prompt.call_args.kwargs["input_boxes"] == expected_box_batch
    assert prompt.call_args.kwargs["original_size"] == (8, 12)
    masks = {20: np.eye(8, 12, dtype=bool), 0: np.ones((8, 12), dtype=bool)}
    expected_mask_batch = [masks[0], masks[20]]
    runtime.prompt_masks(session, 7, masks)
    assert prompt.call_count == 2
    assert prompt.call_args.kwargs["obj_ids"] == [0, 20]
    assert prompt.call_args.kwargs["frame_idx"] == 7
    np.testing.assert_array_equal(prompt.call_args.kwargs["input_masks"], expected_mask_batch)


def test_decoder_rejects_duplicate_ids(runtime: _EdgeTAMRuntime) -> None:
    """Duplicate upstream IDs cannot silently overwrite masks in a dictionary."""
    runtime._model.return_value = SimpleNamespace(object_ids=[0, 0], pred_masks="logits")
    runtime._processor.post_process_masks.return_value = [runtime._torch.ones((2, 1, 2, 3))]
    with pytest.raises(RuntimeError, match="inconsistent"):
        runtime.infer(SimpleNamespace(video_height=2, video_width=3), 0)
