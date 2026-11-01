# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""McByte++ inspection outputs, configuration replay and failure cleanup."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import supervision as sv
from jsonargparse import ActionConfigFile

from trackers import McBytePlusPlusMaskConfig
from trackers.cli._parser import _CLIParser
from trackers.cli.inspect.mcbyteplusplus import SequenceOptions, mcbyteplusplus_inspection
from trackers.core.masks.base import MaskOutput


@pytest.mark.parametrize(
    ("fail_on_second", "emit_warnings"),
    [(False, True), (True, True), (False, False)],
    ids=["completed", "interrupted", "no_captured_warnings"],
)
def test_inspection_streams_bgr_and_saves_reusable_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fail_on_second: bool,
    emit_warnings: bool,
) -> None:
    from trackers.cli.inspect import mcbyteplusplus as module

    image_dir = tmp_path / "img1"
    image_dir.mkdir()
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    frame[:, :, 0] = 230
    for number in (2, 3):
        assert cv2.imwrite(str(image_dir / f"{number:06d}.png"), frame)
    detections = tmp_path / "det.txt"
    detections.write_text("2,999,5,6,10,12,-1,-1,-1,-1\n3,999,5,6,10,12,-1,-1,-1,-1\n")
    calls: list[dict[str, Any]] = []
    backend = SimpleNamespace(model_revision="resolved-commit", device="cpu", session_index=1, session_frame_count=0)
    hub_logger = logging.getLogger("huggingface_hub.utils._http")
    monkeypatch.setattr(hub_logger, "handlers", [logging.StreamHandler()])
    monkeypatch.setattr(hub_logger, "level", logging.WARNING)
    original_logger_filters = hub_logger.filters.copy()
    original_warning_filters = list(warnings.filters)
    auth_notice = "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN."

    class FakeTracker:
        last_mask_output = MaskOutput(masks=np.ones((1, 32, 48), dtype=bool), tracklet_mask_dict={7: 0})
        untrusted_mask_ids = frozenset({7})
        mask_manager = SimpleNamespace(mask_propagator=backend)
        reset_called = False

        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            if emit_warnings:
                warnings.warn("torch.jit.interface is deprecated", FutureWarning, stacklevel=2)
                hub_logger.warning(auth_notice)
            hub_logger.warning("Retrying model download")

        def update(self, values: sv.Detections, pixels: np.ndarray) -> sv.Detections:
            if fail_on_second and backend.session_frame_count == 1:
                raise RuntimeError("inference failed")
            if emit_warnings:
                warnings.warn("overflow encountered in matmul", RuntimeWarning, stacklevel=2)
            assert values.tracker_id is None
            assert values.confidence is not None
            assert values.confidence.tolist() == [1.0]
            np.testing.assert_array_equal(pixels, frame)
            values.tracker_id = np.array([7])
            backend.session_frame_count += 1
            return values

        def reset(self) -> None:
            FakeTracker.reset_called = True

    monkeypatch.setattr(module, "McBytePlusPlusTracker", FakeTracker)
    sequence = SequenceOptions(image_dir, detections, 2, 3, confidence_override=1.0)
    assert mcbyteplusplus_inspection(
        sequence,
        edgetam=McBytePlusPlusMaskConfig(device="cpu", revision="requested-commit"),
        output_dir=tmp_path / "output",
    ) == int(fail_on_second)
    captured = capsys.readouterr()
    assert "torch.jit.interface" not in captured.err
    assert "overflow encountered" not in captured.err
    assert "unauthenticated requests" not in captured.err
    assert "Retrying model download" in captured.err
    assert ("inference failed" in captured.err) == fail_on_second
    assert hub_logger.filters == original_logger_filters
    assert warnings.filters == original_warning_filters
    run_dir = next((tmp_path / "output").iterdir())
    assert (str(run_dir / "run.json") in captured.err) == emit_warnings
    if not fail_on_second:
        assert str(run_dir) in captured.out
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["status"] == ("failed" if fail_on_second else "completed")
    if fail_on_second:
        assert manifest["error"] == "inference failed"
    assert manifest["requested_revision"] == "requested-commit"
    assert manifest["resolved_revision"] == "resolved-commit"
    assert "transformers" in manifest["versions"]
    expected_frames = [2] if fail_on_second else [2, 3]
    assert [entry["frame"] for entry in manifest["frames"]] == expected_frames
    assert all(entry["excluded_mask_ids"] == [7] for entry in manifest["frames"])
    saved_warnings = {(entry["category"], entry["message"]) for entry in manifest["warnings"]}
    if emit_warnings:
        assert ("FutureWarning", "torch.jit.interface is deprecated") in saved_warnings
        assert ("RuntimeWarning", "overflow encountered in matmul") in saved_warnings
        assert ("HuggingFaceHubWarning", auth_notice) in saved_warnings
    else:
        assert saved_warnings == set()
    assert (run_dir / "frames" / "000002.png").is_file()
    assert (run_dir / "frames" / "000003.png").is_file() != fail_on_second
    result_rows = [line.split(",") for line in (run_dir / "results.txt").read_text().splitlines()]
    expected_frame_ids = [(number, 7) for number in expected_frames]
    assert [(int(row[0]), int(row[1])) for row in result_rows] == expected_frame_ids
    assert calls[0]["cmc_method"] == "orb"
    assert calls[0]["cmc_downscale"] == 4
    assert calls[0]["enable_mask_trust_filter"] is True
    assert FakeTracker.reset_called
    parser = _CLIParser(exit_on_error=False)
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(mcbyteplusplus_inspection)
    restored = parser.instantiate_classes(parser.parse_args(["--config", str(run_dir / "config.json")]))
    assert restored.sequence.start_frame == 2
    assert restored.edgetam.revision == "resolved-commit"
    assert restored.edgetam.device == "cpu"
    assert restored.tracker.enable_mask_trust_filter is True
    overridden = parser.instantiate_classes(
        parser.parse_args(["--config", str(run_dir / "config.json"), "--tracker.enable_mask_trust_filter", "false"])
    )
    assert overridden.tracker.enable_mask_trust_filter is False


def test_invalid_frame_range_rejected_before_model_load(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert mcbyteplusplus_inspection(SequenceOptions(tmp_path, tmp_path / "det.txt", 3, 2)) == 1
    assert "frame range" in capsys.readouterr().err
