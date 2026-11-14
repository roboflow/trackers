# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""SoccerNet benchmark outputs, timing boundaries, configuration and failure recovery."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import supervision as sv

from trackers.cli._parser import _CLIParser
from trackers.cli.benchmark import BENCHMARK_COMMANDS
from trackers.cli.benchmark import mcbyteplusplus as module
from trackers.core.mcbyteplusplus.tracker import McBytePlusPlusMaskConfig


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Two short sequences; their last frame has no detections and no GT is provided."""
    root = tmp_path / "test"
    frame = np.full((96, 160, 3), [230, 10, 20], dtype=np.uint8)
    for name, fps in (("SNMOT-116", 25), ("SNMOT-117", 20)):
        sequence = root / name
        (sequence / "img1").mkdir(parents=True)
        (sequence / "det").mkdir()
        (sequence / "seqinfo.ini").write_text(f"[Sequence]\nseqLength=3\nframeRate={fps}\n")
        (sequence / "det/det.txt").write_text("1,999,5,6,10,12,-1,-1,-1,-1\n2,999,5,6,10,12,-1,-1,-1,-1\n")
        for number in range(1, 4):
            assert cv2.imwrite(str(sequence / "img1" / f"{number:06d}.png"), frame)
    return root


def _install_tracker(monkeypatch: pytest.MonkeyPatch, error: BaseException | None = None) -> list[Any]:
    """Replace model inference while retaining the real readers, writers and renderer."""
    instances: list[Any] = []

    class FakeTracker:
        last_mask_output = None
        untrusted_mask_ids: frozenset[int] = frozenset()

        def __init__(self, **kwargs: Any) -> None:
            self.options = kwargs
            self.mask_manager = SimpleNamespace(
                mask_propagator=SimpleNamespace(device="cpu", model_revision="resolved-commit")
            )
            self.frames: list[np.ndarray] = []
            self.reset_called = False
            instances.append(self)

        def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
            if error is not None and self is instances[0] and len(self.frames) == 1:
                raise error
            self.frames.append(frame.copy())
            assert detections.tracker_id is None
            if len(detections):
                np.testing.assert_array_equal(detections.confidence, np.ones(len(detections)))
            detections.tracker_id = np.full(len(detections), 7)
            return detections

        def reset(self) -> None:
            self.reset_called = True

    monkeypatch.setattr(module, "McBytePlusPlusTracker", FakeTracker)
    return instances


@pytest.mark.parametrize("visuals", [True, False], ids=["video_and_frames", "text_only"])
def test_benchmark_saves_complete_sequences_and_replayable_config(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, visuals: bool
) -> None:
    """Preserve frame/ID/color conventions, include empty frames, and reset each sequence."""
    instances = _install_tracker(monkeypatch)
    if not visuals:

        def unexpected_render(*args: Any) -> None:
            pytest.fail("Text-only benchmarks must not render overlays.")

        monkeypatch.setattr(module, "visualize_mcbyteplusplus", unexpected_render)
    output_root = tmp_path / "runs"
    assert (
        module.benchmark_command(
            dataset,
            edgetam=McBytePlusPlusMaskConfig(device="cpu"),
            output_root=output_root,
            save_video=visuals,
            save_frames=visuals,
        )
        == 0
    )
    run = next(output_root.iterdir())
    report = json.loads((run / "run.json").read_text())
    assert report["status"] == "completed"
    assert report["all_discovered_sequences_completed"] is True
    assert report["totals"]["completed_frames"] == 6
    assert report["totals"]["tracking_fps"] > 0
    assert len(instances) == 2
    assert [instance.options["frame_rate"] for instance in instances] == [25, 20]
    for instance, sequence in zip(instances, report["sequences"]):
        assert instance.reset_called
        assert len(instance.frames) == 3
        np.testing.assert_array_equal(instance.frames[0][0, 0], [230, 10, 20])
        assert instance.options["enable_mask_manager"] is True
        assert instance.options["enable_mask_trust_filter"] is True
        assert instance.options["cmc_downscale"] == 4
        assert sequence["resolved_revision"] == "resolved-commit"
        assert sequence["processed_frames"] == sequence["expected_frames"] == 3
        rows = [row.split(",") for row in (run / "results" / f"{sequence['name']}.txt").read_text().splitlines()]
        expected_frame_ids = [(1, 7), (2, 7)]
        assert [(int(row[0]), int(row[1])) for row in rows] == expected_frame_ids
        assert all(len(row) == 10 for row in rows)
        np.testing.assert_allclose([float(value) for value in rows[0][2:7]], [5, 6, 10, 12, 1])
        if visuals:
            cap = cv2.VideoCapture(str(run / "videos" / f"{sequence['name']}.mp4"))
            try:
                assert cap.isOpened()
                assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 3
                assert cap.get(cv2.CAP_PROP_FPS) == sequence["frame_rate"]
                decoded, pixels = cap.read()
                assert decoded and pixels is not None
            finally:
                cap.release()
            assert len(list((run / "frames" / sequence["name"]).glob("*.jpg"))) == 3
        else:
            assert not (run / "videos").exists()
            assert not (run / "frames").exists()
    assert not list(run.rglob("*.partial*"))
    assert BENCHMARK_COMMANDS["mcbyteplusplus"] is module.benchmark_command
    parser = _CLIParser(exit_on_error=False)
    parser.add_argument("--config", action="config")
    parser.add_function_arguments(module.benchmark_command)
    restored = parser.instantiate_classes(
        parser.parse_args(
            [
                "--config",
                str(run / "config.json"),
                "--max_frames",
                "2",
                "--cmc.downscale",
                "2",
                "--tracker.enable_mask_refresh",
                "false",
            ]
        )
    )
    assert restored.edgetam.revision == "resolved-commit"
    assert restored.edgetam.device == "cpu"
    assert restored.max_frames == 2
    assert restored.cmc.downscale == 2
    assert restored.tracker.enable_mask_refresh is False


@pytest.mark.parametrize("interrupt", [False, True], ids=["failure_continues", "interrupt_stops"])
def test_failed_sequence_retains_partial_outputs_and_reports(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool
) -> None:
    """A partial result cannot masquerade as a completed sequence or inflate pooled FPS."""
    error = KeyboardInterrupt() if interrupt else RuntimeError("inference failed")
    instances = _install_tracker(monkeypatch, error)
    output_root = tmp_path / "runs"
    assert module.benchmark_command(dataset, output_root=output_root, save_video=False) == (130 if interrupt else 1)
    run = next(output_root.iterdir())
    report = json.loads((run / "run.json").read_text())
    assert report["status"] == ("interrupted" if interrupt else "failed")
    assert report["all_discovered_sequences_completed"] is False
    assert (run / "results/SNMOT-116.txt.partial").is_file()
    assert not (run / "results/SNMOT-116.txt").exists()
    assert (run / "results/SNMOT-117.txt").exists() is not interrupt
    assert report["totals"]["completed_frames"] == (0 if interrupt else 3)
    assert all(instance.reset_called for instance in instances)
    if not interrupt:
        assert "inference failed" in (run / "run.log").read_text()


def test_subset_and_frame_limit_are_not_reported_as_full_dataset(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances = _install_tracker(monkeypatch)
    output_root = tmp_path / "runs"
    original_visualize = module.visualize_mcbyteplusplus

    def warn_and_visualize(*args: Any) -> np.ndarray:
        warnings.warn("example numerical warning", RuntimeWarning, stacklevel=2)
        return original_visualize(*args)

    monkeypatch.setattr(module, "visualize_mcbyteplusplus", warn_and_visualize)
    assert (
        module.benchmark_command(
            dataset,
            sequences=["SNMOT-117"],
            max_frames=1,
            output_root=output_root,
        )
        == 0
    )
    run = next(output_root.iterdir())
    report = json.loads((run / "run.json").read_text())
    assert report["status"] == "completed"
    assert report["selected_sequences"] == ["SNMOT-117"]
    assert report["all_discovered_sequences_completed"] is False
    assert report["totals"]["completed_frames"] == 1
    assert len(instances) == 1
    assert report["sequences"][0]["warnings"][0]["message"] == "example numerical warning"
    assert (run / "frames/SNMOT-117/000001.jpg").is_file()
    assert not (run / "videos").exists()


@pytest.mark.parametrize("problem", ["missing_metadata", "unknown_sequence", "bad_limit", "bad_rate"])
def test_invalid_dataset_or_options_fail_before_model_loading(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    instances = _install_tracker(monkeypatch)
    kwargs: dict[str, Any] = {}
    if problem == "missing_metadata":
        (dataset / "SNMOT-116/seqinfo.ini").unlink()
    elif problem == "unknown_sequence":
        kwargs["sequences"] = ["SNMOT-999"]
    elif problem == "bad_limit":
        kwargs["max_frames"] = 0
    else:
        (dataset / "SNMOT-116/seqinfo.ini").write_text("[Sequence]\nseqLength=3\nframeRate=nan\n")
    assert module.benchmark_command(dataset, output_root=tmp_path / "runs", **kwargs) == 1
    assert not instances
    assert not (tmp_path / "runs").exists()


def test_pooled_fps_uses_total_frames_and_time_for_completed_sequences() -> None:
    reports = [
        {"status": "completed", "processed_frames": 100, "tracking_seconds": 2.0},
        {"status": "completed", "processed_frames": 20, "tracking_seconds": 4.0},
        {"status": "failed", "processed_frames": 50, "tracking_seconds": 1.0},
    ]
    totals = module._summarize(reports, elapsed=12.0)
    assert totals["completed_frames"] == 120
    assert totals["tracking_fps"] == 20.0
    assert totals["end_to_end_fps"] == 10.0
    assert totals["failed_sequences"] == 1


def test_tracking_timer_includes_pending_inference_but_excludes_rendering(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A controlled clock separates model loading, update work and visual output."""
    _install_tracker(monkeypatch)
    tracker_class = module.McBytePlusPlusTracker
    original_init = tracker_class.__init__
    original_update = tracker_class.update
    clock = {"seconds": 0.0, "pending": 0.0}

    def load_tracker(self: Any, **kwargs: Any) -> None:
        original_init(self, **kwargs)
        clock["seconds"] += 7.0

    def update(self: Any, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        clock["seconds"] += 2.0
        clock["pending"] = 3.0
        return original_update(self, detections, frame)

    def synchronize(device: str) -> None:
        clock["seconds"] += clock["pending"]
        clock["pending"] = 0.0

    def render(frame: np.ndarray, *args: Any) -> np.ndarray:
        clock["seconds"] += 11.0
        return frame

    monkeypatch.setattr(tracker_class, "__init__", load_tracker)
    monkeypatch.setattr(tracker_class, "update", update)
    monkeypatch.setattr(module, "_synchronize", synchronize)
    monkeypatch.setattr(module, "perf_counter", lambda: clock["seconds"])
    monkeypatch.setattr(module, "visualize_mcbyteplusplus", render)
    output_root = tmp_path / "runs"
    assert module.benchmark_command(dataset, sequences=["SNMOT-116"], output_root=output_root) == 0
    report = json.loads(next(output_root.glob("*/run.json")).read_text())
    # Three updates cost 5 seconds each; three renders add 33 seconds and loading adds 7.
    assert report["totals"]["tracking_seconds"] == 15.0
    assert report["totals"]["tracking_fps"] == pytest.approx(3 / 15)
    assert report["totals"]["end_to_end_seconds"] == 55.0
    assert report["sequences"][0]["model_load_seconds"] == 7.0


def test_accelerator_timing_waits_for_the_selected_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU does not wait; CUDA and MPS must finish queued work before timing ends."""
    calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda device: calls.append(device)),
        mps=SimpleNamespace(synchronize=lambda: calls.append("mps")),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    for device in ("cpu", "cuda:1", "mps"):
        module._synchronize(device)
    assert calls == ["cuda:1", "mps"]
