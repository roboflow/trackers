# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Optional CLI appearance configuration and streaming historical results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import supervision as sv

from trackers.cli import track as track_module
from trackers.cli._mcbyteplusplus import CMCOptions, _reid_provenance, visualize_mcbyteplusplus
from trackers.cli._parser import _CLIParser
from trackers.cli._progress import _SourceInfo
from trackers.cli._reid_output import _ReIDResultWriter
from trackers.cli.benchmark import mcbyteplusplus as benchmark_module
from trackers.cli.inspect import mcbyteplusplus as inspect_module
from trackers.cli.track import ReIDModelConfig, ShowOptions
from trackers.core.masks.base import MaskOutput
from trackers.core.mcbyteplusplus.tracker import McBytePlusPlusTracker
from trackers.reid.long_term import LongTermReID
from trackers.reid.long_term_config import LongTermReIDConfig


class _Encoder:
    device = "cpu"
    preprocessing = SimpleNamespace(to_dict=lambda: {"input_size": [256, 128], "to_rgb": True})

    def extract_embeddings(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
        return np.array([[1, 0] if box[0] < 100 else [0, 1] for box in detections.xyxy], dtype=np.float32)


def _detections(identities: list[int], *, class_ids: list[int] | None = None) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(
            [[10 if identity % 2 == 0 else 120, 10, 40 if identity % 2 == 0 else 150, 70] for identity in identities],
            dtype=float,
        ).reshape(-1, 4),
        tracker_id=np.array(identities, dtype=int),
        confidence=np.ones(len(identities)),
        class_id=np.array(class_ids or [1] * len(identities), dtype=int),
    )


def _write_sequence(sequence: Path, frame_numbers: range, detection_rows: list[str]) -> None:
    """Write a small MOT sequence with blank frames for CLI tests."""
    images = sequence / "img1"
    images.mkdir(parents=True)
    (sequence / "det").mkdir()
    (sequence / "seqinfo.ini").write_text(f"[Sequence]\nseqLength={frame_numbers[-1]}\nframeRate=30\n")
    (sequence / "det/det.txt").write_text("".join(detection_rows))
    frame = np.zeros((96, 200, 3), dtype=np.uint8)
    for number in frame_numbers:
        assert cv2.imwrite(str(images / f"{number:06d}.png"), frame)


def test_writer_maps_repeated_historical_frames_without_retaining_completed_references() -> None:
    tracker = McBytePlusPlusTracker(reid_model=_Encoder(), enable_cmc=False)
    joiner = tracker.reid_joiner
    assert joiner is not None
    written: list[tuple[int, int]] = []
    writer = _ReIDResultWriter(
        tracker, lambda number, rows: written.extend((number, int(identity)) for identity in rows.tracker_id)
    )
    frame = np.zeros((96, 200, 3), dtype=np.uint8)
    observations = [
        (10, [0, 1], frame, []),
        (30, [], None, [0, 1]),
        (40, [2, 3], None, []),
        (70, [2], frame, []),
        (90, [3], frame, []),
        (100, [4], None, []),
    ]
    for accepted, (source, identities, pixels, removed) in enumerate(observations, start=1):
        tracker.frame_id = accepted
        current = joiner.update(_detections(identities), pixels, removed_tracklet_ids=removed)
        writer.write_current(source, current)
        if accepted == 4:
            assert writer._pending == {3: (40, 1)}
        if accepted == 5:
            assert writer._pending == {}
    writer.finish()
    expected = [(10, 0), (10, 1), (40, 0), (40, 1), (70, 0), (90, 1), (100, 2)]
    assert sorted(written) == expected
    assert len(written) == len(set(written))
    assert writer.historical_rows == 3
    assert writer._pending == {}


@pytest.mark.parametrize("buffer_results", [True, False])
@pytest.mark.parametrize("interrupt", [None, "between_frames", "inside_update"])
def test_generic_track_maps_sparse_source_frames_and_finishes_before_writer_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, buffer_results: bool, interrupt: str | None
) -> None:
    tracker = McBytePlusPlusTracker(
        reid_model=_Encoder(),
        reid_joining_config=LongTermReIDConfig(buffer_results=buffer_results),
        minimum_consecutive_frames=1,
        enable_cmc=False,
    )
    frame = np.zeros((96, 200, 3), dtype=np.uint8)

    def frames(_: str) -> Iterator[tuple[int, np.ndarray]]:
        for number in (20, 22, 80, 150):
            yield number, frame
        if interrupt == "between_frames":
            raise KeyboardInterrupt

    monkeypatch.setattr(track_module, "frames_from_source", frames)
    update = tracker.update

    def interrupting_update(detections: sv.Detections, frame: np.ndarray | None = None) -> sv.Detections:
        tracked = update(detections, frame)
        if tracker.frame_id == 4:
            # This update buffered a row but never returned its source-frame mapping.
            raise KeyboardInterrupt
        return tracked

    if interrupt == "inside_update":
        monkeypatch.setattr(tracker, "update", interrupting_update)
    detector_outputs = [sv.Detections.empty(), *[_detections([8, 9, 10], class_ids=[1, 1, 2]) for _ in range(3)]]
    monkeypatch.setattr(track_module, "_run_model", lambda *args: detector_outputs.pop(0))
    path = tmp_path / "results.txt"
    assert (
        track_module._run_with_source(
            source="unused",
            source_info=_SourceInfo(source_type="image_dir", total_frames=4),
            model=object(),
            confidence=0.5,
            detections_data=None,
            class_names=[],
            class_filter=[1],
            track_id_filter=[0],
            tracker=tracker,
            output=None,
            mot_results=path,
            display=False,
            show=ShowOptions(),
        )
        == 0
    )
    rows = [(int(parts[0]), int(parts[1])) for row in path.read_text().splitlines() if (parts := row.split(","))]
    # The first box observation remains tentative; confirmed pending rows at
    # source frames 80 and 150 flush with the requested public ID at EOF.
    expected = [(80, 0)] if interrupt == "inside_update" else [(80, 0), (150, 0)]
    assert rows == (expected if buffer_results else [])
    with pytest.raises(RuntimeError, match=r"finish|finished"):
        tracker.update(sv.Detections.empty())


def test_generic_frameless_mcbyteplusplus_finishes_with_reid_disabled(tmp_path: Path) -> None:
    from trackers.io.mot import load_mot_file

    path = tmp_path / "detections.txt"
    path.write_text("1,-1,10,10,30,60,1,1,-1,-1\n")
    tracker = McBytePlusPlusTracker(enable_cmc=False)
    assert (
        track_module._run_frameless(
            detections_data=load_mot_file(path),
            class_filter=None,
            track_id_filter=None,
            tracker=tracker,
            mot_results=tmp_path / "results.txt",
        )
        == 0
    )
    with pytest.raises(RuntimeError, match=r"finish|finished"):
        tracker.update(sv.Detections.empty())


@pytest.mark.parametrize("command", ["inspect", "benchmark", "track"])
@pytest.mark.parametrize("enable_pauses", [False, True])
def test_joining_config_parses_from_none_and_replays(tmp_path: Path, command: str, enable_pauses: bool) -> None:
    parser = _CLIParser(exit_on_error=False)
    parser.add_argument("--config", action="config")
    required: dict[str, Any]
    if command == "inspect":
        parser.add_function_arguments(inspect_module.mcbyteplusplus_inspection)
        required = {
            "sequence": {
                "image_dir": str(tmp_path),
                "det_file": str(tmp_path / "det.txt"),
                "start_frame": 2,
                "end_frame": 3,
            }
        }
    elif command == "benchmark":
        parser.add_function_arguments(benchmark_module.benchmark_command)
        required = {"dataset_root": str(tmp_path)}
    else:
        parser.add_function_arguments(track_module.track_command)
        required = {"tracker": {"name": "mcbyteplusplus"}}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(required))
    args = parser.instantiate(
        parser.parse_args(
            [
                "--config",
                str(config),
                "--reid.model",
                "mcbyteplusplus_sports",
                "--reid.enable",
                "--tracker.reid_joining_config.minimum_cosine_similarity",
                "0.91",
                "--tracker.reid_joining_config.buffer_results",
                "false",
                "--tracker.enable_reid_camera_motion_pause",
                str(enable_pauses).lower(),
                "--tracker.reid_camera_motion_threshold",
                "0.025",
                "--tracker.reid_joining_config.maximum_reconnection_pause_steps",
                "1000",
                "--tracker.reid_joining_config.enable_feature_occlusion_filter",
                str(enable_pauses).lower(),
                "--tracker.reid_joining_config.pause_reconnection_on_occlusion",
                str(enable_pauses).lower(),
            ]
        )
    )
    assert args.reid.model == "mcbyteplusplus_sports"
    assert args.reid.enable is True
    expected = LongTermReIDConfig(
        minimum_cosine_similarity=0.91,
        buffer_results=False,
        maximum_reconnection_pause_steps=1000,
        enable_feature_occlusion_filter=enable_pauses,
        pause_reconnection_on_occlusion=enable_pauses,
    )
    assert args.tracker.reid_joining_config == expected
    assert args.tracker.enable_reid_camera_motion_pause is enable_pauses
    assert args.tracker.reid_camera_motion_threshold == 0.025
    required["tracker"] = asdict(args.tracker)
    required["reid"] = asdict(args.reid)
    config.write_text(json.dumps(required))
    replay = parser.instantiate(parser.parse_args(["--config", str(config)]))
    assert replay.tracker.reid_joining_config == expected
    assert replay.tracker.enable_reid_camera_motion_pause is enable_pauses
    assert replay.tracker.reid_camera_motion_threshold == 0.025


def test_generic_mcbyteplusplus_uses_explicit_loader_and_accepts_joining_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _Encoder()
    loaded = []

    def load(source: str, **kwargs: Any) -> _Encoder:
        loaded.append((source, kwargs))
        return encoder

    monkeypatch.setattr(track_module, "_load_long_term_encoder", load)
    joining = LongTermReIDConfig(minimum_cosine_similarity=0.9)
    options = track_module.TrackerOptions(name="mcbyteplusplus", reid_joining_config=joining)
    tracker = track_module._init_tracker(options, ReIDModelConfig(model="mcbyteplusplus_sports", device="cpu"))
    assert isinstance(tracker, McBytePlusPlusTracker)
    assert tracker.reid_joiner is not None
    assert tracker.reid_joiner.model is encoder
    assert tracker.reid_joiner.config is joining
    assert tracker.enable_reid_camera_motion_pause is True
    assert joining.enable_feature_occlusion_filter is True
    assert joining.pause_reconnection_on_occlusion is True
    assert loaded == [("mcbyteplusplus_sports", {"device": "cpu", "architecture": None})]
    with pytest.raises(ValueError, match=r"requires --reid\.model"):
        track_module._init_tracker(options, ReIDModelConfig(enable=True))
    with pytest.raises(ValueError, match="requires reid_model"):
        track_module._init_tracker(options)


class _ReplayTracker:
    """Drive the real joining component while replacing geometry and masks."""

    last_mask_output = None
    untrusted_mask_ids: frozenset[int] = frozenset()

    def __init__(self, **kwargs: Any) -> None:
        self.frame_id = 0
        self.reset_called = False
        self.finish_called = False
        self.reid_joiner = LongTermReID(kwargs["reid_model"], kwargs["reid_joining_config"])
        self.maximum_frames_without_update = 30
        self.mask_manager = SimpleNamespace(
            mask_propagator=SimpleNamespace(
                device="cpu",
                model_revision="fake-revision",
                session_index=1,
                session_frame_count=0,
            )
        )

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        self.frame_id += 1
        self.mask_manager.mask_propagator.session_frame_count += 1
        detections.tracker_id = np.full(len(detections), 8)
        return self.reid_joiner.update(detections, frame)

    def pop_finalized_results(self) -> dict[int, sv.Detections]:
        return self.reid_joiner.pop_finalized_results()

    def finish(self) -> dict[int, sv.Detections]:
        assert not self.reset_called
        self.finish_called = True
        return self.reid_joiner.finish()

    def reset(self) -> None:
        self.reset_called = True
        self.reid_joiner.reset()


@pytest.mark.parametrize("command", ["inspect", "benchmark"])
def test_dedicated_runs_flush_tail_and_record_replayable_encoder_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """Dedicated commands flush each sequence's history and save replayable model settings."""
    root = tmp_path / "data"
    first_frame, last_frame = (2, 4) if command == "inspect" else (1, 3)
    sequence_names = ["SNMOT-001"] if command == "inspect" else ["SNMOT-001", "SNMOT-002"]
    detection_rows = [f"{number},-1,10,10,30,60,1,-1,-1,-1\n" for number in range(first_frame + 1, last_frame + 1)]
    for name in sequence_names:
        _write_sequence(root / name, range(first_frame, last_frame + 1), detection_rows)
    instances = []
    loads = []
    encoder = _Encoder()

    def construct(**kwargs: Any) -> _ReplayTracker:
        instance = _ReplayTracker(**kwargs)
        instances.append(instance)
        return instance

    def load(options: ReIDModelConfig) -> _Encoder:
        loads.append(options)
        return encoder

    output = tmp_path / "output"
    reid = ReIDModelConfig(model="fake-encoder", device="cpu")
    if command == "inspect":
        monkeypatch.setattr(inspect_module, "McBytePlusPlusTracker", construct)
        monkeypatch.setattr(inspect_module, "_load_mcbyteplusplus_reid", load)
        sequence = root / "SNMOT-001"
        assert (
            inspect_module.mcbyteplusplus_inspection(
                inspect_module.SequenceOptions(sequence / "img1", sequence / "det/det.txt", first_frame, last_frame),
                cmc=CMCOptions(enable=False),
                reid=reid,
                output_dir=output,
            )
            == 0
        )
    else:
        monkeypatch.setattr(benchmark_module, "McBytePlusPlusTracker", construct)
        monkeypatch.setattr(benchmark_module, "_load_mcbyteplusplus_reid", load)
        assert benchmark_module.benchmark_command(root, reid=reid, output_root=output, save_frames=False) == 0
    assert len(loads) == 1
    assert len(instances) == len(sequence_names)
    assert all(
        instance.reid_joiner.model is encoder and instance.finish_called and instance.reset_called
        for instance in instances
    )
    run = next(output.iterdir())
    results = [run / "results.txt"] if command == "inspect" else sorted((run / "results").glob("*.txt"))
    expected_names = ["results.txt"] if command == "inspect" else [f"{name}.txt" for name in sequence_names]
    assert [result.name for result in results] == expected_names
    expected_rows = [(first_frame + 1, 0), (last_frame, 0)]
    for result in results:
        rows = [row.split(",") for row in result.read_text().splitlines()]
        assert [(int(row[0]), int(row[1])) for row in rows] == expected_rows

    manifest = json.loads((run / "run.json").read_text())
    assert manifest["reid"]["source"] == "fake-encoder"
    assert manifest["reid"]["encoder"].endswith("._Encoder")
    assert manifest["reid"]["device"] == "cpu"
    assert manifest["reid"]["preprocessing"]["to_rgb"] is True
    assert "reid_version" in manifest["reid"]
    config = json.loads((run / "config.json").read_text())
    assert config["tracker"]["reid_joining_config"] == asdict(LongTermReIDConfig())
    assert config["tracker"]["enable_reid_camera_motion_pause"] is True
    assert config["tracker"]["reid_camera_motion_threshold"] == 0.02

    parser = _CLIParser(exit_on_error=False)
    parser.add_argument("--config", action="config")
    parser.add_function_arguments(
        inspect_module.mcbyteplusplus_inspection if command == "inspect" else benchmark_module.benchmark_command
    )
    replay = parser.instantiate(parser.parse_args(["--config", str(run / "config.json")]))
    assert replay.reid == reid
    assert replay.tracker.reid_joining_config == LongTermReIDConfig()


def test_provenance_hashes_local_artifacts_and_records_sports_source(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.safetensors"
    checkpoint.write_bytes(b"synthetic checkpoint")
    provenance = _reid_provenance(ReIDModelConfig(model=str(checkpoint)), _Encoder())
    assert provenance["artifact_sha256"] == {str(checkpoint): hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    sports = _reid_provenance(ReIDModelConfig(model="mcbyteplusplus_sports"), _Encoder())
    assert sports["architecture"] == "osnet_x1_0"
    assert len(sports["source_sha256"]) == 64
    assert sports["preprocessing"]["input_size"] == [256, 128]


@pytest.mark.parametrize("command", ["inspect", "benchmark"])
@pytest.mark.parametrize("buffer_interrupted_update", [False, True], ids=["before_update", "inside_update"])
def test_dedicated_interrupt_flushes_completed_pending_rows_and_preserves_partial_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, buffer_interrupted_update: bool
) -> None:
    sequence = tmp_path / "data" / "SNMOT-001"
    images = sequence / "img1"
    rows = []
    for number in range(1, 5):
        rows.append(f"{number},-1,10,10,30,60,1,-1,-1,-1\n")
        if number > 1:
            rows.append(f"{number},-1,120,10,30,60,1,-1,-1,-1\n")
    _write_sequence(sequence, range(1, 5), rows)
    detection_file = sequence / "det/det.txt"
    instances = []

    class InterruptingTracker(_ReplayTracker):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            instances.append(self)

        def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
            if self.frame_id == 3 and not buffer_interrupted_update:
                raise KeyboardInterrupt
            self.frame_id += 1
            self.mask_manager.mask_propagator.session_frame_count += 1
            detections.tracker_id = (detections.xyxy[:, 0] > 100).astype(int)
            tracked = self.reid_joiner.update(detections, frame)
            if self.frame_id == 4:
                raise KeyboardInterrupt
            return tracked

    module = inspect_module if command == "inspect" else benchmark_module
    monkeypatch.setattr(module, "McBytePlusPlusTracker", InterruptingTracker)
    monkeypatch.setattr(module, "_load_mcbyteplusplus_reid", lambda _: _Encoder())
    output = tmp_path / "output"
    reid = ReIDModelConfig(model="fake-encoder", device="cpu")
    if command == "inspect":
        code = inspect_module.mcbyteplusplus_inspection(
            inspect_module.SequenceOptions(images, detection_file, 1, 4), reid=reid, output_dir=output
        )
    else:
        code = benchmark_module.benchmark_command(sequence.parent, reid=reid, output_root=output, save_frames=False)
    assert code == 130
    assert len(instances) == 1
    assert instances[0].finish_called and instances[0].reset_called
    run = next(output.iterdir())
    report = json.loads((run / "run.json").read_text())
    assert report["status"] == "interrupted"
    if command == "inspect":
        assert [entry["frame"] for entry in report["frames"]] == [1, 2, 3]
        assert report["reid_finalized_rows"] == 2
        result_path = run / "results.txt"
    else:
        assert report["sequences"][0]["status"] == "interrupted"
        assert report["sequences"][0]["processed_frames"] == 3
        assert report["sequences"][0]["reid_finalized_rows"] == 2
        assert report["totals"]["completed_sequences"] == 0
        assert report["totals"]["completed_frames"] == 0
        assert report["all_discovered_sequences_completed"] is False
        assert not (run / "results/SNMOT-001.txt").exists()
        result_path = run / "results/SNMOT-001.txt.partial"
    result_rows = [row.split(",") for row in result_path.read_text().splitlines()]
    expected = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1)]
    assert [(int(row[0]), int(row[1])) for row in result_rows] == expected


def test_visualization_keeps_internal_mask_owners_and_labels_public_or_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    from trackers.cli import _mcbyteplusplus as visual_module

    masks = np.ones((2, 96, 200), dtype=bool)
    tracker = McBytePlusPlusTracker(enable_cmc=False, reid_model=_Encoder())
    monkeypatch.setattr(
        type(tracker), "last_mask_output", property(lambda _: MaskOutput(masks=masks, tracklet_mask_dict={8: 0, 9: 1}))
    )
    detections = _detections([0, -1])
    detections.data = {"tracklet_id": np.array([8, 9]), "reid_pending": np.array([False, True])}
    captured: dict[str, Any] = {}
    status_lines: list[list[str]] = []

    def boxes(frame: np.ndarray, xyxy: np.ndarray, identities: list[int], labels: list[str]) -> np.ndarray:
        captured.update(ids=identities, labels=labels)
        return frame

    monkeypatch.setattr(visual_module, "annotate_tracklet_boxes", boxes)

    def status(frame: np.ndarray, lines: list[str]) -> np.ndarray:
        status_lines.append(lines)
        return frame

    monkeypatch.setattr(visual_module, "draw_status_lines", status)
    visualize_mcbyteplusplus(np.zeros((96, 200, 3), dtype=np.uint8), 5, detections, tracker)
    assert captured == {"ids": [8, 9], "labels": ["0", "-1 pending (9)"]}
    mask_detections = track_module._mcbyteplusplus_mask_detections(detections, tracker)
    np.testing.assert_array_equal(mask_detections.tracker_id, [8, 9])
    np.testing.assert_array_equal(mask_detections.mask, masks)
    assert track_module._format_labels(detections, [], ShowOptions()) == ["#0", "#-1 pending"]
    visualize_mcbyteplusplus(np.zeros((96, 200, 3), dtype=np.uint8), 6, _detections([]), tracker)
    pending_lines = [line for lines in status_lines for line in lines if line.startswith("Pending internal IDs:")]
    expected_pending_lines = ["Pending internal IDs: [9]", "Pending internal IDs: []"]
    assert pending_lines == expected_pending_lines
