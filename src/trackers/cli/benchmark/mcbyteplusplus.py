# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Run McByte++ on saved detections and save outputs for later scoring.

Supports SoccerNet oracle detections and six-column YOLOX detections for MOT17, DanceTrack and SportsMOT. Each sequence
supplies ``seqinfo.ini`` and ``img1``. Ground-truth files are never read. Accuracy is evaluated separately from the
completed MOT result files.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import logging
import math
import platform
import re
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import cv2
import numpy as np
from jsonargparse import ArgumentParser
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from trackers.cli._detections import DetectionFileFormat, build_detections, find_frame_path, read_detection_file
from trackers.cli._mcbyteplusplus import CMCOptions, TrackerOptions, _HubAuthNoticeFilter, visualize_mcbyteplusplus
from trackers.cli._parser import register_argument_adder
from trackers.cli._tracking import append_mot_results
from trackers.core.masks.edgetam import EdgeTAMMaskBackend
from trackers.core.mcbyteplusplus.tracker import McBytePlusPlusMaskConfig, McBytePlusPlusTracker
from trackers.io.video import _VideoOutput

DEFAULT_OUTPUT_ROOT = Path("outputs/mcbyteplusplus_benchmarks")
BenchmarkDataset = Literal["soccernet", "mot17", "dancetrack", "sportsmot"]


@dataclass(frozen=True)
class _Sequence:
    """One unique video, its detection file and its declared length/playback rate."""

    name: str
    path: Path
    detection_path: Path
    frame_count: int
    frame_rate: float


def _find_sequences(
    dataset_root: Path,
    names: list[str] | None,
    dataset: BenchmarkDataset = "soccernet",
    detection_root: Path | None = None,
) -> tuple[list[_Sequence], list[str]]:
    """Validate selected sequence metadata before loading EdgeTAM.

    Length comes from ``seqinfo.ini``, not the last detection: trailing frames without detections must still advance the
    tracker and appear in the visual outputs. Only paths are inspected here; images reach the tracker one at a time.
    """
    patterns = {"soccernet": "SNMOT-*", "mot17": "MOT17-*", "dancetrack": "dancetrack*", "sportsmot": "v_*"}
    if dataset not in patterns:
        raise ValueError(f"Unsupported dataset: {dataset}.")
    if dataset != "soccernet" and detection_root is None:
        raise ValueError(f"{dataset} requires --detection_root with six-column XYXY detection files.")
    paths: dict[str, Path] = {}
    for path in sorted(dataset_root.glob(patterns[dataset])):
        if not path.is_dir():
            continue
        name = path.name
        if dataset == "mot17":
            match = re.fullmatch(r"(MOT17-\d{2})(?:-(FRCNN|DPM|SDP))?", name)
            if match is None:
                continue
            name = match.group(1)
            # All detector variants contain the same video. Prefer FRCNN when available.
            if name in paths and match.group(2) != "FRCNN":
                continue
        paths[name] = path
    discovered = sorted(paths)
    if not discovered:
        raise ValueError(
            f"No {dataset} sequences found in {dataset_root}; point --dataset_root at the split directory."
        )
    selected = discovered if names is None else names
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("sequences must be a non-empty list without duplicates, or null for all sequences.")
    unknown = set(selected) - set(discovered)
    if unknown:
        raise ValueError(f"Sequences not found in {dataset_root}: {sorted(unknown)}")
    result = []
    for name in selected:
        path = paths[name]
        metadata = configparser.ConfigParser()
        with (path / "seqinfo.ini").open(encoding="utf-8") as handle:
            metadata.read_file(handle)
        frame_count = metadata.getint("Sequence", "seqLength")
        frame_rate = metadata.getfloat("Sequence", "frameRate")
        if frame_count <= 0 or not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError(f"{name}: seqLength and frameRate must be finite and positive.")
        detection_path = detection_root / f"{name}.txt" if detection_root is not None else path / "det/det.txt"
        if not detection_path.is_file() or not (path / "img1").is_dir():
            raise ValueError(f"{name}: expected {detection_path} and {path / 'img1'}.")
        result.append(_Sequence(name, path, detection_path, frame_count, frame_rate))
    return result, discovered


def _synchronize(device: str) -> None:
    """Wait for the selected accelerator so update timings include its work."""
    if device.split(":")[0] not in ("cuda", "mps"):
        return
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    else:
        torch.mps.synchronize()


def _environment() -> dict[str, Any]:
    """Record software and host information without importing optional models."""
    versions: dict[str, str | None] = {}
    for package in ("trackers", "torch", "torchvision", "transformers", "numpy", "supervision", "opencv-python"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "versions": versions,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """Replace a JSON report only after its new contents have been written."""
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_sequence(
    sequence: _Sequence,
    run_dir: Path,
    options: dict[str, Any],
    report: dict[str, Any],
    progress: Progress,
) -> None:
    """Run one independent tracker and promote its outputs only on success.

    ``tracking_seconds`` sums synchronized ``update`` calls, including CMC,
    EdgeTAM, refreshes and session rollover. Loading images, building input
    detections, model construction, drawing and writing are outside that timer.
    No warm-up frames are discarded. ``end_to_end_seconds`` includes all of
    these operations and tracker reset. Failed outputs retain partial names.

    Args:
        sequence: Input directory and declared frame count/rate.
        run_dir: This run's timestamped output directory.
        options: Resolved command options, also saved as config.json.
        report: Mutable sequence report, retained even if execution raises.
        progress: Progress display shared across the test-set run.
    """
    started = perf_counter()
    tracker = None
    video_path = run_dir / "videos" / f"{sequence.name}.partial.mp4"
    frame_dir = run_dir / "frames" / f"{sequence.name}.partial"
    partial_results = run_dir / "results" / f"{sequence.name}.txt.partial"
    frame_count = min(sequence.frame_count, options["max_frames"] or sequence.frame_count)
    report.update(
        status="failed",
        expected_frames=sequence.frame_count,
        requested_frames=frame_count,
        processed_frames=0,
        tracking_seconds=0.0,
        frame_rate=sequence.frame_rate,
        mask_disabled_frames=0,
        save_frames=options["save_frames"],
        save_video=options["save_video"],
    )
    task = progress.add_task(sequence.name, total=frame_count)
    hub_logger = logging.getLogger("huggingface_hub.utils._http")
    hub_filter = _HubAuthNoticeFilter()
    hub_logger.addFilter(hub_filter)
    with warnings.catch_warnings(record=True) as caught:
        try:
            detection_path = sequence.detection_path
            detection_format: DetectionFileFormat = "mot_tlwh" if options["dataset"] == "soccernet" else "xyxy"
            confidence_override = 1.0 if options["dataset"] == "soccernet" else None
            detections = read_detection_file(detection_path, detection_format, confidence_override=confidence_override)
            if any(number > sequence.frame_count for number in detections):
                raise ValueError(f"Detections exceed seqLength in {detection_path}.")
            report["detection_sha256"] = hashlib.sha256(detection_path.read_bytes()).hexdigest()
            report.update(
                detection_path=str(detection_path.resolve()),
                detection_format=detection_format,
                confidence_override=confidence_override,
                image_dir=str((sequence.path / "img1").resolve()),
            )
            loading_started = perf_counter()
            tracker_frame_rate = options["tracker_frame_rate"] or sequence.frame_rate
            tracker = McBytePlusPlusTracker(
                frame_rate=tracker_frame_rate,
                enable_cmc=options["cmc"]["enable"],
                cmc_method=options["cmc"]["method"],
                cmc_downscale=options["cmc"]["downscale"],
                enable_conditional_cmc=options["cmc"]["conditional"],
                cmc_minimum_matches=options["cmc"]["minimum_matches"],
                cmc_minimum_scale=options["cmc"]["minimum_scale"],
                cmc_maximum_scale=options["cmc"]["maximum_scale"],
                enable_mask_manager=True,
                edgetam_config=McBytePlusPlusMaskConfig(**options["edgetam"]),
                **options["tracker"],
            )
            if tracker.mask_manager is None:
                raise RuntimeError("McByte++ benchmarking requires its mask manager.")
            backend = cast(EdgeTAMMaskBackend, tracker.mask_manager.mask_propagator)
            device = str(backend.device)
            _synchronize(device)
            report.update(
                model_load_seconds=perf_counter() - loading_started,
                device=device,
                resolved_revision=backend.model_revision,
                tracker_frame_rate=tracker_frame_rate,
                effective_lost_track_buffer_frames=tracker.maximum_frames_without_update,
            )
            if device.startswith("cuda"):
                import torch

                report.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda)
            partial_results.parent.mkdir(parents=True, exist_ok=True)
            if options["save_frames"]:
                frame_dir.mkdir(parents=True)
            with (
                partial_results.open("w", encoding="utf-8") as results,
                _VideoOutput(video_path if options["save_video"] else None, fps=sequence.frame_rate) as video,
            ):
                for number in range(1, frame_count + 1):
                    path = find_frame_path(sequence.path / "img1", number)
                    frame = cv2.imread(str(path))
                    if frame is None:
                        raise OSError(f"Could not decode {path}.")
                    values = build_detections(detections.get(number, []))
                    _synchronize(device)
                    update_started = perf_counter()
                    tracked = tracker.update(values, frame)
                    _synchronize(device)
                    report["tracking_seconds"] += perf_counter() - update_started
                    report["processed_frames"] += 1
                    report["mask_disabled_frames"] += int(tracker.mask_manager is None)
                    if not np.isfinite(tracked.xyxy).all():
                        raise RuntimeError(f"Non-finite tracker boxes on frame {number}.")
                    append_mot_results(results_file=results, frame_number=number, tracked_detections=tracked)
                    if options["save_video"] or options["save_frames"]:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        visual = cv2.cvtColor(
                            visualize_mcbyteplusplus(rgb, number, tracked, tracker), cv2.COLOR_RGB2BGR
                        )
                        if not video.write(visual):
                            raise OSError(f"Could not write {video_path}.")
                        if options["save_frames"] and not cv2.imwrite(str(frame_dir / f"{number:06d}.jpg"), visual):
                            raise OSError(f"Could not save annotated frame {number}.")
                    progress.advance(task)
        finally:
            hub_logger.removeFilter(hub_filter)
            if tracker is not None:
                tracker.reset()
            report["end_to_end_seconds"] = perf_counter() - started
            seconds = report["tracking_seconds"]
            report["tracking_fps"] = report["processed_frames"] / seconds if seconds > 0 else None
            report["warnings"] = [
                {
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                    "file": warning.filename,
                    "line": warning.lineno,
                }
                for warning in caught
            ] + [{"category": "HuggingFaceHubWarning", "message": message} for message in hub_filter.messages]
            progress.remove_task(task)
    if options["save_video"]:
        video_path.replace(video_path.with_name(f"{sequence.name}.mp4"))
    if options["save_frames"]:
        frame_dir.replace(frame_dir.with_name(sequence.name))
    completed_results = partial_results.with_suffix("")
    partial_results.replace(completed_results)
    if options["dataset"] == "mot17":
        # The benchmark expects all three detector suffixes for each unique test video.
        submission = run_dir / "submission"
        submission.mkdir(exist_ok=True)
        for detector in ("DPM", "FRCNN", "SDP"):
            shutil.copyfile(completed_results, submission / f"{sequence.name}-{detector}.txt")
    report.update(status="completed", end_to_end_seconds=perf_counter() - started)


def _summarize(reports: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    """Pool completed-sequence time and frames; never average sequence FPS."""
    completed = [report for report in reports if report["status"] == "completed"]
    frames = sum(report["processed_frames"] for report in completed)
    tracking_seconds = sum(report["tracking_seconds"] for report in completed)
    return {
        "completed_sequences": len(completed),
        "failed_sequences": sum(report["status"] == "failed" for report in reports),
        "completed_frames": frames,
        "tracking_seconds": tracking_seconds,
        "tracking_fps": frames / tracking_seconds if tracking_seconds > 0 else None,
        "end_to_end_seconds": elapsed,
        "end_to_end_fps": frames / elapsed if elapsed > 0 else None,
    }


def benchmark_command(
    dataset_root: Path,
    sequences: list[str] | None = None,
    cmc: CMCOptions | None = None,
    edgetam: McBytePlusPlusMaskConfig | None = None,
    tracker: TrackerOptions | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    save_video: bool = False,
    save_frames: bool = True,
    max_frames: int | None = None,
    dataset: BenchmarkDataset = "soccernet",
    detection_root: Path | None = None,
    visual_sequences: list[str] | None = None,
    tracker_frame_rate: float | None = None,
) -> int:
    """Benchmark McByte++ on saved boxes with configurable masks and visual outputs.

    Runs every discovered sequence unless ``sequences`` selects a subset.
    Always enables EdgeTAM; CMC defaults to ORB with downscale four. Detection
    confidences are set to 1.0 for SoccerNet oracle detections; MOT17, DanceTrack
    and SportsMOT retain their supplied YOLOX confidences. Input identities are
    ignored and no ground-truth files are read. Sequence length and video FPS
    come from seqinfo.ini. The tracker uses that FPS unless ``tracker_frame_rate``
    overrides its reference rate. Every image is processed once, without timestamps
    or resampling. Each sequence gets a fresh tracker, with reset on exit.

    Saves results/<sequence>.txt, annotated JPEGs in frames/<sequence>/,
    optional videos/<sequence>.mp4, a reusable config.json,
    run.json with pooled/per-sequence timing and provenance, and run.log with
    library diagnostics including any OOM fallbacks. Python warnings are saved
    in run.json; the CLI beta notice remains visible. HOTA, IDF1 and MOTA are
    computed separately after checking that all requested sequences completed.
    Tracking FPS measures synchronized update calls, including the first
    update, CMC and all mask work. End-to-end FPS also includes model loading,
    input decoding and output writing. No detector inference is performed.
    Failed sequence outputs keep partial names and are excluded from pooled
    tracking FPS; the command continues with the remaining sequences.

    Args:
        dataset_root: Unzipped split directory containing sequence folders.
        sequences: Sequence names, e.g. [SNMOT-116,SNMOT-117]; None selects all.
        cmc: Camera motion settings; default enabled, ORB, downscale four.
        edgetam: Model, device, revision and session bound (default 200 images).
        tracker: Association, confirmation, refresh and mask trust settings.
        output_root: Parent of a new timestamped output directory.
        save_video: Write one annotated video per sequence at its native FPS;
            off by default.
        save_frames: Save one annotated JPEG per frame; enabled by default.
        max_frames: Optional per-sequence frame limit for a short trial. None
            processes complete sequences. Limited runs are marked in run.json.
        dataset: Dataset layout; defaults to soccernet for compatibility.
        detection_root: Directory of <sequence>.txt files. Required for the
            three YOLOX datasets, using frame,x1,y1,x2,y2,confidence. SoccerNet
            defaults to each sequence's det/det.txt in MOT TLWH format.
        visual_sequences: Restrict enabled frame/video outputs to these names;
            None selects all and [] selects none. Text results always cover
            every selected sequence. MOT17 names omit detector suffixes; each
            unique video runs once, with DPM/FRCNN/SDP copies in submission/.
        tracker_frame_rate: Optional positive reference FPS passed to the tracker;
            None uses the sequence FPS. Set 30 together with tracker.lost_track_buffer=30
            to reproduce the research demo's 30-frame lost-track budget on every
            dataset. This does not change image selection or video playback FPS.
            Reports record both rates and the tracker's effective frame budget.

    Returns:
        Zero when every requested sequence completes, one on any failure,
        or 130 after keyboard interruption. Reports and partial outputs survive.
    """
    started = perf_counter()
    cmc = cmc or CMCOptions()
    edgetam = edgetam or McBytePlusPlusMaskConfig()
    tracker = tracker or TrackerOptions()
    try:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive or null.")
        if cmc.downscale <= 0:
            raise ValueError("cmc.downscale must be positive.")
        if tracker_frame_rate is not None and (not math.isfinite(tracker_frame_rate) or tracker_frame_rate <= 0):
            raise ValueError("tracker_frame_rate must be finite and positive or null.")
        selected, discovered = _find_sequences(dataset_root, sequences, dataset, detection_root)
        if visual_sequences is not None and (
            len(set(visual_sequences)) != len(visual_sequences)
            or set(visual_sequences) - {sequence.name for sequence in selected}
        ):
            raise ValueError("visual_sequences must contain unique names from the selected sequences.")
        run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, configparser.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    options: dict[str, Any] = {
        "dataset_root": str(dataset_root.resolve()),
        "sequences": sequences,
        "cmc": asdict(cmc),
        "edgetam": asdict(edgetam),
        "tracker": asdict(tracker),
        "output_root": str(output_root.resolve()),
        "save_video": save_video,
        "save_frames": save_frames,
        "max_frames": max_frames,
        "dataset": dataset,
        "detection_root": str(detection_root.resolve()) if detection_root is not None else None,
        "visual_sequences": visual_sequences,
        "tracker_frame_rate": tracker_frame_rate,
    }
    report: dict[str, Any] = {
        "status": "running",
        "dataset": "SoccerNet-tracking" if dataset == "soccernet" else dataset,
        "config": options,
        "environment": _environment(),
        "requested_revision": edgetam.revision,
        "discovered_sequences": discovered,
        "selected_sequences": [item.name for item in selected],
        "all_discovered_sequences_completed": False,
        "sequences": [],
    }
    log_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(log_handler)
    exit_code = 0
    print(f"McByte++ outputs: {run_dir}", flush=True)
    try:
        _write_report(run_dir / "config.json", options)
        _write_report(run_dir / "run.json", report)
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=Console(stderr=True),
        ) as progress:
            total_task = progress.add_task("Sequences", total=len(selected))
            for sequence in selected:
                sequence_report: dict[str, Any] = {"name": sequence.name, "status": "failed"}
                report["sequences"].append(sequence_report)
                sequence_options = dict(options)
                if visual_sequences is not None and sequence.name not in visual_sequences:
                    sequence_options.update(save_frames=False, save_video=False)
                try:
                    _run_sequence(sequence, run_dir, sequence_options, sequence_report, progress)
                except Exception as error:
                    sequence_report.update(status="failed", error=str(error))
                    progress.console.print(f"{sequence.name} failed: {error}", markup=False)
                    logging.getLogger(__name__).exception("Sequence %s failed", sequence.name)
                    exit_code = 1
                except KeyboardInterrupt:
                    sequence_report.update(status="interrupted", error="Keyboard interruption")
                    raise
                else:
                    resolved = sequence_report["resolved_revision"]
                    if resolved is not None:
                        options["edgetam"]["revision"] = resolved
                    options["edgetam"]["device"] = sequence_report["device"]
                    _write_report(run_dir / "config.json", options)
                report["totals"] = _summarize(report["sequences"], perf_counter() - started)
                _write_report(run_dir / "run.json", report)
                progress.advance(total_task)
        report["status"] = "failed" if exit_code else "completed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        exit_code = 130
    except (OSError, ImportError, ValueError, RuntimeError) as error:
        report.update(status="failed", error=str(error))
        print(f"Error: {error}", file=sys.stderr)
        exit_code = 1
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()
        totals = _summarize(report["sequences"], perf_counter() - started)
        report["totals"] = totals
        report["all_discovered_sequences_completed"] = (
            report["status"] == "completed"
            and set(report["selected_sequences"]) == set(discovered)
            and all(item["processed_frames"] == item["expected_frames"] for item in report["sequences"])
        )
        try:
            _write_report(run_dir / "run.json", report)
        except OSError as error:
            print(f"Error saving run report: {error}", file=sys.stderr)
            exit_code = 1
    fps = totals["tracking_fps"]
    if fps is not None:
        print(
            f"Completed {totals['completed_sequences']}/{len(selected)} sequences; "
            f"{totals['completed_frames']} frames; tracker {fps:.3f} FPS; "
            f"end-to-end {totals['end_to_end_fps']:.3f} FPS."
        )
    print(f"Timing, settings and captured warnings: {run_dir / 'run.json'}")
    print(f"Library diagnostics (including mask fallbacks): {run_dir / 'run.log'}")
    return exit_code


def _add_benchmark_arguments(parser: ArgumentParser) -> list[str]:
    """Use the same nested options and boolean handling as the inspector."""
    parser.description = "Run McByte++ on saved detections; save MOT results, selected annotated frames and FPS."
    added: list[str] = []
    for option_class, key in ((CMCOptions, "cmc"), (McBytePlusPlusMaskConfig, "edgetam"), (TrackerOptions, "tracker")):
        added.extend(parser.add_class_arguments(option_class, key))
    parser.add_argument(
        "--dataset_root", type=Path, required=True, help="Unzipped split directory with sequence folders."
    )
    parser.add_argument("--dataset", type=BenchmarkDataset, default="soccernet", help="Dataset layout.")
    parser.add_argument("--detection_root", type=Path | None, default=None, help="Required YOLOX detection directory.")
    parser.add_argument(
        "--visual_sequences",
        type=list[str] | None,
        default=None,
        help="Visual output names; null for all, [] for none.",
    )
    parser.add_argument("--sequences", type=list[str] | None, default=None, help="Sequence names; null selects all.")
    parser.add_argument(
        "--tracker_frame_rate",
        type=float | None,
        default=None,
        help="Tracker reference FPS; null uses sequence FPS. Does not change video playback or skip frames.",
    )
    parser.add_argument(
        "--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Parent of a new timestamped output directory."
    )
    parser.add_argument("--save_video", type=bool, default=False, help="Save annotated videos at the sequence FPS.")
    parser.add_argument("--save_frames", type=bool, default=True, help="Save one annotated JPEG per frame.")
    parser.add_argument(
        "--max_frames",
        type=int | None,
        default=None,
        help="Per-sequence frame limit for short trials; null processes every declared frame.",
    )
    return [
        *added,
        "dataset_root",
        "sequences",
        "output_root",
        "save_video",
        "save_frames",
        "max_frames",
        "dataset",
        "detection_root",
        "visual_sequences",
        "tracker_frame_rate",
    ]


register_argument_adder(benchmark_command, _add_benchmark_arguments)
