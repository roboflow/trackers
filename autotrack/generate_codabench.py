#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""generate_codabench.py — Generate a MOT17 CodaBench submission ZIP.

Runs a tracker over the MOT17 test sequences and packages the results into
a flat ZIP ready for upload to https://www.codabench.org/competitions/10049.

Submission format (21 files, flat ZIP):
    MOT17-{01,03,06,07,08,12,14}-{DPM,FRCNN,SDP}.txt

Each line (10 comma-separated values, 1-based frame and id):
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
    where x/y/z are always -1 (2-D challenge).

Detector variants
-----------------
``--det-source`` selects which detection file(s) to use:

* ``frcnn`` / ``dpm`` / ``sdp`` — one of the three public MOT17 bundled
  detectors.  The tracker runs on that detector's detections; the same
  result is written to all three detector-name slots in the ZIP (MOT17
  requires 21 files regardless of which detector was used).

* ``all`` — runs the tracker separately on each of DPM, FRCNN, and SDP;
  produces 21 distinct result files.

* Any other string (e.g. ``rfdetr``) — treats it as a custom detector tag.
  Expects ``{dataset_dir}/MOT17-{seq}-{TAG}/det/det.txt`` (TAG = uppercased
  det-source string).  Result is copied to all three DET slots in the ZIP.

Config loading
--------------
Tracker params are resolved in this order (highest priority first):

1. CLI keyword arguments (``--lost-track-buffer 30``, etc.)
2. ``best_config.json`` entry for ``tracker / det_source`` (if the file exists)
3. Tracker constructor defaults

Usage:
    # Best config for bytetrack+frcnn, result replicated to all 3 det slots:
    uv run python generate_codabench.py bytetrack --det-source frcnn

    # Run all 3 public detectors separately (21 unique result files):
    uv run python generate_codabench.py bytetrack --det-source all

    # Custom RF-DETR detections, bytetrack, explicit params:
    uv run python generate_codabench.py bytetrack --det-source rfdetr \\
        --lost-track-buffer 50 --track-activation-threshold 0.4

    # OC-SORT with SDP detections:
    uv run python generate_codabench.py ocsort --det-source sdp

Prerequisites:
    trackers download mot17 --split test --asset detections
    (frames are NOT needed for detection-based tracking)
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

console = Console()
_err = Console(stderr=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_SEQ_IDS = ["01", "03", "06", "07", "08", "12", "14"]
_PUBLIC_DET_TAGS = ["DPM", "FRCNN", "SDP"]
_DEFAULT_DATASET_DIR = Path(__file__).parent / "mot17" / "test"

# ---------------------------------------------------------------------------
# Tracker construction
# ---------------------------------------------------------------------------

# Global to support Kalman monkey-patch for ByteTrack
_ORIG_KALMAN_INIT: Any = None


def _apply_kalman_patch(params: dict, tracker_name: str) -> None:
    """Monkey-patch ByteTrack Kalman matrices from ``params``. No-op for others."""
    if tracker_name != "bytetrack":
        return

    global _ORIG_KALMAN_INIT
    from trackers.core.bytetrack.kalman import ByteTrackKalmanBoxTracker

    if _ORIG_KALMAN_INIT is None:
        _ORIG_KALMAN_INIT = ByteTrackKalmanBoxTracker._initialize_kalman_filter

    q = params.get("q_scale", 0.01)
    r = params.get("r_scale", 0.1)
    p = params.get("p_scale", 1.0)
    vel_decay = params.get("velocity_decay", 0.95)
    q_miss = params.get("q_miss_alpha", 0.1)
    orig = _ORIG_KALMAN_INIT

    def _patched(self: ByteTrackKalmanBoxTracker) -> None:
        orig(self)
        self.Q = np.eye(self.Q.shape[0], dtype=np.float32) * q
        self.R = np.eye(self.R.shape[0], dtype=np.float32) * r
        self.P = np.eye(self.P.shape[0], dtype=np.float32) * p

    setattr(ByteTrackKalmanBoxTracker, "_initialize_kalman_filter", _patched)
    setattr(ByteTrackKalmanBoxTracker, "velocity_decay", vel_decay)
    setattr(ByteTrackKalmanBoxTracker, "q_miss_alpha", q_miss)
    setattr(
        ByteTrackKalmanBoxTracker,
        "p_reset_threshold",
        params.get("p_reset_threshold", 5),
    )
    setattr(ByteTrackKalmanBoxTracker, "oru_threshold", params.get("oru_threshold", 2))


def _build_tracker(params: dict, tracker_name: str):
    """Construct a tracker from ``params``."""
    _apply_kalman_patch(params, tracker_name)

    if tracker_name == "bytetrack":
        from trackers import ByteTrackTracker

        return ByteTrackTracker(
            lost_track_buffer=params.get("lost_track_buffer", 30),
            minimum_consecutive_frames=params.get("minimum_consecutive_frames", 1),
            minimum_iou_threshold=params.get("minimum_iou_threshold", 0.2),
            track_activation_threshold=params.get("track_activation_threshold", 0.25),
            high_conf_det_threshold=params.get("high_conf_det_threshold", 0.6),
        )
    if tracker_name == "sort":
        from trackers import SORTTracker

        return SORTTracker(
            lost_track_buffer=params.get("lost_track_buffer", 1),
            minimum_consecutive_frames=params.get("minimum_consecutive_frames", 3),
            minimum_iou_threshold=params.get("minimum_iou_threshold", 0.3),
        )
    if tracker_name == "ocsort":
        from trackers import OCSORTTracker

        return OCSORTTracker(
            lost_track_buffer=params.get("lost_track_buffer", 30),
            minimum_consecutive_frames=params.get("minimum_consecutive_frames", 3),
            minimum_iou_threshold=params.get("minimum_iou_threshold", 0.3),
            direction_consistency_weight=params.get(
                "direction_consistency_weight", 0.5
            ),
            high_conf_det_threshold=params.get("high_conf_det_threshold", 0.6),
            delta_t=params.get("delta_t", 3),
        )
    raise ValueError(
        f"Unknown tracker: {tracker_name!r}. Choose: sort | bytetrack | ocsort"
    )


# ---------------------------------------------------------------------------
# Sequence tracking
# ---------------------------------------------------------------------------


def _run_sequence(tracker, det_file: Path, max_interpolation_gap: int = 0) -> list[str]:
    """Run tracker on one sequence's detections; return MOT-format lines.

    Args:
        tracker: Initialised tracker instance (will be reset before use).
        det_file: Path to the MOT-format detection file.
        max_interpolation_gap: Fill gaps up to this many frames (0 = off).

    Returns:
        List of formatted MOT result lines (no trailing newline per line).
    """
    from trackers.io.mot import _load_mot_file, _mot_frame_to_detections

    tracker.reset()
    detections_data = _load_mot_file(det_file)
    if not detections_data:
        return []

    max_frame = max(detections_data.keys())
    lines: list[str] = []

    for frame_idx in range(1, max_frame + 1):
        dets = (
            _mot_frame_to_detections(detections_data[frame_idx])
            if frame_idx in detections_data
            else sv.Detections.empty()
        )
        tracked = tracker.update(dets)

        if tracked.tracker_id is None:
            continue

        for i, tid in enumerate(tracked.tracker_id):
            if tid < 0:
                continue
            x1, y1, x2, y2 = tracked.xyxy[i]
            w, h = x2 - x1, y2 - y1
            conf = (
                float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
            )
            # MOT format uses 1-based track IDs; tracker returns 0-based
            lines.append(
                f"{frame_idx},{int(tid) + 1},{x1:.2f},{y1:.2f},"
                f"{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"
            )

    if max_interpolation_gap > 0:
        from trackers.core.sort.utils import interpolate_mot_gaps

        lines = interpolate_mot_gaps(lines, max_gap=max_interpolation_gap)

    return lines


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_submission(
    tracker: str = "bytetrack",
    det_source: str = "all",
    dataset_dir: str | None = None,
    output: str | None = None,
    **kwargs: Any,
) -> None:
    """Generate a MOT17 CodaBench submission ZIP.

    Args:
        tracker: Tracker algorithm — ``sort``, ``bytetrack``, or ``ocsort``.
        det_source: Detection source tag.  One of ``dpm``, ``frcnn``, ``sdp``,
            ``all`` (run all three public detectors separately), or a custom tag
            (e.g. ``rfdetr``) pointing to
            ``{dataset_dir}/MOT17-{seq}-{TAG}/det/det.txt``.
        dataset_dir: Path to the MOT17 test directory.
            Defaults to the ``mot17/test`` sibling of this script.
        output: Path for the output ZIP file.
            Defaults to ``submission-{tracker}-{det_source}.zip``.
        **kwargs: Tracker parameter overrides passed directly to the tracker
            constructor (e.g. ``lost_track_buffer=50``).

    Examples:
        Run from the autotrack directory:

        >>> # uv run python generate_codabench.py bytetrack
        >>> # uv run python generate_codabench.py bytetrack --det-source all
    """
    data_dir = Path(dataset_dir) if dataset_dir else _DEFAULT_DATASET_DIR
    det_source_norm = det_source.lower()

    if output is None:
        output = f"submission-{tracker}-{det_source_norm}.zip"
    output_path = Path(output)

    if det_source_norm == "all":
        run_pairs: list[tuple[str, str]] = [(t, t) for t in _PUBLIC_DET_TAGS]
    else:
        file_tag = det_source_norm.upper()
        run_pairs = [(file_tag, file_tag)]

    params = dict(kwargs)
    max_interp = int(params.pop("max_interpolation_gap", 0))

    # Report configuration
    console.print("\n[bold]MOT17 submission generator[/bold]")
    console.print(f"  tracker   : [cyan]{tracker}[/cyan]")
    console.print(f"  det_source: [cyan]{det_source}[/cyan]")
    console.print(f"  dataset   : [cyan]{data_dir}[/cyan]")
    console.print(f"  output    : [cyan]{output_path}[/cyan]")
    if params:
        console.print(f"  params    : {params}")
    console.print()

    # Validate dataset dir
    if not data_dir.exists():
        _err.print(f"[red]Error: dataset_dir does not exist: {data_dir}[/red]")
        _err.print("  Download test data first:")
        _err.print("    trackers download mot17 --split test --asset detections")
        sys.exit(1)

    # Collect (output_filename → lines) mapping
    # Key = "MOT17-{seq_id}-{DET_TAG}.txt"
    results: dict[str, list[str]] = {}

    total_jobs = len(_TEST_SEQ_IDS) * len(run_pairs)

    with Progress(
        TextColumn("  {task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Tracking", total=total_jobs)

        for seq_id in _TEST_SEQ_IDS:
            for out_tag, file_tag in run_pairs:
                seq_dir_name = f"MOT17-{seq_id}-{file_tag}"
                det_file = data_dir / seq_dir_name / "det" / "det.txt"

                progress.update(task, description=f"[cyan]{seq_dir_name}[/cyan]")

                if not det_file.exists():
                    _err.print(
                        f"\n[yellow]Warning: detection file not found, "
                        f"skipping: {det_file}[/yellow]"
                    )
                    progress.advance(task)
                    continue

                tracker_inst = _build_tracker(params, tracker)
                lines = _run_sequence(tracker_inst, det_file, max_interp)
                out_name = f"MOT17-{seq_id}-{out_tag}.txt"
                results[out_name] = lines
                progress.advance(task)

    if not results:
        _err.print("[red]Error: no sequences were processed.[/red]")
        sys.exit(1)

    # When using a single (non-all) source, replicate to all 3 DET slots
    if det_source_norm != "all":
        expanded: dict[str, list[str]] = {}
        for seq_id in _TEST_SEQ_IDS:
            for out_tag in _PUBLIC_DET_TAGS:
                out_name = f"MOT17-{seq_id}-{out_tag}.txt"
                # Find the result we produced for this sequence
                src_name = next(
                    (k for k in results if k.startswith(f"MOT17-{seq_id}-")),
                    None,
                )
                if src_name is not None:
                    expanded[out_name] = results[src_name]
        results = expanded

    # Package into ZIP
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, lines in sorted(results.items()):
            content = "\n".join(lines) + ("\n" if lines else "")
            zf.writestr(filename, content)

    n_files = len(results)
    n_missing = 21 - n_files
    console.print(f"\n[green]✓ Wrote {n_files}/21 files → {output_path}[/green]")
    if n_missing > 0:
        console.print(
            f"[yellow]  ⚠ {n_missing} files missing — upload may be rejected.[/yellow]"
        )
        console.print(
            f"    Check that all 7 sequence directories are present in {data_dir}"
        )
    else:
        console.print("  Upload at: https://www.codabench.org/competitions/10049")
    console.print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import fire

    fire.Fire(generate_submission)
