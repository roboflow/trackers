# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from trackers.eval.clear import compute_clear_metrics
from trackers.eval.io import load_mot_file, prepare_mot_sequence
from trackers.eval.results import (
    BenchmarkResult,
    CLEARMetrics,
    SequenceResult,
)

logger = logging.getLogger(__name__)

SUPPORTED_METRICS = ["CLEAR"]


@dataclass
class _DetectionResult:
    """Result of auto-detection."""

    format: Literal["flat", "mot17"]
    benchmark: str | None = None
    split: str | None = None
    tracker_name: str | None = None


def evaluate_mot_sequence(
    gt_path: str | Path,
    tracker_path: str | Path,
    metrics: list[str] | None = None,
    threshold: float = 0.5,
) -> SequenceResult:
    """Evaluate tracker predictions against ground truth for a single sequence.

    Load ground truth and tracker files in MOT Challenge format, compute
    IoU-based similarity, and calculate the requested evaluation metrics.

    Args:
        gt_path: Path to ground truth file in MOT format.
        tracker_path: Path to tracker predictions file in MOT format.
        metrics: List of metrics to compute. Currently supports `["CLEAR"]`.
            Defaults to `["CLEAR"]` if not specified.
        threshold: IoU threshold for matching. Defaults to 0.5.

    Returns:
        SequenceResult containing evaluation metrics. Access metrics via
            `result.CLEAR.MOTA`, `result.CLEAR.IDSW`, etc.

    Raises:
        FileNotFoundError: If ground truth or tracker file does not exist.
        ValueError: If an unsupported metric is requested.

    Examples:
        ```python
        from trackers.eval import evaluate_mot_sequence

        result = evaluate_mot_sequence(
            gt_path="data/gt/MOT17-02/gt.txt",
            tracker_path="data/trackers/MOT17-02.txt",
        )
        result.CLEAR.MOTA
        # 0.756
        result.CLEAR.IDSW
        # 42
        print(result.table())
        ```
    """
    if metrics is None:
        metrics = ["CLEAR"]

    # Validate metrics
    for metric in metrics:
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}. Supported metrics: {SUPPORTED_METRICS}"
            )

    gt_path = Path(gt_path)
    tracker_path = Path(tracker_path)

    # Load data
    gt_data = load_mot_file(gt_path)
    tracker_data = load_mot_file(tracker_path)

    # Prepare sequence (compute IoU, remap IDs)
    seq_data = prepare_mot_sequence(gt_data, tracker_data)

    # Compute metrics
    clear_metrics_dict: dict[str, Any] = {}

    if "CLEAR" in metrics:
        clear_metrics_dict = compute_clear_metrics(
            seq_data.gt_ids,
            seq_data.tracker_ids,
            seq_data.similarity_scores,
            threshold=threshold,
        )

    # Build result
    return SequenceResult(
        sequence=gt_path.stem,
        CLEAR=CLEARMetrics.from_dict(clear_metrics_dict),
    )


def evaluate_benchmark(
    gt_dir: str | Path,
    tracker_dir: str | Path,
    seqmap: str | Path | None = None,
    metrics: list[str] | None = None,
    threshold: float = 0.5,
    benchmark: str | None = None,
    split: str | None = None,
    tracker_name: str | None = None,
) -> BenchmarkResult:
    """Evaluate tracker on multiple sequences (benchmark evaluation).

    The directory structure is auto-detected. Supports both flat format
    (files directly in directory) and MOT Challenge format (nested structure).

    Args:
        gt_dir: Directory containing ground truth files.
        tracker_dir: Directory containing tracker prediction files.
        seqmap: Optional path to sequence map file. If provided, only sequences
            listed in this file will be evaluated.
        metrics: List of metrics to compute. Defaults to `["CLEAR"]`.
        threshold: IoU threshold for matching. Defaults to 0.5.
        benchmark: Override auto-detected benchmark name (e.g., "MOT17").
        split: Override auto-detected split name (e.g., "train", "val").
        tracker_name: Override auto-detected tracker name.

    Returns:
        BenchmarkResult containing per-sequence and aggregate metrics.

    Raises:
        FileNotFoundError: If gt_dir or tracker_dir does not exist.
        ValueError: If auto-detection fails with multiple options.

    Examples:
        Auto-detect everything:

        ```python
        from trackers.eval import evaluate_benchmark

        result = evaluate_benchmark(
            gt_dir="data/gt/",
            tracker_dir="data/trackers/",
        )
        print(result.table())
        ```

        Override tracker name when multiple exist:

        ```python
        result = evaluate_benchmark(
            gt_dir="data/gt/",
            tracker_dir="data/trackers/",
            tracker_name="ByteTrack",
        )
        ```
    """
    if metrics is None:
        metrics = ["CLEAR"]

    gt_dir = Path(gt_dir)
    tracker_dir = Path(tracker_dir)

    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    if not tracker_dir.exists():
        raise FileNotFoundError(f"Tracker directory not found: {tracker_dir}")

    # Smart auto-detection
    detection = _auto_detect(
        gt_dir=gt_dir,
        tracker_dir=tracker_dir,
        benchmark_override=benchmark,
        split_override=split,
        tracker_name_override=tracker_name,
    )

    # Get sequence list
    if seqmap is not None:
        sequences = _parse_seqmap(seqmap)
    else:
        sequences = _discover_sequences(
            gt_dir, detection.format, detection.benchmark, detection.split
        )

    if not sequences:
        raise ValueError(f"No sequences found in {gt_dir}")

    logger.info("Evaluating %d sequences...", len(sequences))

    # Evaluate each sequence
    sequence_results: dict[str, SequenceResult] = {}

    for seq_name in sequences:
        gt_path, tracker_path = _get_paths(
            gt_dir=gt_dir,
            tracker_dir=tracker_dir,
            seq_name=seq_name,
            data_format=detection.format,
            benchmark=detection.benchmark,
            split=detection.split,
            tracker_name=detection.tracker_name,
        )

        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        if not tracker_path.exists():
            raise FileNotFoundError(f"Tracker file not found: {tracker_path}")

        sequence_results[seq_name] = evaluate_mot_sequence(
            gt_path=gt_path,
            tracker_path=tracker_path,
            metrics=metrics,
            threshold=threshold,
        )
        # Fix sequence name (evaluate_mot_sequence uses file stem)
        sequence_results[seq_name] = SequenceResult(
            sequence=seq_name,
            CLEAR=sequence_results[seq_name].CLEAR,
        )

    # Compute aggregate metrics
    aggregate = _aggregate_metrics(sequence_results, metrics)

    return BenchmarkResult(
        sequences=sequence_results,
        aggregate=aggregate,
    )


def _auto_detect(
    gt_dir: Path,
    tracker_dir: Path,
    benchmark_override: str | None,
    split_override: str | None,
    tracker_name_override: str | None,
) -> _DetectionResult:
    """Auto-detect format, benchmark, split, and tracker name.

    Uses overrides if provided, otherwise detects from directory structure.
    Logs what was detected for transparency.
    """
    # First, detect format
    data_format = _detect_format(gt_dir)
    logger.info("Detected format: %s", data_format)

    if data_format == "flat":
        return _DetectionResult(format="flat")

    # MOT17 format - detect benchmark and split
    benchmark, split = _detect_benchmark_split(
        gt_dir, benchmark_override, split_override
    )
    logger.info("Detected benchmark: %s, split: %s", benchmark, split)

    # Detect tracker name
    tracker_name = _detect_tracker_name(
        tracker_dir, benchmark, split, tracker_name_override
    )
    logger.info("Detected tracker: %s", tracker_name)

    return _DetectionResult(
        format="mot17",
        benchmark=benchmark,
        split=split,
        tracker_name=tracker_name,
    )


def _detect_format(gt_dir: Path) -> Literal["flat", "mot17"]:
    """Auto-detect directory format.

    Args:
        gt_dir: Ground truth directory.

    Returns:
        "flat" if .txt files exist in gt_dir, "mot17" otherwise.
    """
    # Check for flat format (*.txt files directly in gt_dir)
    txt_files = list(gt_dir.glob("*.txt"))
    if txt_files:
        return "flat"

    # Check for MOT17 format (*/*/gt/gt.txt pattern)
    mot17_files = list(gt_dir.glob("*/*/gt/gt.txt"))
    if mot17_files:
        return "mot17"

    # Default to flat
    return "flat"


def _detect_benchmark_split(
    gt_dir: Path,
    benchmark_override: str | None,
    split_override: str | None,
) -> tuple[str, str]:
    """Detect benchmark and split from directory structure.

    Looks for directories matching {Benchmark}-{split} pattern.
    If only one is found, uses it. If multiple, requires override.

    Args:
        gt_dir: Ground truth directory.
        benchmark_override: User-provided benchmark name.
        split_override: User-provided split name.

    Returns:
        Tuple of (benchmark, split).

    Raises:
        ValueError: If detection is ambiguous and no override provided.
    """
    # Find all {benchmark}-{split} directories that contain sequences
    benchmark_splits: list[tuple[str, str]] = []

    for subdir in gt_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Check if this directory contains sequence folders with gt/gt.txt
        has_sequences = any(subdir.glob("*/gt/gt.txt"))
        if not has_sequences:
            continue

        # Parse {benchmark}-{split} pattern
        match = re.match(r"^(.+)-(\w+)$", subdir.name)
        if match:
            benchmark_splits.append((match.group(1), match.group(2)))

    if not benchmark_splits:
        raise ValueError(
            f"No benchmark directories found in {gt_dir}. "
            "Expected directories like 'MOT17-train' or 'SportsMOT-val'."
        )

    # If overrides provided, validate and use them
    if benchmark_override is not None and split_override is not None:
        expected_dir = gt_dir / f"{benchmark_override}-{split_override}"
        if not expected_dir.exists():
            available = [f"{b}-{s}" for b, s in benchmark_splits]
            raise ValueError(
                f"Directory '{benchmark_override}-{split_override}' not found. "
                f"Available: {available}"
            )
        return benchmark_override, split_override

    # If only one benchmark-split found, use it
    if len(benchmark_splits) == 1:
        return benchmark_splits[0]

    # Multiple found - need override
    available = [f"{b}-{s}" for b, s in benchmark_splits]
    raise ValueError(
        f"Multiple benchmarks found: {available}. "
        "Please specify --benchmark and --split."
    )


def _detect_tracker_name(
    tracker_dir: Path,
    benchmark: str,
    split: str,
    tracker_name_override: str | None,
) -> str:
    """Detect tracker name from directory structure.

    Looks for tracker directories in {tracker_dir}/{benchmark}-{split}/.
    If only one is found, uses it. If multiple, requires override.

    Args:
        tracker_dir: Tracker directory.
        benchmark: Benchmark name.
        split: Split name.
        tracker_name_override: User-provided tracker name.

    Returns:
        Tracker name.

    Raises:
        ValueError: If detection is ambiguous and no override provided.
    """
    split_dir = tracker_dir / f"{benchmark}-{split}"

    if not split_dir.exists():
        raise ValueError(
            f"Tracker directory not found: {split_dir}. "
            f"Expected structure: {tracker_dir}/{benchmark}-{split}/<tracker>/data/"
        )

    # Find tracker directories (those containing a 'data' subfolder)
    trackers: list[str] = []
    for subdir in split_dir.iterdir():
        if subdir.is_dir() and (subdir / "data").is_dir():
            trackers.append(subdir.name)

    if not trackers:
        raise ValueError(
            f"No tracker directories found in {split_dir}. "
            "Expected structure: <tracker>/data/<sequence>.txt"
        )

    # If override provided, validate and use it
    if tracker_name_override is not None:
        if tracker_name_override not in trackers:
            raise ValueError(
                f"Tracker '{tracker_name_override}' not found. "
                f"Available trackers: {trackers}"
            )
        return tracker_name_override

    # If only one tracker found, use it
    if len(trackers) == 1:
        return trackers[0]

    # Multiple found - need override
    raise ValueError(
        f"Multiple trackers found: {trackers}. "
        "Please specify --tracker-name."
    )


def _discover_sequences(
    gt_dir: Path,
    data_format: Literal["flat", "mot17"],
    benchmark: str | None,
    split: str | None,
) -> list[str]:
    """Discover sequence names from directory structure.

    Args:
        gt_dir: Ground truth directory.
        data_format: Directory format.
        benchmark: Benchmark name (for MOT17).
        split: Split name (for MOT17).

    Returns:
        List of sequence names.
    """
    if data_format == "flat":
        return sorted(
            p.stem for p in gt_dir.glob("*.txt") if not p.name.startswith(".")
        )
    else:
        # MOT17 format: gt/{benchmark}-{split}/{seq}/gt/gt.txt
        split_dir = gt_dir / f"{benchmark}-{split}"
        if not split_dir.exists():
            return []
        return sorted(
            p.parent.parent.name
            for p in split_dir.glob("*/gt/gt.txt")
            if not p.name.startswith(".")
        )


def _get_paths(
    gt_dir: Path,
    tracker_dir: Path,
    seq_name: str,
    data_format: Literal["flat", "mot17"],
    benchmark: str | None,
    split: str | None,
    tracker_name: str | None,
) -> tuple[Path, Path]:
    """Get GT and tracker file paths for a sequence.

    Args:
        gt_dir: Ground truth directory.
        tracker_dir: Tracker directory.
        seq_name: Sequence name.
        data_format: Directory format.
        benchmark: Benchmark name (for MOT17).
        split: Split name (for MOT17).
        tracker_name: Tracker name (for MOT17).

    Returns:
        Tuple of (gt_path, tracker_path).
    """
    if data_format == "flat":
        gt_path = gt_dir / f"{seq_name}.txt"
        tracker_path = tracker_dir / f"{seq_name}.txt"
    else:
        # MOT17 format - these are validated by _auto_detect
        # Type narrowing for mypy
        if benchmark is None or split is None or tracker_name is None:
            raise ValueError("MOT17 format requires benchmark, split, and tracker_name")
        split_name = f"{benchmark}-{split}"
        gt_path = gt_dir / split_name / seq_name / "gt" / "gt.txt"
        tracker_path = (
            tracker_dir / split_name / tracker_name / "data" / f"{seq_name}.txt"
        )

    return gt_path, tracker_path


def _parse_seqmap(seqmap_path: str | Path) -> list[str]:
    """Parse a sequence map file to get list of sequence names."""
    seqmap_path = Path(seqmap_path)
    if not seqmap_path.exists():
        raise FileNotFoundError(f"Sequence map file not found: {seqmap_path}")

    sequences = []
    with open(seqmap_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, and header
            if not line or line.startswith("#") or line.lower() == "name":
                continue
            sequences.append(line)

    return sequences


def _aggregate_metrics(
    sequence_results: dict[str, SequenceResult],
    metrics: list[str],
) -> SequenceResult:
    """Aggregate metrics across sequences."""
    clear_agg: dict[str, Any] = {}

    if "CLEAR" in metrics:
        # Sum integer metrics and MOTP_sum, then recompute ratios
        int_keys = ["CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag"]
        totals: dict[str, int] = {k: 0 for k in int_keys}

        motp_sum_total = 0.0
        clr_frames_total = 0

        for seq_result in sequence_results.values():
            clear = seq_result.CLEAR
            for k in int_keys:
                totals[k] += getattr(clear, k)
            # Use MOTP_sum directly for proper aggregation
            motp_sum_total += clear.MOTP_sum
            clr_frames_total += clear.CLR_Frames

        # Compute aggregate ratios
        num_gt = totals["CLR_TP"] + totals["CLR_FN"]
        num_ids = totals["MT"] + totals["PT"] + totals["ML"]

        mota = (totals["CLR_TP"] - totals["CLR_FP"] - totals["IDSW"]) / max(1.0, num_gt)
        motp = motp_sum_total / max(1.0, totals["CLR_TP"])
        moda = (totals["CLR_TP"] - totals["CLR_FP"]) / max(1.0, num_gt)
        clr_re = totals["CLR_TP"] / max(1.0, num_gt)
        clr_pr = totals["CLR_TP"] / max(1.0, totals["CLR_TP"] + totals["CLR_FP"])
        mtr = totals["MT"] / max(1.0, num_ids)
        ptr = totals["PT"] / max(1.0, num_ids)
        mlr = totals["ML"] / max(1.0, num_ids)
        smota = (motp_sum_total - totals["CLR_FP"] - totals["IDSW"]) / max(1.0, num_gt)

        clear_agg = {
            "MOTA": mota,
            "MOTP": motp,
            "MODA": moda,
            "CLR_Re": clr_re,
            "CLR_Pr": clr_pr,
            "MTR": mtr,
            "PTR": ptr,
            "MLR": mlr,
            "sMOTA": smota,
            **totals,
            "MOTP_sum": motp_sum_total,
            "CLR_Frames": clr_frames_total,
        }

    return SequenceResult(
        sequence="COMBINED",
        CLEAR=CLEARMetrics.from_dict(clear_agg),
    )
