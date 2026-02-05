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
from typing import Literal

from trackers.eval.clear import aggregate_clear_metrics, compute_clear_metrics
from trackers.eval.hota import aggregate_hota_metrics, compute_hota_metrics
from trackers.eval.identity import aggregate_identity_metrics, compute_identity_metrics
from trackers.eval.io import load_mot_file, prepare_mot_sequence
from trackers.eval.results import (
    BenchmarkResult,
    CLEARMetrics,
    HOTAMetrics,
    IdentityMetrics,
    SequenceResult,
)

logger = logging.getLogger(__name__)

SUPPORTED_METRICS = ["CLEAR", "HOTA", "Identity"]


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
    """Evaluate a single multi-object tracking result against ground truth. Computes
    standard multi-object tracking metrics (CLEAR MOT, HOTA, Identity) for one sequence
    by matching predicted tracks to ground-truth tracks using per-frame IoU
    (Intersection over Union).

    !!! tip "TrackEval parity"

        This evaluation code is intentionally designed to match the core matching logic
        and metric calculations of
        [TrackEval](https://github.com/JonathonLuiten/TrackEval).

    Args:
        gt_path: Path to the ground-truth MOT file.
        tracker_path: Path to the tracker MOT file.
        metrics: Metric families to compute. Supported values are
            `["CLEAR", "HOTA", "Identity"]`. Defaults to `["CLEAR"]`.
        threshold: IoU threshold for `CLEAR` and `Identity` matching. Defaults
            to `0.5`. `HOTA` evaluates across multiple thresholds internally.

    Returns:
        `SequenceResult` with `CLEAR`, `HOTA`, and/or `Identity` populated based
            on `metrics`.

    Raises:
        FileNotFoundError: If `gt_path` or `tracker_path` does not exist.
        ValueError: If an unsupported metric family is requested.

    Examples:
        ```python
        >>> from trackers.eval import evaluate_mot_sequence  # doctest: +SKIP

        >>> result = evaluate_mot_sequence(  # doctest: +SKIP
        ...     gt_path="data/gt/MOT17-02/gt.txt",
        ...     tracker_path="data/trackers/MOT17-02.txt",
        ...     metrics=["CLEAR", "HOTA", "Identity"],
        ... )

        >>> print(result.CLEAR.MOTA)  # doctest: +SKIP
        0.756

        >>> print(result.table(columns=["MOTA", "HOTA", "IDF1", "IDSW"]))  # doctest: +SKIP
        Sequence                           MOTA    HOTA    IDF1  IDSW
        -------------------------------------------------------------
        MOT17-02                         75.600  62.300  72.100    42
        >>>
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
    clear_metrics: CLEARMetrics | None = None
    hota_metrics: HOTAMetrics | None = None
    identity_metrics: IdentityMetrics | None = None

    if "CLEAR" in metrics:
        clear_metrics_dict = compute_clear_metrics(
            seq_data.gt_ids,
            seq_data.tracker_ids,
            seq_data.similarity_scores,
            threshold=threshold,
        )
        clear_metrics = CLEARMetrics.from_dict(clear_metrics_dict)

    if "HOTA" in metrics:
        hota_dict = compute_hota_metrics(
            seq_data.gt_ids,
            seq_data.tracker_ids,
            seq_data.similarity_scores,
        )
        hota_metrics = HOTAMetrics.from_dict(hota_dict)

    if "Identity" in metrics:
        identity_dict = compute_identity_metrics(
            seq_data.gt_ids,
            seq_data.tracker_ids,
            seq_data.similarity_scores,
            threshold=threshold,
        )
        identity_metrics = IdentityMetrics.from_dict(identity_dict)

    # Build result
    return SequenceResult(
        sequence=gt_path.stem,
        CLEAR=clear_metrics,
        HOTA=hota_metrics,
        Identity=identity_metrics,
    )


def evaluate_mot_sequences(
    gt_dir: str | Path,
    tracker_dir: str | Path,
    seqmap: str | Path | None = None,
    metrics: list[str] | None = None,
    threshold: float = 0.5,
    benchmark: str | None = None,
    split: str | None = None,
    tracker_name: str | None = None,
) -> BenchmarkResult:
    """Evaluate multiple multi-object tracking results against ground truth. Computes
    standard multi-object tracking metrics (CLEAR MOT, HOTA, Identity) across one or
    more sequences by matching predicted tracks to ground-truth tracks using
    per-frame IoU (Intersection over Union). Returns both per-sequence and aggregated
    (combined) results.

    !!! tip "TrackEval parity"

        This evaluation code is intentionally designed to match the core matching logic
        and metric calculations of
        [TrackEval](https://github.com/JonathonLuiten/TrackEval).

    !!! tip "Supported dataset layouts"

        === "MOT layout"

            ```
            gt_dir/
            └── MOT17-train/
                ├── MOT17-02-FRCNN/
                │   └── gt/gt.txt
                ├── MOT17-04-FRCNN/
                │   └── gt/gt.txt
                ├── MOT17-05-FRCNN/
                │   └── gt/gt.txt
                └── ...

            tracker_dir/
            └── MOT17-train/
                └── ByteTrack/
                    └── data/
                        ├── MOT17-02-FRCNN.txt
                        ├── MOT17-04-FRCNN.txt
                        ├── MOT17-05-FRCNN.txt
                        └── ...
            ```

        === "Flat layout"

            ```
            gt_dir/
            ├── MOT17-02.txt
            ├── MOT17-04.txt
            ├── MOT17-05.txt
            └── ...

            tracker_dir/
            ├── MOT17-02.txt
            ├── MOT17-04.txt
            ├── MOT17-05.txt
            └── ...
            ```

    Args:
        gt_dir: Directory with ground-truth files.
        tracker_dir: Directory with tracker prediction files.
        seqmap: Optional sequence map. If provided, only those sequences are
            evaluated.
        metrics: Metric families to compute. Supported values are
            `["CLEAR", "HOTA", "Identity"]`. Defaults to `["CLEAR"]`.
        threshold: IoU threshold for `CLEAR` and `Identity`. Defaults to `0.5`.
        benchmark: Override auto-detected benchmark name (e.g., `"MOT17"`).
        split: Override auto-detected split name (e.g., `"train"`, `"val"`).
        tracker_name: Override auto-detected tracker name.

    Returns:
        `BenchmarkResult` with per-sequence results and a `COMBINED` aggregate.

    Raises:
        FileNotFoundError: If `gt_dir` or `tracker_dir` does not exist.
        ValueError: If auto-detection finds multiple valid options.

    Examples:
        Auto-detect layout and evaluate all sequences:

        ```python
        >>> from trackers.eval import evaluate_mot_sequences  # doctest: +SKIP

        >>> result = evaluate_mot_sequences(  # doctest: +SKIP
        ...     gt_dir="data/gt/",
        ...     tracker_dir="data/trackers/",
        ...     metrics=["CLEAR", "HOTA", "Identity"],
        ... )

        >>> print(result.table(columns=["MOTA", "HOTA", "IDF1", "IDSW"]))  # doctest: +SKIP
        Sequence                           MOTA    HOTA    IDF1  IDSW
        -------------------------------------------------------------
        sequence1                        74.800  60.900  71.200    37
        sequence2                        76.100  63.200  72.500    45
        -------------------------------------------------------------
        COMBINED                         75.450  62.050  71.850    82
        >>>
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

        seq_result = evaluate_mot_sequence(
            gt_path=gt_path,
            tracker_path=tracker_path,
            metrics=metrics,
            threshold=threshold,
        )
        # Fix sequence name (evaluate_mot_sequence uses file stem)
        sequence_results[seq_name] = SequenceResult(
            sequence=seq_name,
            CLEAR=seq_result.CLEAR,
            HOTA=seq_result.HOTA,
            Identity=seq_result.Identity,
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
        f"Multiple trackers found: {trackers}. Please specify --tracker-name."
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
    clear_agg: CLEARMetrics | None = None
    hota_agg: HOTAMetrics | None = None
    identity_agg: IdentityMetrics | None = None

    if "CLEAR" in metrics:
        clear_seq_metrics = [
            seq.CLEAR.to_dict()
            for seq in sequence_results.values()
            if seq.CLEAR is not None
        ]
        if clear_seq_metrics:
            clear_agg = CLEARMetrics.from_dict(
                aggregate_clear_metrics(clear_seq_metrics)
            )

    if "HOTA" in metrics:
        hota_seq_metrics = [
            seq.HOTA.to_dict(include_arrays=True, arrays_as_list=False)
            for seq in sequence_results.values()
            if seq.HOTA is not None
        ]
        if hota_seq_metrics:
            hota_agg = HOTAMetrics.from_dict(aggregate_hota_metrics(hota_seq_metrics))

    if "Identity" in metrics:
        identity_seq_metrics = [
            seq.Identity.to_dict()
            for seq in sequence_results.values()
            if seq.Identity is not None
        ]
        if identity_seq_metrics:
            identity_agg = IdentityMetrics.from_dict(
                aggregate_identity_metrics(identity_seq_metrics)
            )

    return SequenceResult(
        sequence="COMBINED",
        CLEAR=clear_agg,
        HOTA=hota_agg,
        Identity=identity_agg,
    )
