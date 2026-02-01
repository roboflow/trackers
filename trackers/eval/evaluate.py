# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from trackers.eval.clear import compute_clear_metrics
from trackers.eval.io import load_mot_file, prepare_mot_sequence

if TYPE_CHECKING:
    pass

SUPPORTED_METRICS = ["CLEAR"]


def evaluate_mot_sequence(
    gt_path: str | Path,
    tracker_path: str | Path,
    metrics: list[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
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
        Dictionary mapping metric names to their results. For CLEAR metrics,
            includes MOTA, MOTP, CLR_TP, CLR_FN, CLR_FP, IDSW, MT, PT, ML, MTR,
            PTR, MLR, and Frag.

    Raises:
        FileNotFoundError: If ground truth or tracker file does not exist.
        ValueError: If an unsupported metric is requested.

    Examples:
        ```python
        from trackers.eval import evaluate_mot_sequence

        results = evaluate_mot_sequence(
            gt_path="data/gt/MOT17-02/gt.txt",
            tracker_path="data/trackers/MOT17-02.txt",
        )
        results["CLEAR"]["MOTA"]
        # 0.756
        results["CLEAR"]["IDSW"]
        # 42
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

    # Load data
    gt_data = load_mot_file(gt_path)
    tracker_data = load_mot_file(tracker_path)

    # Prepare sequence (compute IoU, remap IDs)
    seq_data = prepare_mot_sequence(gt_data, tracker_data)

    # Compute metrics
    results: dict[str, dict[str, Any]] = {}

    if "CLEAR" in metrics:
        results["CLEAR"] = compute_clear_metrics(
            seq_data.gt_ids,
            seq_data.tracker_ids,
            seq_data.similarity_scores,
            threshold=threshold,
        )

    return results


def evaluate_benchmark(
    gt_dir: str | Path,
    tracker_dir: str | Path,
    seqmap: str | Path | None = None,
    metrics: list[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate tracker on multiple sequences (benchmark evaluation).

    Load sequences from directories and optionally filter by a sequence map file.
    Compute metrics for each sequence and aggregate results.

    Args:
        gt_dir: Directory containing ground truth files. Each file should be
            named `<sequence_name>.txt` in MOT format.
        tracker_dir: Directory containing tracker prediction files. Each file
            should be named `<sequence_name>.txt` in MOT format.
        seqmap: Optional path to sequence map file. If provided, only sequences
            listed in this file will be evaluated. Each line should contain a
            sequence name. Lines starting with `#` or `name` are ignored.
        metrics: List of metrics to compute. Defaults to `["CLEAR"]`.
        threshold: IoU threshold for matching. Defaults to 0.5.

    Returns:
        Dictionary with `sequences` (per-sequence results) and `aggregate`
            (combined metrics across all sequences) keys.

    Raises:
        FileNotFoundError: If gt_dir or tracker_dir does not exist.
        ValueError: If an unsupported metric is requested.

    Examples:
        ```python
        from trackers.eval import evaluate_benchmark

        results = evaluate_benchmark(
            gt_dir="data/gt/MOT17/",
            tracker_dir="data/trackers/MOT17/",
        )
        for seq, seq_results in results["sequences"].items():
            print(f"{seq}: MOTA={seq_results['CLEAR']['MOTA']:.1%}")
        print(f"Average MOTA: {results['aggregate']['CLEAR']['MOTA']:.1%}")
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

    # Get sequence list
    if seqmap is not None:
        sequences = _parse_seqmap(seqmap)
    else:
        # Auto-detect from GT directory
        sequences = sorted(
            p.stem for p in gt_dir.glob("*.txt") if not p.name.startswith(".")
        )

    if not sequences:
        raise ValueError(f"No sequences found in {gt_dir}")

    # Evaluate each sequence
    sequence_results: dict[str, dict[str, dict[str, Any]]] = {}

    for seq_name in sequences:
        gt_path = gt_dir / f"{seq_name}.txt"
        tracker_path = tracker_dir / f"{seq_name}.txt"

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

    # Compute aggregate metrics
    aggregate = _aggregate_metrics(sequence_results, metrics)

    return {
        "sequences": sequence_results,
        "aggregate": aggregate,
    }


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
    sequence_results: dict[str, dict[str, dict[str, Any]]],
    metrics: list[str],
) -> dict[str, dict[str, Any]]:
    """Aggregate metrics across sequences."""
    aggregate: dict[str, dict[str, Any]] = {}

    if "CLEAR" in metrics:
        # Sum integer metrics, then recompute ratios
        int_keys = ["CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag"]
        totals = {k: 0 for k in int_keys}

        motp_sum = 0.0
        total_tp = 0

        for seq_results in sequence_results.values():
            clear = seq_results["CLEAR"]
            for k in int_keys:
                totals[k] += clear[k]
            # Weight MOTP by TP count
            motp_sum += clear["MOTP"] * clear["CLR_TP"]
            total_tp += clear["CLR_TP"]

        # Compute aggregate ratios
        num_gt = totals["CLR_TP"] + totals["CLR_FN"]
        num_ids = totals["MT"] + totals["PT"] + totals["ML"]

        aggregate["CLEAR"] = {
            **totals,
            "MOTA": (totals["CLR_TP"] - totals["CLR_FP"] - totals["IDSW"])
            / max(1.0, num_gt),
            "MOTP": motp_sum / max(1.0, total_tp),
            "MTR": totals["MT"] / max(1.0, num_ids),
            "PTR": totals["PT"] / max(1.0, num_ids),
            "MLR": totals["ML"] / max(1.0, num_ids),
        }

    return aggregate
